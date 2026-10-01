#!/usr/bin/env python3
"""Source-preserving GR1 v2.1 conversion to native absolute EEF + hand actions.

Writes a separate dataset, never edits source parquet, videos or sidecars.
Task workers open one HDF5 at a time. Reruns verify existing output before reuse.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path('/root/nas/code/d0/data/robot_data/PhysicalAI-Robotics-GR00T-Teleop-Sim')
NAMES = ([f'{arm}_eef_{component}' for arm in ('right', 'left')
          for component in ('x', 'y', 'z', 'rotvec_x', 'rotvec_y', 'rotvec_z')]
         + [f'{arm}_hand_{i}' for arm in ('right', 'left') for i in range(6)])
ACTION_FEATURE = dict(dtype='float32', shape=[24], names=NAMES)
CONTRACT = dict(version=1, representation='absolute_target', reference_frame='base',
                position_units='meters', rotation='axis_angle_radians',
                order=['right_eef_pose_6', 'left_eef_pose_6', 'right_hand_6', 'left_hand_6'],
                state='unchanged 44D joint observations', names=NAMES)


def read_json(path):
    return json.loads(path.read_text())


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def link_verified(source, target, verify):
    if not source.is_file():
        raise FileNotFoundError(source)
    if not target.exists():
        if verify:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            tmp = target.with_name(target.name + '.tmp')
            shutil.copy2(source, tmp)
            tmp.replace(target)
    a, b = source.stat(), target.stat()
    if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
        return 'hardlink'
    if a.st_size != b.st_size or sha(source) != sha(target):
        raise ValueError(f'Output differs from source: {target}')
    return 'copy'


def validate_controller(handle):
    c = json.loads(handle['data'].attrs['env_args'])['env_kwargs']['controller_configs']['composite_controller_specific_configs']
    if (c['ik_input_type'], c['ik_input_ref_frame'], c['ik_input_rotation_repr']) != ('absolute', 'base', 'axis_angle'):
        raise ValueError('Unexpected GR1 controller semantics')
    if c['ref_name'] != ['gripper0_right_grip_site', 'gripper0_left_grip_site']:
        raise ValueError('Unexpected GR1 arm ordering')


def aligned_actions(raw, joints):
    if raw.shape != (len(joints), 24) or joints.shape != (len(raw), 44):
        raise ValueError('HDF5 / LeRobot action shape mismatch')
    if not np.isfinite(raw).all() or not np.isfinite(joints).all():
        raise ValueError('Nonfinite action')
    if not (np.allclose(joints[:, 29:35], raw[:, 12:18], atol=1e-6, rtol=0)
            and np.allclose(joints[:, 7:13], raw[:, 18:24], atol=1e-6, rtol=0)):
        raise ValueError('Hand trajectories do not align')
    return raw.astype(np.float32)


def action_stats(action):
    x = action.astype(np.float64)
    return dict(min=x.min(0).tolist(), max=x.max(0).tolist(), mean=x.mean(0).tolist(),
                std=x.std(0).tolist(), count=[len(x)])


def aggregate_action(stats):
    count = np.array([s['count'][0] for s in stats], dtype=np.float64)
    weights = count / count.sum()
    means = np.array([s['mean'] for s in stats])
    mean = (means * weights[:, None]).sum(0)
    variance = ((np.array([s['std'] for s in stats]) ** 2 + (means - mean) ** 2) * weights[:, None]).sum(0)
    return dict(min=np.min([s['min'] for s in stats], axis=0).tolist(),
                max=np.max([s['max'] for s in stats], axis=0).tolist(),
                mean=mean.tolist(), std=np.sqrt(variance).tolist(), count=[int(count.sum())])


def task_worker(job):
    source, target, hdf5_root, max_episodes, verify = job
    source, target, hdf5_root = map(Path, (source, target, hdf5_root))
    info = read_json(source / 'meta/info.json')
    manifest = source / 'meta/episodes.jsonl'
    episodes = [json.loads(line) for line in manifest.read_text().splitlines()]
    if max_episodes:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f'Empty task {source}')
    stats_by_id = {s['episode_index']: s for s in map(json.loads, (source / 'meta/episodes_stats.jsonl').read_text().splitlines())}
    task_names = {r['trajectory_id'].rsplit('-', 1)[0] for r in episodes}
    if len(task_names) != 1 or Path(next(iter(task_names))).name != next(iter(task_names)):
        raise ValueError('Unexpected trajectory task names')
    hp = hdf5_root / (next(iter(task_names)) + '.hdf5')
    fingerprints = dict(manifest_sha256=sha(manifest), hdf5_bytes=hp.stat().st_size,
                        hdf5_mtime_ns=hp.stat().st_mtime_ns)
    record_path = target / 'meta/eef24_conversion.json'
    if record_path.exists():
        prior = read_json(record_path)
        if prior['fingerprints'] != fingerprints or prior['contract'] != CONTRACT:
            raise ValueError(f'Source or contract changed: {target}')
    elif verify:
        raise FileNotFoundError(record_path)
    new_stats, provenance = [], []
    links = dict(hardlink=0, copy=0)
    offset = 0
    with h5py.File(hp, 'r') as h:
        validate_controller(h)
        for row in episodes:
            i, n = row['episode_index'], row['length']
            paths = dict(episode_chunk=i // info['chunks_size'], episode_index=i)
            rel = info['data_path'].format(**paths)
            table = pq.read_table(source / rel)
            if len(table) != n or n <= 0:
                raise ValueError(f'Frame count mismatch: {rel}')
            for key, expected in [('episode_index', np.full(n, i)), ('frame_index', np.arange(n)), ('index', np.arange(offset, offset + n))]:
                if not np.array_equal(table[key].to_numpy(), expected):
                    raise ValueError(f'{key} mismatch: {rel}')
            times = table['timestamp'].to_numpy()
            if not np.allclose(times, np.arange(n) / info['fps'], atol=1e-4, rtol=0):
                raise ValueError(f'Timestamp mismatch: {rel}')
            _, number = row['trajectory_id'].rsplit('-', 1)
            demo = f'demo_{int(number)}'
            raw = h[f'data/{demo}/actions'][:]
            actions = aligned_actions(raw, np.asarray(table['action'].to_pylist()))
            updated = table.set_column(table.column_names.index('action'), 'action',
                                       pa.array(actions.tolist(), type=pa.list_(pa.float32())))
            dest = target / rel
            if not dest.exists():
                if verify:
                    raise FileNotFoundError(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + '.tmp')
                pq.write_table(updated, tmp)
                tmp.replace(dest)
            if not pq.read_table(dest).equals(updated):
                raise ValueError(f'Parquet verification failed: {dest}')
            for key, feature in info['features'].items():
                if feature['dtype'] == 'video':
                    video_rel = info['video_path'].format(**paths, video_key=key)
                    links[link_verified(source / video_rel, target / video_rel, verify)] += 1
            for key in ('language_action_path', 't5_embedding_path'):
                cache_rel = row[key]
                if Path(cache_rel).is_absolute() or '..' in Path(cache_rel).parts:
                    raise ValueError(f'Unsafe cache path {cache_rel}')
                links[link_verified(source / cache_rel, target / cache_rel, verify)] += 1
            lines = (target / row['language_action_path']).read_text().splitlines()
            if len(lines) != n or not all(line.strip() for line in lines) or not row['remarks'].strip():
                raise ValueError(f'Invalid LAP or instruction: {rel}')
            stat = copy.deepcopy(stats_by_id[i])
            stat['stats']['action'] = action_stats(actions)
            new_stats.append(stat)
            provenance.append(dict(episode_index=i, trajectory_id=row['trajectory_id'], demo=demo,
                                   source_parquet_sha256=sha(source / rel), output_parquet_sha256=sha(dest)))
            offset += n
            if (len(new_stats) % 200) == 0:
                print(f'{source.name}: {len(new_stats)}/{len(episodes)} verified', flush=True)
    new_info = copy.deepcopy(info)
    new_info['features']['action'] = ACTION_FEATURE
    new_info.update(total_episodes=len(episodes), total_frames=offset,
                    total_videos=len(episodes) * sum(f['dtype'] == 'video' for f in info['features'].values()),
                    total_chunks=(len(episodes) + info['chunks_size'] - 1) // info['chunks_size'])
    if max_episodes:
        new_info['splits'] = {'train': f'0:{len(episodes)}'}
    global_stats = read_json(source / 'meta/stats.json')
    if max_episodes:
        # Smoke output also has correct scalar/vector/video statistics.
        from lerobot.datasets.compute_stats import aggregate_stats
        aggregate = aggregate_stats([{k: {f: np.asarray(v) for f, v in s.items()}
                                      for k, s in r['stats'].items()} for r in new_stats])
        global_stats = {k: {f: np.asarray(v).tolist() for f, v in s.items()} for k, s in aggregate.items()}
    global_stats['action'] = aggregate_action([r['stats']['action'] for r in new_stats])
    outputs = {'info.json': new_info, 'stats.json': global_stats}
    modality_path = source / 'meta/modality.json'
    if modality_path.exists():
        modality = read_json(modality_path)
        modality['action'] = {
            'right_eef_position': dict(start=0, end=3, absolute=True),
            'right_eef_rotation': dict(start=3, end=6, absolute=True, rotation_type='axis_angle'),
            'left_eef_position': dict(start=6, end=9, absolute=True),
            'left_eef_rotation': dict(start=9, end=12, absolute=True, rotation_type='axis_angle'),
            'right_hand': dict(start=12, end=18), 'left_hand': dict(start=18, end=24)}
        outputs['modality.json'] = modality
    for name, value in outputs.items():
        out = target / 'meta' / name
        if verify:
            if read_json(out) != value:
                raise ValueError(f'Metadata mismatch: {out}')
        else:
            write_json(out, value)
    for name, rows in [('episodes.jsonl', episodes), ('episodes_stats.jsonl', new_stats), ('eef24_sources.jsonl', provenance)]:
        data = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
        out = target / 'meta' / name
        if verify:
            if out.read_text() != data:
                raise ValueError(f'Manifest mismatch: {out}')
        else:
            atomic_text(out, data)
    for name in ('tasks.jsonl', 'embodiment.json', 'excluded_source_episodes.json'):
        if (source / 'meta' / name).exists():
            # Metadata is independently copied, unlike immutable video/cache blobs.
            src, dst = source / 'meta' / name, target / 'meta' / name
            if verify:
                if dst.read_bytes() != src.read_bytes():
                    raise ValueError(f'Metadata mismatch: {dst}')
            else:
                atomic_text(dst, src.read_text())
    result = dict(status='complete', source=str(source), output=str(target), hdf5=str(hp),
                  episodes=len(episodes), frames=offset, links=links, contract=CONTRACT,
                  fingerprints=fingerprints, all_parquet_columns_verified=True,
                  cache_provenance='Immutable source LAP/T5 reused: same frames, remarks and HDF5 target poses; hand descriptions remain omitted.',
                  source_lap_provenance=str(source / 'meta/sim_lap_language.jsonl'))
    if not verify:
        write_json(record_path, result)
        write_json(target / 'COMPLETE.json', result)
    elif read_json(target / 'COMPLETE.json') != result:
        raise ValueError(f'Completion marker mismatch: {target}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BASE / 'LeRobot_v21')
    parser.add_argument('--output', type=Path, default=BASE / 'LeRobot_v21_eef24')
    parser.add_argument('--hdf5-root', type=Path, default=BASE / 'HDF5')
    parser.add_argument('--report', type=Path, default=Path('outputs/gr1_eef24_conversion/summary.json'))
    parser.add_argument('--tasks', default='*')
    parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.source.resolve() in args.output.resolve().parents:
        raise ValueError('Output must be separate from source')
    tasks = sorted(args.source.glob(f'{args.tasks}/meta/info.json'))
    if not tasks:
        raise ValueError('No tasks')
    jobs = [(str(p.parent.parent), str(args.output / p.parent.parent.name), str(args.hdf5_root), args.max_episodes, args.verify) for p in tasks]
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"COMPLETE {result['output']}: {result['episodes']} episodes", flush=True)
            except Exception as exc:
                failures.append(dict(task=futures[future], error=repr(exc)))
                print(f'FAILED {failures[-1]}', flush=True)
            write_json(args.report, dict(status='running', tasks=results, failures=failures))
    result = dict(status='failed' if failures else 'complete', verify_only=args.verify,
                  episodes=sum(r['episodes'] for r in results), frames=sum(r['frames'] for r in results),
                  tasks=sorted(results, key=lambda r: r['source']), failures=failures)
    write_json(args.report, result)
    if failures:
        raise SystemExit(1)
    print(json.dumps({k:v for k,v in result.items() if k != 'tasks'}), flush=True)


if __name__ == '__main__':
    main()
