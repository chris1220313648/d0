#!/usr/bin/env python3
"""Reproducible LAP coverage and checkpoint representation analysis.

No training imports or full dataset construction. See the adjacent README.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = Path('/root/nasbak/cjy/robot_raw/motus_in_depth_analysis')
DEFAULT_OUTPUT = ROOT / 'outputs/in_depth_analysis'
VLM_CONFIG = Path('/root/nas/xicheng/qwen3vl_2b_repro/outputs/qwen3vl2b_full_40pct_8gpu_nframes16/checkpoint-74056')
CHECKPOINTS = {
    'without_lap': ROOT / 'checkpoints/multidataset_lap_v0/multidataset_lap_v0_lap_hypertrain/checkpoint_step_400000/pytorch_model_0.bin',
    'with_lap': ROOT / 'checkpoints/multidataset_lap_v4_14a/multidataset_lap_v4_14a_lap_hypertrain/checkpoint_step_200000/pytorch_model_0.bin',
}
EGO_ROOT = Path('/root/nasbak2/shuai/ready_data_egoverse/processed/egoverse_active_wrist')
PROMPT = 'Task: {instruction}\nPlease provide a language description of the next action.'
CAMERAS = ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right']
PROTOCOL = 'lap-analysis-v1'


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_id(path):
    p = Path(path).resolve()
    st = p.stat()
    return dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def normalize_lap(text):
    # Preserve sidedness, quantities, units, and gripper operations.
    return re.sub(r'\b(wrist|arm|hand)\b', 'effector', text, flags=re.I).lower().strip()


def clean_instruction(text):
    text = re.sub(r'<setup_start>.*?<setup_end>', '', text, flags=re.S)
    text = re.sub(r'<control_start>.*?<control_end>', '', text, flags=re.S)
    return text.strip()


def prompt_for(row):
    return PROMPT.format(instruction=clean_instruction(row['instruction']))


def letterbox(frame, height=384, width=320):
    import cv2
    import numpy as np
    scale = min(height / frame.shape[0], width / frame.shape[1])
    h, w = max(1, int(frame.shape[0] * scale)), max(1, int(frame.shape[1] * scale))
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w] = cv2.resize(frame, (w, h))
    return result


def decode_frame(path, index):
    # Exact frame ordinal: PyAV's sequential decoder avoids approximate seeking.
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for i, frame in enumerate(container.decode(stream)):
            if i == index:
                return frame.to_ndarray(format='rgb24')
    raise ValueError(f'frame {index} outside {path}')


def observation(row):
    import cv2
    import numpy as np
    frames = [decode_frame(p, row['frame_index']) for p in row['video_paths']]
    if len(frames) == 3:
        top, left, right = frames
        width = max(top.shape[1], left.shape[1] + right.shape[1])
        bottom_h = max(left.shape[0], right.shape[0])
        upper = cv2.resize(top, (width, top.shape[0]))
        lower = np.concatenate([cv2.resize(left, (width//2, bottom_h)),
                                cv2.resize(right, (width-width//2, bottom_h))], axis=1)
        frame = np.concatenate([upper, lower], axis=0)
    else:
        frame = frames[0]
    return letterbox(frame)


def materialize(row, cache):
    from PIL import Image
    lines = Path(row['lap_path']).read_text().splitlines()
    if row['source'] == 'egoverse':
        # Reproduce the sidecar generator's parquet-row selection. In particular,
        # a sparse frame_index column is not a contiguous video-frame offset.
        import pyarrow.parquet as pq
        import numpy as np
        parquet = pq.ParquetFile(row['data_parquet'])
        columns = ['frame_index']
        if 'episode_index' in parquet.schema_arrow.names:
            columns.append('episode_index')
        table = parquet.read(columns=columns)
        frames = np.asarray(table['frame_index']).reshape(-1)
        filtered = False
        if 'episode_index' in columns:
            mask = np.asarray(table['episode_index']).reshape(-1) == row['episode_index']
            if mask.any():
                frames, filtered = frames[mask], True
        start, end = row.get('dataset_from_index'), row.get('dataset_to_index')
        if start is not None and end is not None and not filtered:
            frames = frames[max(0, int(start)):min(len(frames), int(end))]
        selected = frames[(frames >= row['segment_start']) & (frames < row['segment_end'])]
        if not len(selected) and start is not None and end is not None:
            selected = frames[max(0, row['segment_start']):min(len(frames), row['segment_end'])]
        if len(lines) != len(selected):
            raise ValueError(f'Ego sidecar/parquet length mismatch: {len(lines)} vs {len(selected)}')
        matches = np.flatnonzero(selected == row['frame_index'])
        if len(matches) != 1:
            raise ValueError('Observation frame does not uniquely match a sidecar row')
        row['lap_index'] = int(matches[0])
    idx = row['lap_index']
    if idx < 0 or idx >= len(lines) or not lines[idx].strip():
        raise ValueError(f'Invalid LAP line {idx}/{len(lines)}: {row["lap_path"]}')
    row['lap_text'] = lines[idx].strip()
    row['lap_normalized'] = normalize_lap(row['lap_text'])
    if not row['instruction'].strip():
        raise ValueError('Empty task instruction')
    row['source_files'] = [file_id(p) for p in [row['lap_path'], *row['video_paths']]]
    image_path = cache / 'images' / (digest(row) + '.png')
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        Image.fromarray(observation(row)).save(image_path)
    row['image_path'] = str(image_path)
    row['image_sha256'] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    row['id'] = f'{row["source"]}:{row["episode_key"]}:{row["frame_index"]}'
    return row


def bridge_candidates(source, folder, seed):
    """Task round-robin, held-out split first; files are discovered lazily."""
    rng = random.Random(seed)
    root = ROOT / 'data/robot_data' / folder
    for split in ['test', 'val', 'train']:
        if not (root / split).exists():
            continue
        groups = []
        for task in sorted((root / split).iterdir()):
            if (task / 'language_action').is_dir():
                paths = sorted((task / 'language_action').glob('*.txt'))
                rng.shuffle(paths)
                groups.append((task, paths))
        rng.shuffle(groups)
        while groups:
            remaining = []
            for task, paths in groups:
                if not paths:
                    continue
                p = paths.pop()
                remaining.append((task, paths))
                instruction = task / 'instructions' / p.name
                video = task / 'videos' / (p.stem + '.mp4')
                if not instruction.exists() or not video.exists():
                    continue
                lines = p.read_text().splitlines()
                if not lines:
                    continue
                idx = rng.randrange(max(1, len(lines)-16))
                yield dict(source=source, domain='robot', task=task.name, split=split,
                           episode_key=f'{source}/{task.name}/{p.stem}', frame_index=idx,
                           lap_index=idx, lap_path=str(p.resolve()),
                           video_paths=[str(video.resolve())], instruction=instruction.read_text().strip())
            groups = remaining


def agibot_candidates(seed):
    rng = random.Random(seed)
    root = ROOT / 'data/robot_data/AgiBotWorld2026'
    tasks = defaultdict(list)
    for base, dirs, _ in os.walk(root):
        dirs.sort()
        p = Path(base)
        if 'meta' in dirs and p.name.endswith('_split'):
            tasks[str(p.parent.relative_to(root))].append(p)
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                       ['data', 'videos', 'language_action', 't5_embedding', '__pycache__', 'meta']]
    log(f'AgiBot discovery: {len(tasks)} tasks, {sum(map(len, tasks.values()))} repos')

    def per_task(task, repos):
        rng.shuffle(repos)
        for p in repos:
            info = json.loads((p / 'meta/info.json').read_text())
            episodes = list(read_jsonl(p / 'meta/episodes.jsonl'))
            rng.shuffle(episodes)
            for ep in episodes:
                eid = int(ep['episode_index'])
                idx = rng.randrange(max(1, int(ep['length'])-16))
                lap = Path(ep.get('language_action_path', f'language_action/episode_{eid:06d}.txt'))
                lap = lap if lap.is_absolute() else p / lap
                videos = [p / info['video_path'].format(episode_index=eid,
                          episode_chunk=eid//int(info.get('chunks_size', 1000)), video_key=k) for k in CAMERAS]
                if not lap.exists() or not all(v.exists() for v in videos):
                    continue
                instr = ep.get('tasks', [''])[0]
                for segment in info.get('instruction_segments', {}).get(str(eid), []):
                    if segment.get('start_frame_index', 0) <= idx <= segment.get('end_frame_index', ep['length']):
                        instr = segment.get('instruction', instr)
                        break
                # h5_path identifies the parent recording even after segmentation.
                original = info.get('h5_path', {}).get(str(eid), f'{p.relative_to(root)}/{eid}')
                yield dict(source='agibot', domain='robot', task=task, split='train',
                           episode_key=f'agibot/{original}', frame_index=idx, lap_index=idx,
                           lap_path=str(lap.resolve()), video_paths=[str(v.resolve()) for v in videos], instruction=instr)

    names = sorted(tasks)
    rng.shuffle(names)
    streams = [iter(per_task(t, tasks[t])) for t in names]
    while streams:
        live = []
        for stream in streams:
            try:
                yield next(stream)
                live.append(stream)
            except StopIteration:
                pass
        streams = live


def ego_candidates(seed, per_task_limit=64):
    rng = random.Random(seed)
    for split in ['val', 'train']:
        # Bounded per-task reservoir, never construct a LeRobot/HF dataset.
        groups, counts = defaultdict(list), Counter()
        for row in read_jsonl(EGO_ROOT / 'manifests' / f'{split}.jsonl'):
            task = row.get('task_key', row['task'])
            counts[task] += 1
            bucket = groups[task]
            if len(bucket) < per_task_limit:
                bucket.append(row)
            else:
                index = rng.randrange(counts[task])
                if index < per_task_limit:
                    bucket[index] = row
        tasks = sorted(groups)
        rng.shuffle(tasks)
        for task in tasks:
            rng.shuffle(groups[task])
        while tasks:
            live = []
            for task in tasks:
                if not groups[task]:
                    continue
                r = groups[task].pop()
                live.append(task)
                lap = EGO_ROOT / 'language_action' / split / f'{r["id"]}.txt'
                if not lap.exists():
                    continue
                size = int(r['end_frame']) - int(r['start_frame'])
                local = rng.randrange(max(1, size-64))
                frame = int(r['start_frame']) + local
                if r.get('decode_frames'):
                    frames = r['decode_frames']
                    frame = int(frames[rng.randrange(max(1, len(frames)-64))])
                yield dict(source='egoverse', domain='ego', task=task, split=split,
                           episode_key=r.get('episode_key', r['sample_id']), frame_index=frame,
                           lap_index=local, lap_path=str(lap), video_paths=[r['video_path']],
                           instruction=r['instruction'], segment_id=r['id'], segment_start=int(r['start_frame']),
                           segment_end=int(r['end_frame']), data_parquet=r['data_parquet'], episode_index=r['episode_index'],
                           dataset_from_index=r.get('dataset_from_index'), dataset_to_index=r.get('dataset_to_index'))
            tasks = live


def sample(args):
    if args.per_domain < 4:
        raise ValueError('--per-domain must be at least four')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'samples.jsonl'
    if manifest.exists():
        raise FileExistsError(f'{manifest}; use a separate output directory to resample')
    rows, errors, seen = [], [], set()
    sources = [('agibot', agibot_candidates(args.seed)),
               ('droid', bridge_candidates('droid', 'droid_dataset', args.seed)),
               ('fractal', bridge_candidates('fractal', 'fractal', args.seed)),
               ('bridge', bridge_candidates('bridge', 'bridge_dataset', args.seed)),
               ('egoverse', ego_candidates(args.seed))]
    for source_index, (source, stream) in enumerate(sources):
        quota = args.per_domain if source == 'egoverse' else args.per_domain//4 + int(source_index < args.per_domain % 4)
        taken = 0
        for row in stream:
            if row['episode_key'] in seen:
                continue
            try:
                row = materialize(row, args.cache)
            except Exception as exc:
                errors.append(dict(source=source, episode=row['episode_key'], error=str(exc)))
                log(f'skip {source}: {exc}')
                continue
            rows.append(row)
            seen.add(row['episode_key'])
            taken += 1
            if taken % 25 == 0 or taken == quota:
                log(f'sample {source}: {taken}/{quota}')
            if taken == quota:
                break
        if taken != quota:
            save_json(out / 'sample_errors.json', errors)
            raise RuntimeError(f'Insufficient {source} episodes: {taken}/{quota}; no silent rebalancing')
    tmp = manifest.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tmp.replace(manifest)
    save_json(out / 'sampling.json', dict(protocol=PROTOCOL, seed=args.seed, per_domain=args.per_domain,
              manifest_hash=digest(rows), counts=dict(Counter(f'{r["source"]}/{r["split"]}' for r in rows)),
              errors=errors, caveat='Existing split names do not establish checkpoint training exclusion.'))


def load_t5(device):
    """Load only vendored T5 and tokenizer modules, avoiding WAN pipeline imports."""
    import torch
    name = '_analysis_wan'
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / 'bak/wan/modules')]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + '.t5', ROOT / 'bak/wan/modules/t5.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    path = ROOT / 'pretrained_models/Wan2.2-TI2V-5B'
    return module.T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device,
            checkpoint_path=str(path / 'models_t5_umt5-xxl-enc-bf16.pth'), tokenizer_path=str(path / 'google/umt5-xxl'))


def load_vlm(checkpoint, device):
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.modeling_utils import no_init_weights
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    cfg = AutoConfig.from_pretrained(VLM_CONFIG, local_files_only=True)
    cfg._attn_implementation = 'sdpa'
    # Keep derived, non-persistent RoPE buffers real. Meta construction would
    # leave these absent from the state dict and impossible to move to CUDA.
    with no_init_weights():
        model = Qwen3VLForConditionalGeneration(cfg)
    with torch.device('meta'):
        adapter = torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.SiLU(),
                    torch.nn.Linear(512, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512))
    vlm = {k.removeprefix('vlm_model.'): v for k, v in state.items() if k.startswith('vlm_model.')}
    projection = {k.removeprefix('und_expert.vlm_adapter.'): v for k, v in state.items() if k.startswith('und_expert.vlm_adapter.')}
    model.load_state_dict(vlm, strict=True, assign=True)
    adapter.load_state_dict(projection, strict=True, assign=True)
    log(f'strict checkpoint load: {len(vlm)} VLM keys, {len(projection)} adapter keys')
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    adapter.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    del state, vlm, projection
    processor = AutoProcessor.from_pretrained(VLM_CONFIG, local_files_only=True)
    return model, adapter, processor


def extract(args):
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rows = list(read_jsonl(args.output / 'samples.jsonl'))
    model_name = args.model
    identity = dict(protocol=PROTOCOL, code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    manifest=digest(rows), model=model_name, prompt=PROMPT, pooling='last-valid-prompt-token',
                    transformers=__import__('transformers').__version__, torch=torch.__version__)
    if model_name == 'text':
        identity.update(encoder=file_id(ROOT / 'pretrained_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth'),
                        pooling='mean-valid-tokens-including-EOS', text_len=512)
    else:
        identity['checkpoint'] = file_id(CHECKPOINTS[model_name])
        identity['vlm_config'] = json.loads((VLM_CONFIG / 'config.json').read_text())
    cache = args.cache / 'features' / digest(identity)
    cache.mkdir(parents=True, exist_ok=True)
    save_json(cache / 'identity.json', identity)
    save_json(args.output / f'{model_name}_cache.json', dict(cache=str(cache), identity=identity))
    needed = [i for i in range(len(rows)) if not (cache / f'{i:06d}.npz').exists()]
    if needed:
        log(f'extract {model_name}: {len(needed)}/{len(rows)} uncached')
        if model_name == 'text':
            encoder = load_t5(args.device)
        else:
            model, adapter, processor = load_vlm(CHECKPOINTS[model_name], args.device)
        for i in needed:
            row = rows[i]
            with torch.inference_mode():
                if model_name == 'text':
                    texts = [row['lap_text'], row['lap_normalized']]
                    # Refuse truncation rather than silently lose motion clauses.
                    lengths = encoder.tokenizer.tokenizer(texts, add_special_tokens=True, truncation=False)['input_ids']
                    if any(len(ids) > 512 for ids in lengths):
                        raise ValueError(f'LAP text exceeds 512 tokens: {row["id"]}')
                    values = encoder(texts, args.device)
                    data = {name: val.float().mean(0).cpu().numpy() for name, val in zip(['raw', 'normalized'], values)}
                else:
                    if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() != row['image_sha256']:
                        raise ValueError('Cached observation image changed after sampling')
                    image = Image.open(row['image_path']).convert('RGB')
                    messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=prompt_for(row))])]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors='pt')
                    if 'labels' in inputs:
                        raise AssertionError('Analysis must not contain supervised answers')
                    input_hash = hashlib.sha256()
                    for key in sorted(inputs):
                        value = inputs[key].detach().cpu().contiguous()
                        input_hash.update(key.encode())
                        input_hash.update(value.numpy().tobytes())
                    inputs = inputs.to(args.device)
                    hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                    last = int(inputs['attention_mask'][0].nonzero()[-1])
                    raw = hidden[0, last]
                    data = dict(vlm=raw.float().cpu().numpy(), adapter=adapter(raw).float().cpu().numpy(),
                                input_hash=np.array(input_hash.hexdigest()))
                data['id'] = np.array(row['id'])
                for name, value in data.items():
                    if value.dtype.kind == 'f' and (not np.isfinite(value).all() or np.linalg.norm(value) == 0):
                        raise ValueError(f'Invalid feature {row["id"]}/{name}')
                tmp = cache / f'{i:06d}.tmp.npz'
                np.savez(tmp, **data)
                tmp.replace(cache / f'{i:06d}.npz')
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                log(f'{model_name}: {i+1}/{len(rows)}')
    shards = []
    for i, row in enumerate(rows):
        with np.load(cache / f'{i:06d}.npz', allow_pickle=False) as shard:
            data = dict(shard)
        if data['id'].item() != row['id']:
            raise ValueError('Feature/sample identity mismatch')
        shards.append(data)
    merged = {key: np.stack([s[key] for s in shards]) for key in shards[0]}
    feature_path = cache / 'features.npz'
    np.savez(feature_path, **merged)
    save_json(args.output / f'{model_name}_complete.json', dict(features=str(feature_path), count=len(rows), identity=identity))
    log(f'COMPLETE {model_name}: {len(rows)} samples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['sample', 'extract', 'analyze', 'report'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--per-domain', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', choices=['text', 'without_lap', 'with_lap'], default='text')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.cache / 'python_deps'))
    if args.stage in ['analyze', 'report']:
        from analysis_report import analyze, report
        {'analyze': analyze, 'report': report}[args.stage](args)
    else:
        {'sample': sample, 'extract': extract}[args.stage](args)


if __name__ == '__main__':
    main()
