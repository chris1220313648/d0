#!/usr/bin/env python3
"""Generate episode sidecars for the three local simulation datasets.

No parquet/video/task labels are changed. English motion descriptions use explicit
coordinate axes (GR1: base frame; Panda: world frame), not assumed forward/left.
Cosmos rotation source: NVlabs/cosmos-policy, regenerate_robocasa_dataset.py,
ee_states = [robot0_eef_pos, quat2axisangle(robot0_eef_quat)].
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import partial
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "data/robot_data"
ROOTS = {
    "gr1": BASE / "PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_v21",
    "xemb": BASE / "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim_v21",
    "cosmos": BASE / "RoboCasa-Cosmos-Policy_lerobot_v21",
}
VERSION = 1


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_rows(path, rows):
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def stat_key(path):
    s = Path(path).stat()
    return [str(Path(path).resolve()), s.st_size, s.st_mtime_ns]


def backup(path):
    path = Path(path)
    if path.exists():
        dest = path.with_name(path.name + f".bak-sim-lap-{time.time_ns()}")
        shutil.copy2(path, dest)


@contextmanager
def task_lock(root):
    with (root / "meta/.sim_lap.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def instruction(kind, row):
    text = row.get("remarks", "") if kind == "gr1" else (row.get("tasks") or [""])[0]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing full instruction: {row['episode_index']}")
    return text.strip()


def gr1_source(root, row):
    task, number = row["trajectory_id"].rsplit("-", 1)
    if Path(task).name != task:
        raise ValueError("Invalid trajectory_id")
    return root.parent.parent / "HDF5" / f"{task}.hdf5", f"demo_{int(number)}"


def parquet_path(root, info, row):
    i = int(row["episode_index"])
    return root / info["data_path"].format(episode_index=i, episode_chunk=i // int(info.get("chunks_size", 1000)))


def source_parquet(kind, root, info, row):
    path = parquet_path(root, info, row)
    if path.exists() or kind != "gr1":
        return path
    # A known v2.1 conversion gap can be read from the preserved original.
    original = root.parent.parent / "LeRobot" / root.name
    old_row = next((r for r in read_rows(original / "meta/episodes.jsonl") if r["episode_index"] == row["episode_index"]), None)
    keys = ("episode_index", "trajectory_id", "length", "remarks")
    if old_row is None or any(old_row.get(k) != row.get(k) for k in keys):
        raise ValueError("Original GR1 episode identity mismatch")
    old_info = json.loads((original / "meta/info.json").read_text())
    return parquet_path(original, old_info, row)


def array(table, key):
    a = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite {key}")
    return a


def validate_frames(table, row, fps, legacy=False):
    if len(table) != row["length"] or not np.all(array(table, "episode_index") == row["episode_index"]):
        raise ValueError("Parquet episode/frame alignment mismatch")
    if legacy:
        indices = array(table, "index").reshape(-1)
        timestamps = array(table, "timestamp").reshape(-1)
        if not np.all(np.diff(indices) == 1) or not np.allclose(timestamps, np.arange(len(table)) / fps, atol=1e-3, rtol=1e-5):
            raise ValueError("Legacy frame order/timestamps mismatch")
    elif not np.array_equal(array(table, "frame_index").reshape(-1), np.arange(len(table))):
        raise ValueError("Parquet frame_index mismatch")


def poses(kind, root, row, table, stack, handles):
    """Return [(label, xyz, rotation, optional finger aperture)], provenance."""
    if kind == "gr1":
        path, demo = gr1_source(root, row)
        if path not in handles:
            handles[path] = stack.enter_context(h5py.File(path, "r"))
            cfg = json.loads(handles[path]["data"].attrs["env_args"])["env_kwargs"]["controller_configs"]["composite_controller_specific_configs"]
            if (cfg["ik_input_type"], cfg["ik_input_ref_frame"], cfg["ik_input_rotation_repr"]) != ("absolute", "base", "axis_angle"):
                raise ValueError("Unsupported GR1 controller semantics")
            if cfg["ref_name"] != ["gripper0_right_grip_site", "gripper0_left_grip_site"]:
                raise ValueError("Unexpected GR1 arm order")
        a = handles[path][f"data/{demo}/actions"][:]
        q = array(table, "action")
        if a.shape != (len(q), 24) or q.shape[1] != 44 or not np.isfinite(a).all():
            raise ValueError("GR1 frame count/action shape mismatch")
        if not (np.allclose(q[:, 29:35], a[:, 12:18], atol=1e-6, rtol=0) and np.allclose(q[:, 7:13], a[:, 18:24], atol=1e-6, rtol=0)):
            raise ValueError("GR1 hand trajectories do not align")
        source = {"hdf5": stat_key(path), "demo": demo, "frame": "base", "motion": "absolute_target", "rotation": "axis_angle", "gripper_omitted": "multi_finger_mapping_not_verified"}
        return [("Left arm", a[:, 6:9], Rotation.from_rotvec(a[:, 9:12]), None),
                ("Right arm", a[:, :3], Rotation.from_rotvec(a[:, 3:6]), None)], source
    if kind == "xemb":
        s = array(table, "observation.state")
        if s.shape[1] != 53:
            raise ValueError("Expected 53D Panda state")
        for start in [3, 13, 17]:
            if not np.allclose(np.linalg.norm(s[:, start:start+4], axis=1), 1, atol=1e-3):
                raise ValueError("Invalid quaternion norm")
        base = Rotation.from_quat(s[:, 3:7])
        rot = Rotation.from_quat(s[:, 13:17])
        # This identity discriminates xyzw from wxyz without guessing.
        if not np.allclose(base.apply(s[:, 10:13]) + s[:, :3], s[:, 7:10], atol=1e-4):
            raise ValueError("Absolute/relative Panda position mismatch")
        if np.max(((base * Rotation.from_quat(s[:, 17:21])).inv() * rot).magnitude()) > 1e-3:
            raise ValueError("Absolute/relative Panda rotation mismatch")
        return [("Arm", s[:, 7:10], rot, np.abs(s[:, 21] - s[:, 22]))], {"frame": "world", "motion": "observed", "rotation": "xyzw", "gripper": "finger_aperture_change"}
    pos, ori = array(table, "observation.ee_pos"), array(table, "observation.ee_ori")
    state = array(table, "observation.state")
    ee = array(table, "observation.ee_states")
    if pos.shape[1] != 3 or ori.shape[1] != 3 or state.shape[1] != 9 or not np.allclose(ee, np.c_[pos, ori]):
        raise ValueError("Cosmos observation schema mismatch")
    return [("Arm", pos, Rotation.from_rotvec(ori), np.abs(state[:, 7] - state[:, 8]))], {"frame": "world", "motion": "observed", "rotation": "axis_angle", "gripper": "finger_aperture_change"}


def describe(arms, window):
    n = len(arms[0][1])
    end = np.minimum(np.arange(n) + window - 1, n - 1)
    parts = [[] for _ in range(n)]
    for label, pos, rot, aperture in arms:
        delta = (pos[end] - pos) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            angles = (rot[end] * rot.inv()).as_euler("xyz", degrees=True)
        for i in range(n):
            phrases = []
            for axis, value in zip("XYZ", delta[i]):
                amount = round(abs(float(value)), 1)
                if amount:
                    phrases.append(f"move {'+' if value > 0 else '-'}{axis} {amount:.1f} cm")
            for axis, value in zip("XYZ", angles[i]):
                amount = round(abs(float(value)))
                if amount:
                    phrases.append(f"rotate {amount} degrees about {'+' if value > 0 else '-'}{axis}")
            if aperture is not None:
                change = aperture[end[i]] - aperture[i]
                if abs(change) >= 0.002:
                    phrases.append("open gripper" if change > 0 else "close gripper")
            parts[i].append(f"{label}: {', '.join(phrases) if phrases else 'hold position'}.")
    return [" ".join(p) for p in parts]


def la_valid(path, count, expected_hash=None):
    try:
        data = path.read_bytes()
        lines = data.decode().splitlines()
        return len(lines) == count and all(x.strip() for x in lines) and (expected_hash is None or hashlib.sha256(data).hexdigest() == expected_hash)
    except (OSError, UnicodeError):
        return False


def selected_rows(root, maximum):
    rows = read_rows(root / "meta/episodes.jsonl")
    return rows, rows[:maximum] if maximum else rows


def language_task(job, keep_backup=True):
    kind, root_s, window, maximum = job
    root = Path(root_s)
    counts = Counter()
    errors = []
    with task_lock(root), ExitStack() as stack:
        info = json.loads((root / "meta/info.json").read_text())
        rows, selected = selected_rows(root, maximum)
        record_path = root / "meta/sim_lap_language.jsonl"
        records = {r["episode_index"]: r for r in read_rows(record_path)} if record_path.exists() else {}
        handles = {}
        for row in selected:
            i = int(row["episode_index"])
            try:
                p = source_parquet(kind, root, info, row)
                signature = {"version": VERSION, "window": window, "kind": kind, "parquet": stat_key(p)}
                if kind == "gr1":
                    hp, demo = gr1_source(root, row)
                    signature.update(hdf5=stat_key(hp), demo=demo)
                key = digest(signature)
                rel = f"language_action/episode_{i:06d}.txt"
                out = root / rel
                previous = records.get(i, {})
                if previous.get("input_hash") == key and la_valid(out, row["length"], previous.get("sha256")):
                    counts["reused"] += 1
                else:
                    columns = ["frame_index", "episode_index", "action"] if kind == "gr1" else ["frame_index", "episode_index", "observation.state"]
                    if kind == "cosmos":
                        columns += ["observation.ee_pos", "observation.ee_ori", "observation.ee_states"]
                    legacy = p != parquet_path(root, info, row) and "frame_index" not in pq.read_schema(p).names
                    if legacy:
                        columns.remove("frame_index")
                        columns += ["timestamp", "index"]
                    table = pq.read_table(p, columns=columns)
                    validate_frames(table, row, info["fps"], legacy)
                    arms, source = poses(kind, root, row, table, stack, handles)
                    text = "\n".join(describe(arms, window)) + "\n"
                    if keep_backup:
                        backup(out)
                    atomic_text(out, text)
                    records[i] = {"episode_index": i, "input_hash": key, "input": signature, "source": source, "sha256": hashlib.sha256(text.encode()).hexdigest(), "length": len(table)}
                    counts["generated"] += 1
                row["language_action_path"] = rel
                if kind == "gr1":
                    counts["gripper_omitted"] += 1
                if p != parquet_path(root, info, row):
                    counts["original_parquet_fallback"] += 1
            except Exception as exc:
                errors.append({"episode_index": i, "error": repr(exc)})
        backup(root / "meta/episodes.jsonl")
        write_rows(root / "meta/episodes.jsonl", rows)
        write_rows(record_path, list(records.values()))
        write_rows(root / "meta/sim_lap_language_failures.jsonl", errors)
    result = {"kind": kind, "root": str(root), **dict(counts), "failed": len(errors), "errors": errors[:3]}
    print(json.dumps(result), flush=True)
    return result


def valid_tensor(t):
    import torch
    return isinstance(t, torch.Tensor) and t.ndim == 2 and 0 < t.shape[0] <= 512 and t.shape[1] == 4096 and bool(torch.isfinite(t).all())


def save_tensor(path, tensor):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def install_tensor(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
    try:
        os.link(src, tmp)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def t5_stage(jobs, args):
    import torch
    # Keep CPU validation small; huge default thread pools hurt tiny tensors.
    torch.set_num_threads(2)
    sys.path.insert(0, str(PROJECT))
    from data.lerobot.add_t5_cache_to_lerobot_dataset import _init_wan_t5_encoder
    checkpoint = Path(args.wan_path) / "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = checkpoint.parent / "google/umt5-xxl"
    identity = {"checkpoint": stat_key(checkpoint), "tokenizer": [stat_key(p) for p in sorted(tokenizer.iterdir()) if p.is_file()], "text_len": 512, "dtype": "bfloat16", "version": VERSION}
    pool = args.output / "t5_pool" / digest(identity)
    texts = {}
    for kind, root_s, _, maximum in jobs:
        for row in selected_rows(Path(root_s), maximum)[1]:
            text = instruction(kind, row)
            texts[digest(text)] = text
    missing = []
    for key, text in texts.items():
        p = pool / (key + ".pt")
        try:
            ok = valid_tensor(torch.load(p, map_location="cpu", weights_only=True))
        except Exception:
            ok = False
        if not ok:
            missing.append((key, text))
    print(f"T5 unique={len(texts)} encode={len(missing)} device={args.device}", flush=True)
    if missing:
        encoder = _init_wan_t5_encoder(str(args.wan_path), args.device, 512)
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start:start + args.batch_size]
            with torch.inference_mode():
                embeddings = encoder([t for _, t in batch], args.device)
            for (key, _), embedding in zip(batch, embeddings):
                embedding = embedding.detach().cpu()
                if not valid_tensor(embedding):
                    raise ValueError("Invalid encoder output")
                save_tensor(pool / (key + ".pt"), embedding)
            print(f"T5 encoded {min(start + len(batch), len(missing))}/{len(missing)}", flush=True)
        del encoder
        torch.cuda.empty_cache()
    results = []
    for kind, root_s, _, maximum in jobs:
        root = Path(root_s)
        counts = Counter()
        with task_lock(root):
            rows, selected = selected_rows(root, maximum)
            record_path = root / "meta/sim_lap_t5.jsonl"
            records = {r["episode_index"]: r for r in read_rows(record_path)} if record_path.exists() else {}
            expected = {}
            for row in selected:
                i = int(row["episode_index"])
                text = instruction(kind, row)
                key = digest(text)
                src = pool / (key + ".pt")
                rel = f"t5_embedding/episode_{i:06d}.pt"
                dest = root / rel
                if key not in expected:
                    expected[key] = torch.load(src, map_location="cpu", weights_only=True)
                old_path = root / row.get("t5_embedding_path", rel)
                valid = False
                try:
                    old = torch.load(old_path, map_location="cpu", weights_only=True)
                    valid = valid_tensor(old) and old.shape == expected[key].shape and torch.allclose(old.float(), expected[key].float(), atol=0.02, rtol=0.01)
                except Exception:
                    pass
                if valid and old_path == dest:
                    counts["verified_reused"] += 1
                else:
                    if old_path.exists():
                        backup(old_path)
                        counts["rebuilt"] += 1
                    else:
                        counts["generated"] += 1
                    if dest != old_path:
                        backup(dest)
                    install_tensor(src, dest)
                row["t5_embedding_path"] = rel
                records[i] = {"episode_index": i, "instruction": text, "instruction_sha256": key, "encoder": digest(identity), "shape": list(expected[key].shape), "source": "remarks" if kind == "gr1" else "tasks[0]"}
            backup(root / "meta/episodes.jsonl")
            write_rows(root / "meta/episodes.jsonl", rows)
            write_rows(record_path, list(records.values()))
        result = {"kind": kind, "root": str(root), **dict(counts)}
        print(json.dumps(result), flush=True)
        results.append(result)
    atomic_text(args.output / "t5_encoder.json", json.dumps(identity, indent=2))
    return results


def verify(jobs, window):
    import torch
    torch.set_num_threads(2)
    results = []
    checked_tensors = set()
    for kind, root_s, _, maximum in jobs:
        root = Path(root_s)
        errors = []
        la_path, t5_path = root / "meta/sim_lap_language.jsonl", root / "meta/sim_lap_t5.jsonl"
        la = {x["episode_index"]: x for x in read_rows(la_path)} if la_path.exists() else {}
        t5 = {x["episode_index"]: x for x in read_rows(t5_path)} if t5_path.exists() else {}
        _, rows = selected_rows(root, maximum)
        for row in rows:
            i = row["episode_index"]
            try:
                record = la[i]
                if record["input"]["window"] != window or not la_valid(root / row["language_action_path"], row["length"], record["sha256"]):
                    raise ValueError("Language action validation failed")
                if t5[i]["instruction_sha256"] != digest(instruction(kind, row)):
                    raise ValueError("T5 instruction mismatch")
                p = root / row["t5_embedding_path"]
                s = p.stat()
                key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
                if key not in checked_tensors:
                    if not valid_tensor(torch.load(p, map_location="cpu", weights_only=True)):
                        raise ValueError("Invalid T5 tensor")
                    checked_tensors.add(key)
            except Exception as exc:
                errors.append({"episode_index": i, "error": repr(exc)})
        result = {"kind": kind, "root": str(root), "episodes": len(rows), "valid": len(rows) - len(errors), "failed": len(errors), "errors": errors}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "errors"}), flush=True)
    return results


def preflight(jobs):
    report = []
    for kind, root_s, _, maximum in jobs:
        root = Path(root_s)
        info = json.loads((root / "meta/info.json").read_text())
        all_rows, rows = selected_rows(root, maximum)
        indices = [r["episode_index"] for r in all_rows]
        if len(set(indices)) != len(indices) or len(all_rows) != info["total_episodes"]:
            raise ValueError(f"Episode metadata mismatch: {root}")
        fallbacks = []
        for row in rows:
            instruction(kind, row)
            source = source_parquet(kind, root, info, row)
            if not source.is_file() or row["length"] <= 0:
                raise ValueError(f"Missing/empty episode: {root} {row['episode_index']}")
            if source != parquet_path(root, info, row):
                fallbacks.append({"episode_index": row["episode_index"], "source": str(source)})
            if kind == "gr1" and not gr1_source(root, row)[0].is_file():
                raise ValueError("Missing GR1 HDF5")
        report.append({"kind": kind, "root": str(root), "episodes": len(rows), "frames": sum(r["length"] for r in rows), "episode_indices": [r["episode_index"] for r in rows], "original_parquet_fallbacks": fallbacks})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=list(ROOTS), default=list(ROOTS))
    p.add_argument("--stage", choices=["preflight", "language", "t5", "verify", "all"], default="all")
    p.add_argument("--window-size", type=int, default=17, help="Pose samples: 17 points cover 16 action steps")
    p.add_argument("--no-backup", action="store_true", help="Replace language text without keeping its old content")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Tasks per dataset; 0 = all")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--task-name", help="Process only this exact task directory name")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--wan-path", type=Path, default=PROJECT / "pretrained_models")
    p.add_argument("--output", type=Path, default=PROJECT / "outputs/sim_lap_h16")
    args = p.parse_args()
    if args.window_size < 1 or args.workers < 1 or args.batch_size < 1 or min(args.limit, args.max_episodes) < 0:
        p.error("Invalid size/limit")
    jobs = []
    for kind in args.datasets:
        roots = sorted(m.parent.parent for m in ROOTS[kind].glob("*/meta/info.json"))
        if args.task_name:
            roots = [r for r in roots if r.name == args.task_name]
        if not roots:
            raise ValueError(f"No datasets: {ROOTS[kind]}")
        jobs.extend((kind, str(r), args.window_size, args.max_episodes) for r in (roots[:args.limit] if args.limit else roots))
    inventory = preflight(jobs)
    print(json.dumps({"tasks": len(jobs), "episodes": sum(x["episodes"] for x in inventory), "frames": sum(x["frames"] for x in inventory)}), flush=True)
    if args.stage == "preflight":
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output / "input_scan.json", json.dumps(inventory, indent=2))
    report = {"window_size": args.window_size, "version": VERSION, "started": time.time()}
    if args.stage in ("all", "language"):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            report["language"] = list(pool.map(partial(language_task, keep_backup=not args.no_backup), jobs))
        atomic_text(args.output / "language_report.json", json.dumps(report["language"], indent=2))
    if args.stage in ("all", "t5"):
        report["t5"] = t5_stage(jobs, args)
        atomic_text(args.output / "t5_report.json", json.dumps(report["t5"], indent=2))
    if args.stage in ("all", "verify"):
        report["verification"] = verify(jobs, args.window_size)
    report["finished"] = time.time()
    atomic_text(args.output / f"{args.stage}_report.json", json.dumps(report, indent=2))
    failed = sum(x.get("failed", 0) for section in ("language", "verification") for x in report.get(section, []))
    print(f"DONE failed={failed} elapsed={report['finished']-report['started']:.1f}s", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
