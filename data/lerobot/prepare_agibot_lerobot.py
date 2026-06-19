#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def load_jsonlines(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonlines_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def find_archives(root: Path, include_temp: bool) -> List[Path]:
    archives: List[Path] = []
    for path in root.rglob("*.tar.gz"):
        if not include_temp and any(part.startswith("._____") for part in path.parts):
            continue
        archives.append(path)
    return sorted(archives)


def archive_output_root(archive: Path) -> Path:
    name = archive.name
    if not name.endswith(".tar.gz"):
        raise ValueError(f"Not a .tar.gz archive: {archive}")
    return archive.with_name(name[: -len(".tar.gz")])


def extract_archive(archive: Path, overwrite_extract: bool) -> Path:
    out_root = archive_output_root(archive)
    marker = out_root / ".extract_complete"
    if marker.exists() and not overwrite_extract:
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tar",
        "-xzf",
        str(archive),
        "--strip-components=1",
        "-C",
        str(out_root),
    ]
    print(f"[extract] {archive} -> {out_root}", flush=True)
    subprocess.run(cmd, check=True)
    marker.write_text("ok\n", encoding="utf-8")
    return out_root


def extract_archives(archives: List[Path], overwrite_extract: bool, workers: int) -> None:
    if workers <= 1:
        for idx, archive in enumerate(archives, start=1):
            print(f"[{idx}/{len(archives)}] archive={archive}", flush=True)
            extract_archive(archive, overwrite_extract=overwrite_extract)
        return

    print(f"extract_workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_archive = {
            executor.submit(extract_archive, archive, overwrite_extract): archive
            for archive in archives
        }
        done = 0
        for future in as_completed(future_to_archive):
            archive = future_to_archive[future]
            done += 1
            try:
                out_root = future.result()
            except Exception as err:
                print(f"[extract failed] {archive}: {err}", flush=True)
                raise
            print(f"[extract done {done}/{len(archives)}] {archive} -> {out_root}", flush=True)


def episode_instruction(row: Dict[str, Any]) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], str):
        return tasks[0]
    task = row.get("task", "")
    return task if isinstance(task, str) else str(task)


def init_t5_encoder(wan_path: str, device: str, text_len: int) -> Any:
    bak_root = "/root/nas/code/d0/bak"
    if bak_root not in sys.path:
        sys.path.insert(0, bak_root)
    from wan.modules.t5 import T5EncoderModel  # type: ignore

    ckpt = Path(wan_path) / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth"
    tok = Path(wan_path) / "Wan2.2-TI2V-5B" / "google" / "umt5-xxl"
    if not ckpt.exists():
        raise FileNotFoundError(f"T5 checkpoint not found: {ckpt}")
    if not tok.exists():
        raise FileNotFoundError(f"T5 tokenizer not found: {tok}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=int(text_len),
        dtype=dtype,
        device=device,
        checkpoint_path=str(ckpt),
        tokenizer_path=str(tok),
    )


def encode_t5(encoder: Any, instruction: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        out = encoder([instruction], device)
    if isinstance(out, list):
        emb = out[0]
    elif isinstance(out, torch.Tensor):
        emb = out
    else:
        raise TypeError(f"Unexpected encoder output: {type(out)}")
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.detach().cpu()


def ensure_t5_cache(
    dataset_root: Path,
    encoder: Any,
    instruction_cache: Dict[str, torch.Tensor],
    device: str,
    folder_name: str,
    overwrite: bool,
) -> Tuple[int, int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = load_jsonlines(episodes_path)
    out_dir = dataset_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped = 0
    for row in episodes:
        ep_idx = int(row["episode_index"])
        rel = f"{folder_name}/episode_{ep_idx:06d}.pt"
        abs_pt = dataset_root / rel
        ptr_ok = isinstance(row.get("t5_embedding_path"), str) and (dataset_root / row["t5_embedding_path"]).exists()
        if not overwrite and (ptr_ok or abs_pt.exists()):
            if not isinstance(row.get("t5_embedding_path"), str):
                row["t5_embedding_path"] = rel
                updated += 1
            else:
                skipped += 1
            continue

        instr = episode_instruction(row)
        if instr not in instruction_cache:
            instruction_cache[instr] = encode_t5(encoder, instr, device)
        torch.save(instruction_cache[instr], abs_pt)
        row["t5_embedding_path"] = rel
        updated += 1

    write_jsonlines_atomic(episodes_path, episodes)
    return updated, skipped


def quat_to_rpy(quat: np.ndarray, order: str) -> np.ndarray:
    q = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    q /= norm
    if order == "xyzw":
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == "wxyz":
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"Unsupported quat order: {order}")

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1).astype(np.float32)


def angle_diff(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return (curr - prev + np.pi) % (2.0 * np.pi) - np.pi


def action40_to_abs_bimanual(action: np.ndarray, quat_order: str) -> np.ndarray:
    if action.ndim != 2 or action.shape[1] < 16:
        raise ValueError(f"Expected action [T,>=16], got {action.shape}")
    t = action.shape[0]
    left_xyz = action[:, 2:5]
    right_xyz = action[:, 5:8]
    left_rpy = quat_to_rpy(action[:, 8:12], quat_order)
    right_rpy = quat_to_rpy(action[:, 12:16], quat_order)
    left_g = action[:, 0:1]
    right_g = action[:, 1:2]
    return np.concatenate([left_xyz, left_rpy, left_g, right_xyz, right_rpy, right_g], axis=1).astype(np.float32)


def abs_to_delta_window(window_abs: np.ndarray) -> np.ndarray:
    if window_abs.shape[0] <= 1:
        out = np.zeros((1, 14), dtype=np.float32)
        out[:, 6:7] = window_abs[-1:, 6:7]
        out[:, 13:14] = window_abs[-1:, 13:14]
        return out

    left = window_abs[:, :7]
    right = window_abs[:, 7:14]

    def one_arm_delta(arm: np.ndarray) -> np.ndarray:
        dpos = arm[1:, :3] - arm[:-1, :3]
        drot = angle_diff(arm[1:, 3:6], arm[:-1, 3:6])
        grip = arm[1:, 6:7]
        return np.concatenate([dpos, drot, grip], axis=1).astype(np.float32)

    return np.concatenate([one_arm_delta(left), one_arm_delta(right)], axis=1)


def round_nearest(value: float, n: int = 1) -> int:
    return int(round(value / n) * n)


def summarize_arm(arr: np.ndarray, include_gripper: bool = True) -> str:
    dx_m, dy_m, dz_m = (float(arr[:, i].sum()) for i in range(3))
    dr, dp, dyaw = (float(arr[:, i].sum()) for i in range(3, 6))
    parts: List[str] = []
    for value, pos, neg in [
        (dx_m, "move forward", "move back"),
        (dz_m, "move up", "move down"),
        (dy_m, "move left", "move right"),
    ]:
        cm = round(abs(value * 100.0), 1)
        if cm != 0:
            parts.append(f"{pos if value > 0 else neg} {cm:.1f} cm")
    for value, pos, neg in [
        (dr, "tilt left", "tilt right"),
        (dp, "tilt back", "tilt forward"),
        (dyaw, "rotate counterclockwise", "rotate clockwise"),
    ]:
        deg = round_nearest(abs(value * 180.0 / math.pi), 1)
        if deg > 0:
            parts.append(f"{pos if value > 0 else neg} {deg} degrees")
    if include_gripper and arr.shape[1] >= 7:
        parts.append("open gripper" if float(arr[-1, 6]) >= 0.5 else "close gripper")
    return ", ".join(parts) if parts else "hold position"


def summarize_bimanual_delta(delta: np.ndarray) -> str:
    left = summarize_arm(delta[:, :7], include_gripper=True)
    right = summarize_arm(delta[:, 7:14], include_gripper=True)
    return f"Left arm: {left}. Right arm: {right}"


def build_language_action(action: np.ndarray, window_size: int, quat_order: str) -> List[str]:
    abs_pose = action40_to_abs_bimanual(action, quat_order)
    lines: List[str] = []
    for i in range(abs_pose.shape[0]):
        j = min(i + window_size, abs_pose.shape[0])
        lines.append(summarize_bimanual_delta(abs_to_delta_window(abs_pose[i:j, :])))
    return lines


def read_action_parquet(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["action"])
    col = table.column("action").to_pylist()
    return np.asarray(col, dtype=np.float32)


def ensure_language_action(
    dataset_root: Path,
    folder_name: str,
    window_size: int,
    quat_order: str,
    overwrite: bool,
) -> Tuple[int, int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = load_jsonlines(episodes_path)
    out_dir = dataset_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_files = {p.stem: p for p in (dataset_root / "data").glob("chunk-*/episode_*.parquet")}

    updated = 0
    skipped = 0
    for row in episodes:
        ep_idx = int(row["episode_index"])
        stem = f"episode_{ep_idx:06d}"
        rel = f"{folder_name}/{stem}.txt"
        out_path = dataset_root / rel
        if out_path.exists() and not overwrite:
            row["language_action_path"] = rel
            skipped += 1
            continue
        parquet = data_files.get(stem)
        if parquet is None:
            raise FileNotFoundError(f"Missing parquet for {stem} under {dataset_root / 'data'}")
        action = read_action_parquet(parquet)
        lines = build_language_action(action, window_size=window_size, quat_order=quat_order)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        row["language_action_path"] = rel
        updated += 1

    write_jsonlines_atomic(episodes_path, episodes)
    return updated, skipped


def dataset_roots_from_archives(archives: Iterable[Path]) -> List[Path]:
    roots = []
    for archive in archives:
        root = archive_output_root(archive)
        if (root / "meta" / "episodes.jsonl").exists():
            roots.append(root)
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare AgiBotWorld2026 LeRobot archives for Motus.")
    parser.add_argument("--root", type=Path, default=Path("/root/nas/robot_data/AgiBotWorld2026"))
    parser.add_argument("--wan_path", type=str, default="/root/nas/code/d0/pretrained_models")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding")
    parser.add_argument("--language_action_dir_name", type=str, default="language_action")
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--include_temp", action="store_true")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_t5", action="store_true")
    parser.add_argument("--skip_language_action", action="store_true")
    parser.add_argument("--overwrite_extract", action="store_true")
    parser.add_argument("--overwrite_t5", action="store_true")
    parser.add_argument("--overwrite_language_action", action="store_true")
    parser.add_argument("--max_archives", type=int, default=0, help="Debug limit; 0 means all archives.")
    parser.add_argument("--extract_workers", type=int, default=1)
    args = parser.parse_args()

    archives = find_archives(args.root, include_temp=args.include_temp)
    if args.max_archives and args.max_archives > 0:
        archives = archives[: args.max_archives]
    print(f"archives={len(archives)} root={args.root}", flush=True)

    if not args.skip_extract:
        extract_archives(
            archives=archives,
            overwrite_extract=args.overwrite_extract,
            workers=max(1, int(args.extract_workers)),
        )

    dataset_roots = dataset_roots_from_archives(archives)
    print(f"dataset_roots={len(dataset_roots)}", flush=True)

    encoder = None
    instruction_cache: Dict[str, torch.Tensor] = {}
    for idx, dataset_root in enumerate(dataset_roots, start=1):
        print(f"[{idx}/{len(dataset_roots)}] dataset={dataset_root}", flush=True)
        if not args.skip_t5:
            if encoder is None:
                print(f"Loading T5 encoder on {args.device} ...", flush=True)
                encoder = init_t5_encoder(args.wan_path, args.device, args.text_len)
            updated, skipped = ensure_t5_cache(
                dataset_root=dataset_root,
                encoder=encoder,
                instruction_cache=instruction_cache,
                device=args.device,
                folder_name=args.t5_folder_name,
                overwrite=args.overwrite_t5,
            )
            print(f"  t5 updated={updated} skipped={skipped}", flush=True)
        if not args.skip_language_action:
            updated, skipped = ensure_language_action(
                dataset_root=dataset_root,
                folder_name=args.language_action_dir_name,
                window_size=args.window_size,
                quat_order=args.quat_order,
                overwrite=args.overwrite_language_action,
            )
            print(f"  language_action updated={updated} skipped={skipped}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
