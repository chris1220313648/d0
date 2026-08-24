#!/usr/bin/env python3
"""Prepare RoboCOIN LeRobot datasets for EEF-LAP training."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_ROOT = Path("/root/nas/code/d0/data/robot_data/robocoin")


@dataclass(frozen=True)
class PreparationStats:
    updated: int
    skipped: int
    failed: int = 0


def load_jsonlines(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonlines_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def discover_dataset_roots(root: Path) -> List[Path]:
    roots: List[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "meta" / "info.json").exists() and (path / "data").exists() and (path / "videos").exists():
            roots.append(path)
    return roots


def _feature_names(info: Mapping[str, Any], key: str) -> List[str]:
    feature = info.get("features", {}).get(key, {})
    names = feature.get("names", []) if isinstance(feature, Mapping) else []
    return [str(name) for name in names]


def _name_to_idx(names: Sequence[str]) -> Dict[str, int]:
    return {str(name): idx for idx, name in enumerate(names)}


def quat_to_rpy_xyzw(quat: np.ndarray) -> np.ndarray:
    q = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    q /= norm
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1).astype(np.float32)


def _quat_to_rpy(quat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "xyzw":
        return quat_to_rpy_xyzw(quat)
    if quat_order == "wxyz":
        return quat_to_rpy_xyzw(quat[:, [1, 2, 3, 0]])
    raise ValueError(f"Unsupported quat_order={quat_order!r}")


def _gather_columns(values: np.ndarray, name_to_idx: Mapping[str, int], names: Sequence[str]) -> Optional[np.ndarray]:
    indices = [name_to_idx.get(name) for name in names]
    if any(idx is None or idx >= values.shape[1] for idx in indices):
        return None
    return values[:, [int(idx) for idx in indices]].astype(np.float32, copy=False)


def _extract_arm_abs_pose(
    values: np.ndarray,
    name_to_idx: Mapping[str, int],
    side: str,
    quat_order: str,
) -> Optional[Tuple[np.ndarray, bool]]:
    gripper = _gather_columns(values, name_to_idx, [f"{side}_gripper_open"])
    include_gripper = gripper is not None
    if gripper is None:
        gripper = np.zeros((values.shape[0], 1), dtype=np.float32)

    for prefix in (f"{side}_eef", f"{side}_end"):
        xyz = _gather_columns(values, name_to_idx, [f"{prefix}_pos_{axis}_m" for axis in ("x", "y", "z")])
        if xyz is None:
            continue

        rpy = _gather_columns(
            values,
            name_to_idx,
            [f"{prefix}_rot_euler_{axis}_rad" for axis in ("x", "y", "z")],
        )
        if rpy is None:
            quat = _gather_columns(values, name_to_idx, [f"{prefix}_quat_{axis}" for axis in ("x", "y", "z", "w")])
            if quat is not None:
                rpy = _quat_to_rpy(quat, quat_order=quat_order)
        if rpy is None:
            continue

        return np.concatenate([xyz, rpy, gripper], axis=1).astype(np.float32), include_gripper

    return None


def extract_bimanual_eef_abs_pose(
    values: np.ndarray,
    names: Sequence[str],
    quat_order: str = "xyzw",
) -> Optional[Tuple[np.ndarray, bool]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected values [T,D], got {arr.shape}")
    name_to_idx = _name_to_idx(names)
    left = _extract_arm_abs_pose(arr, name_to_idx, "left", quat_order)
    right = _extract_arm_abs_pose(arr, name_to_idx, "right", quat_order)
    if left is None or right is None:
        return None
    include_gripper = bool(left[1] and right[1])
    return np.concatenate([left[0], right[0]], axis=1).astype(np.float32), include_gripper


def _angle_diff(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return (curr - prev + np.pi) % (2.0 * np.pi) - np.pi


def _delta_single_arm(window_abs: np.ndarray, include_gripper: bool) -> np.ndarray:
    if window_abs.shape[0] <= 1:
        base = np.zeros((1, 6), dtype=np.float32)
        if include_gripper:
            return np.concatenate([base, window_abs[-1:, 6:7].astype(np.float32)], axis=1)
        return base

    dpos = (window_abs[1:, :3] - window_abs[:-1, :3]).astype(np.float32)
    drot = _angle_diff(window_abs[1:, 3:6], window_abs[:-1, 3:6]).astype(np.float32)
    if include_gripper:
        return np.concatenate([dpos, drot, window_abs[1:, 6:7].astype(np.float32)], axis=1)
    return np.concatenate([dpos, drot], axis=1)


def _abs_to_delta_window(window_abs: np.ndarray, include_gripper: bool) -> np.ndarray:
    left = _delta_single_arm(window_abs[:, :7], include_gripper=include_gripper)
    right = _delta_single_arm(window_abs[:, 7:14], include_gripper=include_gripper)
    return np.concatenate([left, right], axis=1).astype(np.float32)


def _round_nearest(value: float, n: int = 1) -> int:
    return int(round(value / n) * n)


def _summarize_arm(delta: np.ndarray, include_gripper: bool) -> str:
    dx_m, dy_m, dz_m = (float(delta[:, i].sum()) for i in range(3))
    dr, dp, dyaw = (float(delta[:, i].sum()) for i in range(3, 6))
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
        deg = _round_nearest(abs(value * 180.0 / math.pi), 1)
        if deg > 0:
            parts.append(f"{pos if value > 0 else neg} {deg} degrees")
    if include_gripper and delta.shape[1] >= 7:
        parts.append("open gripper" if float(delta[-1, 6]) >= 0.5 else "close gripper")
    return ", ".join(parts) if parts else "hold position"


def _summarize_bimanual_delta(delta: np.ndarray, include_gripper: bool) -> str:
    arm_dim = 7 if include_gripper else 6
    left = _summarize_arm(delta[:, :arm_dim], include_gripper=include_gripper)
    right = _summarize_arm(delta[:, arm_dim: arm_dim * 2], include_gripper=include_gripper)
    return f"Left arm: {left}. Right arm: {right}"


def build_language_action_from_eef_source(
    values: np.ndarray,
    names: Sequence[str],
    window_size: int,
    quat_order: str = "xyzw",
) -> List[str]:
    extracted = extract_bimanual_eef_abs_pose(values, names, quat_order=quat_order)
    if extracted is None:
        raise ValueError("No complete bimanual EEF pose fields found")
    abs_pose, include_gripper = extracted
    lines: List[str] = []
    for idx in range(abs_pose.shape[0]):
        end = min(idx + int(window_size), abs_pose.shape[0])
        delta = _abs_to_delta_window(abs_pose[idx:end], include_gripper=include_gripper)
        lines.append(_summarize_bimanual_delta(delta, include_gripper=include_gripper))
    return lines


def build_language_action_from_episode_arrays(
    action: Optional[np.ndarray],
    action_names: Sequence[str],
    state: Optional[np.ndarray],
    state_names: Sequence[str],
    window_size: int,
    quat_order: str = "xyzw",
) -> Optional[Tuple[List[str], str]]:
    if action is not None and extract_bimanual_eef_abs_pose(action, action_names, quat_order=quat_order) is not None:
        return build_language_action_from_eef_source(action, action_names, window_size, quat_order), "action"
    if state is not None and extract_bimanual_eef_abs_pose(state, state_names, quat_order=quat_order) is not None:
        return build_language_action_from_eef_source(state, state_names, window_size, quat_order), "observation.state"
    return None


def _read_parquet_arrays(path: Path) -> Dict[str, Optional[np.ndarray]]:
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(path).names)
    columns = [name for name in ("action", "observation.state") if name in schema_names]
    if not columns:
        return {"action": None, "observation.state": None}
    table = pq.read_table(path, columns=columns)
    out: Dict[str, Optional[np.ndarray]] = {"action": None, "observation.state": None}
    for column in columns:
        out[column] = np.asarray(table.column(column).to_pylist(), dtype=np.float32)
    return out


def ensure_language_action(
    dataset_root: Path,
    folder_name: str,
    window_size: int,
    quat_order: str,
    overwrite: bool,
    max_episodes: int = 0,
) -> PreparationStats:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    info_path = dataset_root / "meta" / "info.json"
    episodes = load_jsonlines(episodes_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    action_names = _feature_names(info, "action")
    state_names = _feature_names(info, "observation.state")

    out_dir = dataset_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_files = {path.stem: path for path in (dataset_root / "data").glob("chunk-*/episode_*.parquet")}

    updated = 0
    skipped = 0
    failed = 0
    selected = episodes[:max_episodes] if max_episodes > 0 else episodes
    for row in selected:
        ep_idx = int(row["episode_index"])
        stem = f"episode_{ep_idx:06d}"
        rel = f"{folder_name}/{stem}.txt"
        out_path = dataset_root / rel
        if out_path.exists() and not overwrite:
            row["language_action_path"] = rel
            skipped += 1
            continue

        parquet_path = data_files.get(stem)
        if parquet_path is None:
            failed += 1
            continue
        arrays = _read_parquet_arrays(parquet_path)
        result = build_language_action_from_episode_arrays(
            arrays.get("action"),
            action_names,
            arrays.get("observation.state"),
            state_names,
            window_size=window_size,
            quat_order=quat_order,
        )
        if result is None:
            failed += 1
            row.pop("language_action_path", None)
            continue
        lines, _source = result
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        row["language_action_path"] = rel
        updated += 1

    if max_episodes > 0:
        patch = {int(row["episode_index"]): row for row in selected}
        for idx, row in enumerate(episodes):
            ep_idx = int(row["episode_index"])
            if ep_idx in patch:
                episodes[idx] = patch[ep_idx]
    write_jsonlines_atomic(episodes_path, episodes)
    return PreparationStats(updated=updated, skipped=skipped, failed=failed)


def _load_t5_helpers():
    try:
        from data.lerobot.add_t5_cache_to_lerobot_dataset import (  # type: ignore
            _encode_t5,
            _episode_instruction_from_meta,
            _init_wan_t5_encoder,
        )
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from data.lerobot.add_t5_cache_to_lerobot_dataset import (  # type: ignore
            _encode_t5,
            _episode_instruction_from_meta,
            _init_wan_t5_encoder,
        )
    return _init_wan_t5_encoder, _encode_t5, _episode_instruction_from_meta


def ensure_t5_cache(
    dataset_root: Path,
    encoder: Any,
    device: str,
    folder_name: str,
    overwrite: bool,
    max_episodes: int = 0,
) -> PreparationStats:
    _init_unused, encode_t5, episode_instruction = _load_t5_helpers()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = load_jsonlines(episodes_path)
    selected = episodes[:max_episodes] if max_episodes > 0 else episodes
    out_dir = dataset_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped = 0
    for row in selected:
        ep_idx = int(row["episode_index"])
        rel = f"{folder_name}/episode_{ep_idx:06d}.pt"
        out_path = dataset_root / rel
        if out_path.exists() and not overwrite:
            row["t5_embedding_path"] = rel
            skipped += 1
            continue
        emb = encode_t5(encoder, episode_instruction(row), device=device)
        torch.save(emb, out_path)
        row["t5_embedding_path"] = rel
        updated += 1

    if max_episodes > 0:
        patch = {int(row["episode_index"]): row for row in selected}
        for idx, row in enumerate(episodes):
            ep_idx = int(row["episode_index"])
            if ep_idx in patch:
                episodes[idx] = patch[ep_idx]
    write_jsonlines_atomic(episodes_path, episodes)
    return PreparationStats(updated=updated, skipped=skipped)


def write_failures(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RoboCOIN EEF-LAP language actions and WAN T5 cache.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--wan_path", type=str, default="/root/nas/code/d0/pretrained_models")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--language_action_dir_name", type=str, default="language_action")
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding")
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N dataset roots; 0 means all.")
    parser.add_argument("--max_episodes", type=int, default=0, help="Per-dataset debug limit; 0 means all.")
    parser.add_argument("--skip_language_action", action="store_true")
    parser.add_argument("--skip_t5", action="store_true")
    parser.add_argument("--overwrite_language_action", action="store_true")
    parser.add_argument("--overwrite_t5", action="store_true")
    parser.add_argument(
        "--failure_log",
        type=Path,
        default=DEFAULT_ROOT.parent / "prepare_robocoin_lap_failures.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = discover_dataset_roots(args.root)
    if args.limit > 0:
        roots = roots[: args.limit]
    print(f"root={args.root}", flush=True)
    print(f"datasets={len(roots)}", flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None
    failures: List[Dict[str, Any]] = []

    for idx, dataset_root in enumerate(roots, start=1):
        print(f"[{idx}/{len(roots)}] dataset={dataset_root}", flush=True)
        if not args.skip_language_action:
            try:
                stats = ensure_language_action(
                    dataset_root=dataset_root,
                    folder_name=args.language_action_dir_name,
                    window_size=args.window_size,
                    quat_order=args.quat_order,
                    overwrite=bool(args.overwrite_language_action),
                    max_episodes=int(args.max_episodes),
                )
                print(f"  language_action updated={stats.updated} skipped={stats.skipped} failed={stats.failed}", flush=True)
            except Exception as err:  # noqa: BLE001
                failures.append({"path": str(dataset_root), "stage": "language_action", "error": repr(err)})
                write_failures(args.failure_log, failures)
                print(f"  language_action FAILED {err!r}", flush=True)
                continue

        if not args.skip_t5:
            try:
                if encoder is None:
                    init_t5, _encode_t5_unused, _episode_instruction_unused = _load_t5_helpers()
                    print(f"Loading WAN T5 encoder from {args.wan_path} on {device} ...", flush=True)
                    encoder = init_t5(args.wan_path, device=device, text_len=int(args.text_len))
                stats = ensure_t5_cache(
                    dataset_root=dataset_root,
                    encoder=encoder,
                    device=device,
                    folder_name=args.t5_folder_name,
                    overwrite=bool(args.overwrite_t5),
                    max_episodes=int(args.max_episodes),
                )
                print(f"  t5 updated={stats.updated} skipped={stats.skipped}", flush=True)
            except Exception as err:  # noqa: BLE001
                failures.append({"path": str(dataset_root), "stage": "t5", "error": repr(err)})
                write_failures(args.failure_log, failures)
                print(f"  t5 FAILED {err!r}", flush=True)

    write_failures(args.failure_log, failures)
    print("done", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
