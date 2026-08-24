#!/usr/bin/env python3
"""Backfill InternData LeRobot datasets with LAP text and WAN T5 embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    from data.lerobot.add_t5_cache_to_lerobot_dataset import _encode_t5, _init_wan_t5_encoder
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data.lerobot.add_t5_cache_to_lerobot_dataset import _encode_t5, _init_wan_t5_encoder


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_ROOT = Path("/root/nas/code/d0/data/robot_data/interndata")
DEFAULT_INCLUDE = ("physical", "sim_updated")


@dataclass
class PreparationStats:
    datasets_seen: int = 0
    episodes_seen: int = 0
    language_updated: int = 0
    language_skipped: int = 0
    language_failed: int = 0
    t5_updated: int = 0
    t5_skipped: int = 0
    t5_failed: int = 0

    def add(self, other: "PreparationStats") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def _natural_key(path_or_name: Path | str) -> list[object]:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def load_jsonlines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def discover_dataset_roots(root: Path | str, include: Sequence[str] = DEFAULT_INCLUDE) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    starts: list[Path] = []
    if include:
        starts = [root_path / item for item in include]
    else:
        starts = [root_path]

    out: list[Path] = []
    seen: set[Path] = set()
    prune_names = {"data", "videos", "language_action", "t5_embedding", "umt5_wan", "__pycache__"}
    for start in starts:
        if not start.exists():
            logger.warning("Include path does not exist, skip: %s", start)
            continue
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in prune_names)
            if Path(dirpath).name != "meta" or "info.json" not in filenames:
                continue
            dataset_root = Path(dirpath).parent
            if dataset_root in seen:
                continue
            if (
                (dataset_root / "meta" / "episodes.jsonl").exists()
                and (dataset_root / "data").is_dir()
                and (dataset_root / "videos").is_dir()
            ):
                seen.add(dataset_root)
                out.append(dataset_root)
    return sorted(out, key=lambda p: str(p))


def select_shard(roots: Sequence[Path], num_shards: int = 1, shard_index: int = 0) -> list[Path]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return list(roots)[shard_index::num_shards]


def _select_first_existing(columns: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _as_2d_float(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected column values to be 2D-compatible, got shape={arr.shape}")
    return arr


def _read_episode_columns(parquet_path: Path, columns: Sequence[str]) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    if not columns:
        return {}
    table = pq.read_table(parquet_path, columns=list(columns))
    return {name: _as_2d_float(table.column(name).to_pylist()) for name in columns}


def _available_columns(parquet_path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(parquet_path).names)


def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat must be [T,4], got {quat.shape}")
    q = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    q /= norm
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1).astype(np.float32)


def _pose_wxyz_to_xyzrpy(pose: np.ndarray) -> np.ndarray:
    if pose.ndim != 2 or pose.shape[1] < 7:
        raise ValueError(f"pose must be [T,7], got {pose.shape}")
    return np.concatenate([pose[:, :3], _quat_wxyz_to_rpy(pose[:, 3:7])], axis=1).astype(np.float32)


def _angle_diff(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return (curr - prev + np.pi) % (2.0 * np.pi) - np.pi


def _delta_single_arm(abs_pose: np.ndarray, include_gripper: bool) -> np.ndarray:
    if abs_pose.shape[0] <= 1:
        base = np.zeros((1, 6), dtype=np.float32)
        if include_gripper:
            return np.concatenate([base, abs_pose[-1:, 6:7].astype(np.float32)], axis=1)
        return base
    dpos = (abs_pose[1:, :3] - abs_pose[:-1, :3]).astype(np.float32)
    drot = _angle_diff(abs_pose[1:, 3:6], abs_pose[:-1, 3:6]).astype(np.float32)
    if include_gripper:
        return np.concatenate([dpos, drot, abs_pose[1:, 6:7].astype(np.float32)], axis=1)
    return np.concatenate([dpos, drot], axis=1)


def _round_nearest(value: float, n: int = 1) -> int:
    return int(round(value / n) * n)


def _format_numeric(value: float) -> str:
    return f"{value:.1f}"


def _summarize_arm(delta: np.ndarray, include_gripper: bool) -> str:
    dx_m = float(delta[:, 0].sum())
    dy_m = float(delta[:, 1].sum())
    dz_m = float(delta[:, 2].sum())
    droll_rad = float(delta[:, 3].sum())
    dpitch_rad = float(delta[:, 4].sum())
    dyaw_rad = float(delta[:, 5].sum())

    parts: list[str] = []
    dx, dy, dz = abs(dx_m * 100.0), abs(dy_m * 100.0), abs(dz_m * 100.0)
    if dx_m > 0 and dx > 0:
        parts.append(f"move forward {_format_numeric(dx)} cm")
    elif dx_m < 0 and dx > 0:
        parts.append(f"move back {_format_numeric(dx)} cm")
    if dz_m > 0 and dz > 0:
        parts.append(f"move up {_format_numeric(dz)} cm")
    elif dz_m < 0 and dz > 0:
        parts.append(f"move down {_format_numeric(dz)} cm")
    if dy_m > 0 and dy > 0:
        parts.append(f"move left {_format_numeric(dy)} cm")
    elif dy_m < 0 and dy > 0:
        parts.append(f"move right {_format_numeric(dy)} cm")

    droll = _round_nearest(abs(droll_rad * 180.0 / np.pi), 1)
    dpitch = _round_nearest(abs(dpitch_rad * 180.0 / np.pi), 1)
    dyaw = _round_nearest(abs(dyaw_rad * 180.0 / np.pi), 1)
    if droll_rad > 0 and droll > 0:
        parts.append(f"tilt left {droll} degrees")
    elif droll_rad < 0 and droll > 0:
        parts.append(f"tilt right {droll} degrees")
    if dpitch_rad > 0 and dpitch > 0:
        parts.append(f"tilt back {dpitch} degrees")
    elif dpitch_rad < 0 and dpitch > 0:
        parts.append(f"tilt forward {dpitch} degrees")
    if dyaw_rad > 0 and dyaw > 0:
        parts.append(f"rotate counterclockwise {dyaw} degrees")
    elif dyaw_rad < 0 and dyaw > 0:
        parts.append(f"rotate clockwise {dyaw} degrees")

    if include_gripper and delta.shape[1] >= 7:
        parts.append("open gripper" if float(delta[-1, 6]) >= 0.5 else "close gripper")
    return ", ".join(parts) if parts else "hold position"


def _build_lines(abs_actions: np.ndarray, arm_count: int, include_gripper: bool, window_size: int) -> list[str]:
    lines: list[str] = []
    total = int(abs_actions.shape[0])
    for idx in range(total):
        window = abs_actions[idx : min(idx + window_size, total)]
        if arm_count == 1:
            lines.append(_summarize_arm(_delta_single_arm(window, include_gripper), include_gripper))
        else:
            left = _delta_single_arm(window[:, :7], include_gripper)
            right = _delta_single_arm(window[:, 7:14], include_gripper)
            lines.append(f"Left arm: {_summarize_arm(left, include_gripper)}. Right arm: {_summarize_arm(right, include_gripper)}")
    return lines


def _side_pose_candidates(side: str) -> list[str]:
    return [
        f"actions.{side}_tcp_to_robot_pose",
        f"actions.{side}_ee_to_robot_pose",
        f"actions.{side}_tcp_to_{side}_armbase_pose",
        f"actions.{side}_ee_to_{side}_armbase_pose",
    ]


def _single_pose_candidates() -> list[str]:
    return [
        "actions.tcp_to_robot_pose",
        "actions.ee_to_robot_pose",
        "actions.tcp_to_armbase_pose",
        "actions.ee_to_armbase_pose",
    ]


def _gripper_candidates(side: str | None) -> list[str]:
    if side is None:
        return ["actions.gripper.openness", "actions.gripper.position"]
    return [f"actions.{side}_gripper.openness", f"actions.{side}_gripper.position"]


def _extract_language_action_source(parquet_path: Path) -> tuple[np.ndarray, int, bool] | None:
    columns = _available_columns(parquet_path)
    left_col = _select_first_existing(columns, _side_pose_candidates("left"))
    right_col = _select_first_existing(columns, _side_pose_candidates("right"))
    if left_col and right_col:
        left_grip_col = _select_first_existing(columns, _gripper_candidates("left"))
        right_grip_col = _select_first_existing(columns, _gripper_candidates("right"))
        selected = [left_col, right_col]
        if left_grip_col:
            selected.append(left_grip_col)
        if right_grip_col:
            selected.append(right_grip_col)
        data = _read_episode_columns(parquet_path, selected)
        left = _pose_wxyz_to_xyzrpy(data[left_col])
        right = _pose_wxyz_to_xyzrpy(data[right_col])
        include_gripper = bool(left_grip_col and right_grip_col)
        left_grip = data[left_grip_col] if left_grip_col else np.zeros((left.shape[0], 1), dtype=np.float32)
        right_grip = data[right_grip_col] if right_grip_col else np.zeros((right.shape[0], 1), dtype=np.float32)
        return (
            np.concatenate([left, left_grip[:, :1], right, right_grip[:, :1]], axis=1).astype(np.float32),
            2,
            include_gripper,
        )

    pose_col = _select_first_existing(columns, _single_pose_candidates())
    if pose_col:
        grip_col = _select_first_existing(columns, _gripper_candidates(None))
        selected = [pose_col] + ([grip_col] if grip_col else [])
        data = _read_episode_columns(parquet_path, selected)
        pose = _pose_wxyz_to_xyzrpy(data[pose_col])
        gripper = data[grip_col] if grip_col else np.zeros((pose.shape[0], 1), dtype=np.float32)
        return np.concatenate([pose, gripper[:, :1]], axis=1).astype(np.float32), 1, bool(grip_col)

    return None


def _data_files_by_stem(dataset_root: Path) -> dict[str, Path]:
    return {path.stem: path for path in (dataset_root / "data").glob("chunk-*/episode_*.parquet")}


def ensure_language_action(
    dataset_root: Path | str,
    folder_name: str = "language_action",
    window_size: int = 16,
    overwrite: bool = False,
    max_episodes: int = 0,
) -> PreparationStats:
    root = Path(dataset_root)
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes = load_jsonlines(episodes_path)
    selected = episodes[:max_episodes] if max_episodes > 0 else episodes
    data_files = _data_files_by_stem(root)
    out_dir = root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = PreparationStats(episodes_seen=len(selected))
    for row in selected:
        ep_idx = int(row["episode_index"])
        stem = f"episode_{ep_idx:06d}"
        rel = f"{folder_name}/{stem}.txt"
        out_path = root / rel
        if out_path.exists() and not overwrite:
            row["language_action_path"] = rel
            stats.language_skipped += 1
            continue

        parquet_path = data_files.get(stem)
        if parquet_path is None:
            row.pop("language_action_path", None)
            stats.language_failed += 1
            continue

        try:
            source = _extract_language_action_source(parquet_path)
            if source is None:
                row.pop("language_action_path", None)
                stats.language_failed += 1
                continue
            abs_actions, arm_count, include_gripper = source
            lines = _build_lines(abs_actions, arm_count, include_gripper, window_size)
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            row["language_action_path"] = rel
            stats.language_updated += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed language_action for %s episode %d: %s", root, ep_idx, exc)
            row.pop("language_action_path", None)
            stats.language_failed += 1

    if max_episodes > 0:
        patch = {int(row["episode_index"]): row for row in selected}
        for idx, row in enumerate(episodes):
            if int(row["episode_index"]) in patch:
                episodes[idx] = patch[int(row["episode_index"])]
    write_jsonlines_atomic(episodes_path, episodes)
    return stats


def _episode_instruction(row: Mapping[str, Any]) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], str):
        return tasks[0]
    task = row.get("task", "")
    return task if isinstance(task, str) else str(task)


def ensure_t5_cache(
    dataset_root: Path | str,
    encoder: Any,
    device: str,
    folder_name: str = "t5_embedding",
    overwrite: bool = False,
    max_episodes: int = 0,
) -> PreparationStats:
    root = Path(dataset_root)
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes = load_jsonlines(episodes_path)
    selected = episodes[:max_episodes] if max_episodes > 0 else episodes
    out_dir = root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = PreparationStats(episodes_seen=len(selected))
    for row in selected:
        ep_idx = int(row["episode_index"])
        rel = f"{folder_name}/episode_{ep_idx:06d}.pt"
        out_path = root / rel
        if out_path.exists() and not overwrite:
            row["t5_embedding_path"] = rel
            stats.t5_skipped += 1
            continue
        try:
            emb = _encode_t5(encoder, _episode_instruction(row), device=device)
            torch.save(emb, out_path)
            row["t5_embedding_path"] = rel
            stats.t5_updated += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed T5 cache for %s episode %d: %s", root, ep_idx, exc)
            stats.t5_failed += 1

    if max_episodes > 0:
        patch = {int(row["episode_index"]): row for row in selected}
        for idx, row in enumerate(episodes):
            if int(row["episode_index"]) in patch:
                episodes[idx] = patch[int(row["episode_index"])]
    write_jsonlines_atomic(episodes_path, episodes)
    return stats


def run_backfill(args: argparse.Namespace) -> PreparationStats:
    all_roots = discover_dataset_roots(args.root, include=args.include)
    if args.max_datasets and args.max_datasets > 0:
        all_roots = all_roots[: args.max_datasets]
    roots = select_shard(all_roots, num_shards=args.num_shards, shard_index=args.shard_index)
    logger.info(
        "Discovered %d dataset roots, running shard %d/%d with %d roots",
        len(all_roots),
        args.shard_index,
        args.num_shards,
        len(roots),
    )

    encoder = None
    if not args.skip_t5:
        wan_path = args.wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
        if not wan_path:
            raise ValueError("WAN path is required unless --skip_t5 is set. Use --wan_path or WAN_PATH/WAN_ROOT.")
        logger.info("Loading WAN T5 encoder from %s on %s", wan_path, args.device)
        encoder = _init_wan_t5_encoder(wan_path=wan_path, device=args.device, text_len=args.text_len)

    total = PreparationStats(datasets_seen=len(roots))
    for dataset_root in roots:
        logger.info("Processing %s", dataset_root)
        if not args.skip_language_action:
            total.add(
                ensure_language_action(
                    dataset_root,
                    folder_name=args.language_action_folder,
                    window_size=args.window_size,
                    overwrite=args.overwrite_language_action,
                    max_episodes=args.max_episodes,
                )
            )
        if not args.skip_t5:
            total.add(
                ensure_t5_cache(
                    dataset_root,
                    encoder=encoder,
                    device=args.device,
                    folder_name=args.t5_folder_name,
                    overwrite=args.overwrite_t5,
                    max_episodes=args.max_episodes,
                )
            )
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill InternData LeRobot datasets with LAP assets.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="InternData root.")
    parser.add_argument("--include", nargs="+", default=list(DEFAULT_INCLUDE), help="Top-level subtrees to scan.")
    parser.add_argument("--wan_path", type=str, default=None, help="Base path containing Wan2.2-TI2V-5B.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for WAN T5 encoding.")
    parser.add_argument("--text_len", type=int, default=512, help="WAN T5 text length.")
    parser.add_argument("--language_action_folder", type=str, default="language_action")
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding")
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--overwrite_language_action", action="store_true")
    parser.add_argument("--overwrite_t5", action="store_true")
    parser.add_argument("--skip_language_action", action="store_true")
    parser.add_argument("--skip_t5", action="store_true")
    parser.add_argument("--max_datasets", type=int, default=0, help="0 means all discovered datasets.")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 means all episodes per dataset.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of dataset-root shards.")
    parser.add_argument("--shard_index", type=int, default=0, help="This process shard index, 0-based.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    stats = run_backfill(args)
    logger.info("Completed: %s", stats)


if __name__ == "__main__":
    main()
