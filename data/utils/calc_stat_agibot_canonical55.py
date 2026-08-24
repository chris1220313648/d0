#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from data.canonical55 import (
    ARM_JOINT,
    BASE,
    CANONICAL55_DIM,
    HEAD,
    LEFT_EEF,
    RIGHT_EEF,
    WAIST,
)


DEFAULT_ROOT = Path("/root/nas/code/d0/data/robot_data/AgiBotWorld2026")
DEFAULT_STAT_PATH = Path("/root/nas/code/d0/data/utils/stat.json")
STATE_ORIENTATION_SLOTS = (slice(LEFT_EEF.start + 3, LEFT_EEF.stop), slice(RIGHT_EEF.start + 3, RIGHT_EEF.stop))
ACTION_ORIENTATION_SLOTS = STATE_ORIENTATION_SLOTS


def discover_lerobot_roots(root: Path, suffix: str = "_split", required: Sequence[str] = ("meta/info.json", "data")) -> List[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"AgiBot root not found: {root}")

    roots: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if suffix and not path.name.endswith(suffix):
            continue
        if all((path / rel).exists() for rel in required):
            roots.append(path)
    roots.sort()
    if not roots:
        raise ValueError(f"No AgiBot LeRobot datasets found under {root} with suffix {suffix!r}")
    return roots


def _load_fields(dataset_root: Path, feature_name: str) -> Dict[str, Tuple[int, ...]]:
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    descriptions = info.get("features", {}).get(feature_name, {}).get("field_descriptions", {})
    fields: Dict[str, Tuple[int, ...]] = {}
    for name, desc in descriptions.items():
        indices = desc.get("indices", []) if isinstance(desc, dict) else []
        fields[str(name)] = tuple(int(i) for i in indices)
    return fields


def _flatten_stat(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def _field_minmax(
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    fields: Dict[str, Tuple[int, ...]],
    field_name: str,
    expected_dim: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    indices = fields.get(field_name)
    if not indices:
        return None
    if expected_dim is not None and len(indices) < expected_dim:
        return None
    selected = tuple(indices[:expected_dim])
    if max(selected) >= len(raw_min) or max(selected) >= len(raw_max):
        raise ValueError(f"Field {field_name!r} exceeds raw stat length")
    index = np.asarray(selected, dtype=np.int64)
    return raw_min[index], raw_max[index]


def _update_range(
    stat_min: np.ndarray,
    stat_max: np.ndarray,
    active: np.ndarray,
    dst: slice,
    values_min: np.ndarray,
    values_max: np.ndarray,
) -> None:
    stat_min[dst] = np.minimum(stat_min[dst], values_min)
    stat_max[dst] = np.maximum(stat_max[dst], values_max)
    active[dst] = True


def _update_scalar_range(
    stat_min: np.ndarray,
    stat_max: np.ndarray,
    active: np.ndarray,
    dst: int,
    values_min: np.ndarray,
    values_max: np.ndarray,
) -> None:
    stat_min[dst] = min(stat_min[dst], float(values_min.reshape(-1)[0]))
    stat_max[dst] = max(stat_max[dst], float(values_max.reshape(-1)[0]))
    active[dst] = True


def _update_direct_state_stats(
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    fields: Dict[str, Tuple[int, ...]],
    stat_min: np.ndarray,
    stat_max: np.ndarray,
    active: np.ndarray,
) -> None:
    pair = _field_minmax(raw_min, raw_max, fields, "state/joint/position", 14)
    if pair is not None:
        _update_range(stat_min, stat_max, active, ARM_JOINT, pair[0], pair[1])

    pair = _field_minmax(raw_min, raw_max, fields, "state/end/arm_position", 6)
    if pair is not None:
        _update_range(stat_min, stat_max, active, slice(LEFT_EEF.start, LEFT_EEF.start + 3), pair[0][0:3], pair[1][0:3])
        _update_range(stat_min, stat_max, active, slice(RIGHT_EEF.start, RIGHT_EEF.start + 3), pair[0][3:6], pair[1][3:6])

    pair = _field_minmax(raw_min, raw_max, fields, "state/left_effector/position", 1)
    if pair is not None:
        _update_scalar_range(stat_min, stat_max, active, 26, pair[0], pair[1])
    pair = _field_minmax(raw_min, raw_max, fields, "state/right_effector/position", 1)
    if pair is not None:
        _update_scalar_range(stat_min, stat_max, active, 27, pair[0], pair[1])

    pair = _field_minmax(raw_min, raw_max, fields, "state/waist/position", 4)
    if pair is not None:
        _update_range(stat_min, stat_max, active, WAIST, pair[0], pair[1])
    pair = _field_minmax(raw_min, raw_max, fields, "state/head/position", 2)
    if pair is not None:
        _update_range(stat_min, stat_max, active, HEAD, pair[0], pair[1])


def _update_direct_action_stats(
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    fields: Dict[str, Tuple[int, ...]],
    stat_min: np.ndarray,
    stat_max: np.ndarray,
    active: np.ndarray,
) -> None:
    pair = _field_minmax(raw_min, raw_max, fields, "action/joint/position", 14)
    if pair is not None:
        _update_range(stat_min, stat_max, active, ARM_JOINT, pair[0], pair[1])

    pair = _field_minmax(raw_min, raw_max, fields, "action/end/position", 6)
    if pair is not None:
        _update_range(stat_min, stat_max, active, slice(LEFT_EEF.start, LEFT_EEF.start + 3), pair[0][0:3], pair[1][0:3])
        _update_range(stat_min, stat_max, active, slice(RIGHT_EEF.start, RIGHT_EEF.start + 3), pair[0][3:6], pair[1][3:6])

    pair = _field_minmax(raw_min, raw_max, fields, "action/left_effector/position", 1)
    if pair is not None:
        _update_scalar_range(stat_min, stat_max, active, 26, pair[0], pair[1])
    pair = _field_minmax(raw_min, raw_max, fields, "action/right_effector/position", 1)
    if pair is not None:
        _update_scalar_range(stat_min, stat_max, active, 27, pair[0], pair[1])

    pair = _field_minmax(raw_min, raw_max, fields, "action/waist/position", 4)
    if pair is not None:
        _update_range(stat_min, stat_max, active, WAIST, pair[0], pair[1])
    pair = _field_minmax(raw_min, raw_max, fields, "action/head/position", 2)
    if pair is not None:
        _update_range(stat_min, stat_max, active, HEAD, pair[0], pair[1])
    pair = _field_minmax(raw_min, raw_max, fields, "action/robot/velocity", 2)
    if pair is not None:
        _update_range(stat_min, stat_max, active, slice(BASE.start, BASE.start + 2), pair[0], pair[1])


def _iter_episode_stat_rows(dataset_roots: Sequence[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for dataset_root in dataset_roots:
        stat_path = dataset_root / "meta" / "episodes_stats.jsonl"
        if not stat_path.exists():
            raise FileNotFoundError(f"Missing episodes_stats.jsonl: {stat_path}")
        with stat_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield dataset_root, json.loads(line)


def _aggregate_direct_stats(
    dataset_roots: Sequence[Path],
    state_min: np.ndarray,
    state_max: np.ndarray,
    state_active: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    action_active: np.ndarray,
) -> int:
    fields_cache: Dict[Path, Tuple[Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]] = {}
    row_count = 0
    for dataset_root, row in tqdm(list(_iter_episode_stat_rows(dataset_roots)), desc="AgiBot raw episode stats"):
        if dataset_root not in fields_cache:
            fields_cache[dataset_root] = (_load_fields(dataset_root, "observation.state"), _load_fields(dataset_root, "action"))
        state_fields, action_fields = fields_cache[dataset_root]
        stats = row.get("stats", {})
        if "observation.state" in stats:
            raw = stats["observation.state"]
            _update_direct_state_stats(
                _flatten_stat(raw["min"]),
                _flatten_stat(raw["max"]),
                state_fields,
                state_min,
                state_max,
                state_active,
            )
        if "action" in stats:
            raw = stats["action"]
            _update_direct_action_stats(
                _flatten_stat(raw["min"]),
                _flatten_stat(raw["max"]),
                action_fields,
                action_min,
                action_max,
                action_active,
            )
        row_count += 1
    return row_count


def _list_episode_parquets(dataset_roots: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for dataset_root in dataset_roots:
        files.extend(sorted((dataset_root / "data").rglob("episode_*.parquet")))
    files.sort()
    if not files:
        raise ValueError("No AgiBot parquet files found under discovered dataset roots")
    return files


def _fixed_list_column_to_numpy(table, column_name: str) -> np.ndarray:
    values = table[column_name].combine_chunks().values.to_numpy(zero_copy_only=False)
    width = table[column_name].type.list_size
    return np.asarray(values, dtype=np.float32).reshape(-1, width)


def _field_array(source: np.ndarray, fields: Dict[str, Tuple[int, ...]], field_name: str, expected_dim: int) -> Optional[np.ndarray]:
    indices = fields.get(field_name)
    if not indices or len(indices) < expected_dim:
        return None
    selected = tuple(indices[:expected_dim])
    if max(selected) >= source.shape[-1]:
        raise ValueError(f"Field {field_name!r} exceeds source shape {source.shape}")
    return source[:, selected].astype(np.float64, copy=False)


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    if quat.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return Rotation.from_quat(quat).as_rotvec()


def _relative_rotvec(target_quat: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
    if target_quat.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return (Rotation.from_quat(target_quat) * Rotation.from_quat(base_quat).inv()).as_rotvec()


def _update_vector_observations(
    stat_min: np.ndarray,
    stat_max: np.ndarray,
    active: np.ndarray,
    dst: slice,
    observations: np.ndarray,
) -> None:
    if observations.size == 0:
        return
    stat_min[dst] = np.minimum(stat_min[dst], np.min(observations, axis=0))
    stat_max[dst] = np.maximum(stat_max[dst], np.max(observations, axis=0))
    active[dst] = True


def _compute_state_orientation_stats(
    parquet_files: Sequence[Path],
    state_min: np.ndarray,
    state_max: np.ndarray,
    state_active: np.ndarray,
) -> int:
    processed = 0
    fields_cache: Dict[Path, Dict[str, Tuple[int, ...]]] = {}
    for parquet_path in tqdm(parquet_files, desc="AgiBot state rotvec"):
        dataset_root = parquet_path.parents[2]
        if dataset_root not in fields_cache:
            fields_cache[dataset_root] = _load_fields(dataset_root, "observation.state")
        fields = fields_cache[dataset_root]
        indices = fields.get("state/end/arm_orientation")
        if not indices or len(indices) < 8:
            continue
        pf = pq.ParquetFile(parquet_path)
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["observation.state"])
            raw_state = _fixed_list_column_to_numpy(table, "observation.state")
            quat = _field_array(raw_state, fields, "state/end/arm_orientation", 8)
            if quat is None:
                continue
            _update_vector_observations(state_min, state_max, state_active, STATE_ORIENTATION_SLOTS[0], _quat_to_rotvec(quat[:, 0:4]))
            _update_vector_observations(state_min, state_max, state_active, STATE_ORIENTATION_SLOTS[1], _quat_to_rotvec(quat[:, 4:8]))
            processed += raw_state.shape[0]
    return processed


def _parquet_row_count(parquet_path: Path) -> int:
    return int(pq.ParquetFile(parquet_path).metadata.num_rows)


def _condition_indices(
    n_rows: int,
    max_action_offset: int,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_count = max(0, n_rows - max_action_offset)
    if valid_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if sample_count >= valid_count:
        return np.arange(valid_count, dtype=np.int64)
    return np.sort(rng.choice(valid_count, size=sample_count, replace=False).astype(np.int64))


def _compute_action_relative_orientation_stats(
    parquet_files: Sequence[Path],
    action_min: np.ndarray,
    action_max: np.ndarray,
    action_active: np.ndarray,
    *,
    max_action_offset: int,
    rotvec_sample_conditions: int,
    seed: int,
) -> int:
    row_counts = [_parquet_row_count(path) for path in parquet_files]
    valid_counts = [max(0, n - max_action_offset) for n in row_counts]
    total_valid = int(sum(valid_counts))
    if total_valid == 0:
        return 0

    sample_total = int(rotvec_sample_conditions)
    if sample_total <= 0 or sample_total >= total_valid:
        sample_counts = valid_counts
    else:
        raw_counts = np.asarray(valid_counts, dtype=np.float64) * (sample_total / float(total_valid))
        sample_counts = np.floor(raw_counts).astype(np.int64).tolist()
        remainders = raw_counts - np.floor(raw_counts)
        remaining = sample_total - int(sum(sample_counts))
        for idx in np.argsort(-remainders)[:remaining]:
            if valid_counts[int(idx)] > sample_counts[int(idx)]:
                sample_counts[int(idx)] += 1
        sample_counts = [min(max(1, c) if v > 0 else 0, v) for c, v in zip(sample_counts, valid_counts)]

    rng = np.random.default_rng(seed)
    fields_cache: Dict[Path, Dict[str, Tuple[int, ...]]] = {}
    sampled_conditions = 0
    for parquet_path, n_rows, sample_count in tqdm(
        list(zip(parquet_files, row_counts, sample_counts)),
        desc="AgiBot action relative rotvec",
    ):
        if sample_count <= 0:
            continue
        dataset_root = parquet_path.parents[2]
        if dataset_root not in fields_cache:
            fields_cache[dataset_root] = _load_fields(dataset_root, "action")
        fields = fields_cache[dataset_root]
        if not fields.get("action/end/orientation"):
            continue
        table = pq.read_table(parquet_path, columns=["action"])
        raw_action = _fixed_list_column_to_numpy(table, "action")
        quat = _field_array(raw_action, fields, "action/end/orientation", 8)
        if quat is None:
            continue
        cond_idx = _condition_indices(n_rows, max_action_offset, sample_count, rng)
        if cond_idx.size == 0:
            continue
        base_quat = quat[cond_idx]
        for offset in range(1, max_action_offset + 1):
            target_idx = cond_idx + offset
            valid = target_idx < n_rows
            if not np.any(valid):
                continue
            target_quat = quat[target_idx[valid]]
            base = base_quat[valid]
            _update_vector_observations(
                action_min,
                action_max,
                action_active,
                ACTION_ORIENTATION_SLOTS[0],
                _relative_rotvec(target_quat[:, 0:4], base[:, 0:4]),
            )
            _update_vector_observations(
                action_min,
                action_max,
                action_active,
                ACTION_ORIENTATION_SLOTS[1],
                _relative_rotvec(target_quat[:, 4:8], base[:, 4:8]),
            )
        sampled_conditions += int(cond_idx.size)
    return sampled_conditions


def _finalize_stats(stat_min: np.ndarray, stat_max: np.ndarray, active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    final_min = np.where(active, stat_min, 0.0)
    final_max = np.where(active, stat_max, 1.0)
    if np.isinf(final_min).any() or np.isinf(final_max).any():
        bad = np.where(np.isinf(final_min) | np.isinf(final_max))[0].tolist()
        raise ValueError(f"Unfinished AgiBot stat dimensions: {bad}")
    return final_min.astype(np.float32), final_max.astype(np.float32)


def compute_agibot_stats(
    root: Path | str = DEFAULT_ROOT,
    *,
    suffix: str = "_split",
    max_action_offset: int = 48,
    rotvec_sample_conditions: int = 250_000,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    start = time.time()
    dataset_roots = discover_lerobot_roots(Path(root), suffix=suffix)

    state_min = np.full(CANONICAL55_DIM, np.inf, dtype=np.float64)
    state_max = np.full(CANONICAL55_DIM, -np.inf, dtype=np.float64)
    action_min = np.full(CANONICAL55_DIM, np.inf, dtype=np.float64)
    action_max = np.full(CANONICAL55_DIM, -np.inf, dtype=np.float64)
    state_active = np.zeros(CANONICAL55_DIM, dtype=bool)
    action_active = np.zeros(CANONICAL55_DIM, dtype=bool)

    episode_stat_rows = _aggregate_direct_stats(dataset_roots, state_min, state_max, state_active, action_min, action_max, action_active)
    parquet_files = _list_episode_parquets(dataset_roots)
    state_orientation_frames = _compute_state_orientation_stats(parquet_files, state_min, state_max, state_active)
    action_rotvec_conditions = _compute_action_relative_orientation_stats(
        parquet_files,
        action_min,
        action_max,
        action_active,
        max_action_offset=int(max_action_offset),
        rotvec_sample_conditions=int(rotvec_sample_conditions),
        seed=int(seed),
    )

    state_final_min, state_final_max = _finalize_stats(state_min, state_max, state_active)
    action_final_min, action_final_max = _finalize_stats(action_min, action_max, action_active)
    elapsed = time.time() - start

    return {
        "agibot": {
            "state": {
                "min": state_final_min.tolist(),
                "max": state_final_max.tolist(),
                "action_dim": CANONICAL55_DIM,
                "active_mask": state_active.tolist(),
                "dataset_count": len(dataset_roots),
                "episode_stat_rows": int(episode_stat_rows),
                "orientation_frames_scanned": int(state_orientation_frames),
                "stats_mode": "hybrid_raw_stats_plus_rotvec",
                "processing_time_seconds": elapsed,
            },
            "action": {
                "min": action_final_min.tolist(),
                "max": action_final_max.tolist(),
                "action_dim": CANONICAL55_DIM,
                "active_mask": action_active.tolist(),
                "dataset_count": len(dataset_roots),
                "episode_stat_rows": int(episode_stat_rows),
                "max_action_offset": int(max_action_offset),
                "rotvec_stats_mode": "sampled" if rotvec_sample_conditions > 0 else "full",
                "rotvec_sample_conditions": int(action_rotvec_conditions),
                "seed": int(seed),
                "stats_mode": "hybrid_raw_stats_plus_relative_rotvec",
                "processing_time_seconds": elapsed,
            },
        }
    }


def _validate_agibot_stats(stats: Dict[str, Any]) -> None:
    agibot = stats.get("agibot")
    if not isinstance(agibot, dict):
        raise ValueError("Missing agibot stats")
    for signal in ("state", "action"):
        entry = agibot.get(signal)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing agibot/{signal} stats")
        for key in ("min", "max"):
            values = entry.get(key)
            if not isinstance(values, list) or len(values) != CANONICAL55_DIM:
                raise ValueError(f"agibot/{signal}/{key} must be a {CANONICAL55_DIM}D list")


def merge_agibot_stats(stat_path: Path | str, agibot_stats: Dict[str, Any]) -> None:
    stat_path = Path(stat_path)
    _validate_agibot_stats(agibot_stats)
    if stat_path.exists():
        original_mode = stat_path.stat().st_mode & 0o777
        with stat_path.open("r", encoding="utf-8") as f:
            merged = json.load(f)
    else:
        original_mode = 0o644
        merged = {}
    merged["agibot"] = agibot_stats["agibot"]

    stat_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=stat_path.name + ".", suffix=".tmp", dir=str(stat_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=4, ensure_ascii=False)
            f.write("\n")
        with open(tmp_name, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        _validate_agibot_stats({"agibot": loaded["agibot"]})
        os.chmod(tmp_name, original_mode)
        os.replace(tmp_name, stat_path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AgiBot canonical55 min/max stats and merge into stat.json.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stat-path", type=Path, default=DEFAULT_STAT_PATH)
    parser.add_argument("--stats-key", type=str, default="agibot")
    parser.add_argument("--suffix", type=str, default="_split")
    parser.add_argument("--max-action-offset", type=int, default=48)
    parser.add_argument("--rotvec-sample-conditions", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Compute and print summary without writing stat.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stats_key != "agibot":
        raise ValueError("This script currently writes the fixed 'agibot' stats key.")
    stats = compute_agibot_stats(
        args.root,
        suffix=args.suffix,
        max_action_offset=args.max_action_offset,
        rotvec_sample_conditions=args.rotvec_sample_conditions,
        seed=args.seed,
    )
    _validate_agibot_stats(stats)
    if args.dry_run:
        agibot = stats["agibot"]
        print(
            json.dumps(
                {
                    "state_active_dims": int(sum(agibot["state"]["active_mask"])),
                    "action_active_dims": int(sum(agibot["action"]["active_mask"])),
                    "dataset_count": agibot["state"]["dataset_count"],
                    "episode_stat_rows": agibot["state"]["episode_stat_rows"],
                    "orientation_frames_scanned": agibot["state"]["orientation_frames_scanned"],
                    "rotvec_sample_conditions": agibot["action"]["rotvec_sample_conditions"],
                },
                indent=2,
            )
        )
        return
    merge_agibot_stats(args.stat_path, stats)
    print(f"Wrote AgiBot canonical55 stats to {args.stat_path}")


if __name__ == "__main__":
    main()
