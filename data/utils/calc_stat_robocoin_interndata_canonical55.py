#!/usr/bin/env python3
"""Build canonical55 min/max statistics for RoboCOIN and InternData-A1."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.canonical55 import CANONICAL55_DIM, map_robocoin_named_vector
from data.lerobot.lerobot_interndata_dataset import build_interndata_canonical55


class CanonicalStatsAccumulator:
    def __init__(self) -> None:
        self.minimum = torch.full((CANONICAL55_DIM,), float("inf"), dtype=torch.float64)
        self.maximum = torch.full((CANONICAL55_DIM,), float("-inf"), dtype=torch.float64)
        self.active_mask = torch.zeros(CANONICAL55_DIM, dtype=torch.bool)
        self.rows = 0
        self.source_fallback_rows = 0

    def update(
        self,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        mask: torch.Tensor,
        *,
        rows: int = 1,
        source_fallback: bool = False,
    ) -> None:
        minimum = minimum.reshape(-1, CANONICAL55_DIM).to(dtype=torch.float64)
        maximum = maximum.reshape(-1, CANONICAL55_DIM).to(dtype=torch.float64)
        mask = mask.reshape(-1, CANONICAL55_DIM).any(dim=0).to(dtype=torch.bool)
        low = torch.minimum(minimum, maximum).amin(dim=0)
        high = torch.maximum(minimum, maximum).amax(dim=0)
        finite = torch.isfinite(low) & torch.isfinite(high)
        valid = mask & finite
        self.minimum[valid] = torch.minimum(self.minimum[valid], low[valid])
        self.maximum[valid] = torch.maximum(self.maximum[valid], high[valid])
        self.active_mask |= valid
        self.rows += int(rows)
        if source_fallback:
            self.source_fallback_rows += int(rows)

    def update_finalized(self, stats: Mapping[str, Any]) -> None:
        minimum = torch.tensor(stats["min"], dtype=torch.float64)
        maximum = torch.tensor(stats["max"], dtype=torch.float64)
        mask = torch.tensor(stats["active_mask"], dtype=torch.bool)
        self.update(minimum, maximum, mask, rows=int(stats.get("stat_rows", 0)))
        self.source_fallback_rows += int(stats.get("source_fallback_rows", 0))

    def force_range(self, indices: slice, minimum: float, maximum: float) -> None:
        self.minimum[indices] = torch.minimum(
            self.minimum[indices], torch.full_like(self.minimum[indices], minimum)
        )
        self.maximum[indices] = torch.maximum(
            self.maximum[indices], torch.full_like(self.maximum[indices], maximum)
        )
        self.active_mask[indices] = True

    def finalize(self) -> Dict[str, Any]:
        minimum = torch.where(self.active_mask, self.minimum, torch.zeros_like(self.minimum))
        maximum = torch.where(self.active_mask, self.maximum, torch.zeros_like(self.maximum))
        return {
            "min": minimum.to(dtype=torch.float32).tolist(),
            "max": maximum.to(dtype=torch.float32).tolist(),
            "active_mask": self.active_mask.tolist(),
            "action_dim": CANONICAL55_DIM,
            "stat_rows": self.rows,
            "source_fallback_rows": self.source_fallback_rows,
        }


def _stat_count(signal_stats: Mapping[str, Any]) -> int:
    count = signal_stats.get("count", 1)
    if isinstance(count, list):
        return int(count[0]) if count else 1
    return int(count)


def _has_robocoin_quaternion(names: Iterable[str], side: str) -> bool:
    prefixes = (f"{side}_eef_quat_", f"{side}_end_quat_")
    return any(str(name).startswith(prefixes) for name in names)


def build_robocoin_task_stats(
    info: Mapping[str, Any],
    episode_stats_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate one RoboCOIN task's episode metadata into canonical55 stats."""
    features = info.get("features", {}) or {}
    signal_specs = {
        "state": ("observation.state", features.get("observation.state", {}).get("names")),
        "action": ("action", features.get("action", {}).get("names")),
    }
    accumulators = {signal: CanonicalStatsAccumulator() for signal in signal_specs}

    for signal, (feature_key, names) in signal_specs.items():
        if not names:
            continue
        raw_minimum: Optional[np.ndarray] = None
        raw_maximum: Optional[np.ndarray] = None
        total_rows = 0
        for row in episode_stats_rows:
            row_stats = row.get("stats", {}) or {}
            signal_stats = row_stats.get(feature_key)
            if not signal_stats:
                continue
            episode_minimum = np.asarray(signal_stats["min"], dtype=np.float64)
            episode_maximum = np.asarray(signal_stats["max"], dtype=np.float64)
            raw_minimum = episode_minimum if raw_minimum is None else np.minimum(raw_minimum, episode_minimum)
            raw_maximum = episode_maximum if raw_maximum is None else np.maximum(raw_maximum, episode_maximum)
            total_rows += _stat_count(signal_stats)
        if raw_minimum is None or raw_maximum is None:
            continue
        mapped_min, mask_min = map_robocoin_named_vector(
            torch.from_numpy(raw_minimum).float(), names
        )
        mapped_max, mask_max = map_robocoin_named_vector(
            torch.from_numpy(raw_maximum).float(), names
        )
        accumulator = accumulators[signal]
        accumulator.update(mapped_min, mapped_max, mask_min & mask_max, rows=total_rows)
        if _has_robocoin_quaternion(names, "left"):
            accumulator.force_range(slice(17, 20), -math.pi, math.pi)
        if _has_robocoin_quaternion(names, "right"):
            accumulator.force_range(slice(23, 26), -math.pi, math.pi)

    return {signal: accumulator.finalize() for signal, accumulator in accumulators.items()}


def _intern_orientation_slices(columns: Mapping[str, Any], prefix: str) -> Iterable[slice]:
    single_keys = (
        f"{prefix}.tcp_to_robot_pose",
        f"{prefix}.ee_to_robot_pose",
        f"{prefix}.tcp_to_armbase_pose",
        f"{prefix}.ee_to_armbase_pose",
        f"{prefix}.gripper.pose",
    )
    for key in single_keys:
        value = columns.get(key)
        if value is not None and len(value) >= 7:
            yield slice(17, 20)
            break
    for side, rotation_slice in (("left", slice(17, 20)), ("right", slice(23, 26))):
        keys = (
            f"{prefix}.{side}_tcp_to_robot_pose",
            f"{prefix}.{side}_ee_to_robot_pose",
            f"{prefix}.{side}_tcp_to_{side}_armbase_pose",
            f"{prefix}.{side}_ee_to_{side}_armbase_pose",
        )
        if any(columns.get(key) is not None and len(columns[key]) >= 7 for key in keys):
            yield rotation_slice


def build_interndata_task_stats(
    episode_stats_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate one InternData task's column stats using the runtime canonical mapper."""
    accumulators = {
        "state": CanonicalStatsAccumulator(),
        "action": CanonicalStatsAccumulator(),
    }
    raw_minimum: Dict[str, np.ndarray] = {}
    raw_maximum: Dict[str, np.ndarray] = {}
    raw_counts: Dict[str, int] = {}
    for row in episode_stats_rows:
        row_stats = row.get("stats", {}) or {}
        for key, signal_stats in row_stats.items():
            if not isinstance(signal_stats, Mapping):
                continue
            if not (key == "action" or key == "observation.state" or key.startswith(("actions.", "states."))):
                continue
            if "min" not in signal_stats or "max" not in signal_stats:
                continue
            episode_minimum = np.asarray(signal_stats["min"], dtype=np.float64)
            episode_maximum = np.asarray(signal_stats["max"], dtype=np.float64)
            raw_minimum[key] = (
                episode_minimum
                if key not in raw_minimum
                else np.minimum(raw_minimum[key], episode_minimum)
            )
            raw_maximum[key] = (
                episode_maximum
                if key not in raw_maximum
                else np.maximum(raw_maximum[key], episode_maximum)
            )
            raw_counts[key] = raw_counts.get(key, 0) + _stat_count(signal_stats)

    if not raw_minimum:
        return {signal: accumulator.finalize() for signal, accumulator in accumulators.items()}
    minimum_columns = {key: value.tolist() for key, value in raw_minimum.items()}
    maximum_columns = {key: value.tolist() for key, value in raw_maximum.items()}

    action_min, action_min_mask = build_interndata_canonical55(minimum_columns, prefix="actions")
    action_max, action_max_mask = build_interndata_canonical55(maximum_columns, prefix="actions")
    action_mask = action_min_mask & action_max_mask
    if action_mask.any():
        action_rows = max(
            (count for key, count in raw_counts.items() if key == "action" or key.startswith("actions.")),
            default=1,
        )
        accumulators["action"].update(action_min, action_max, action_mask, rows=action_rows)
        for indices in _intern_orientation_slices(minimum_columns, "actions"):
            accumulators["action"].force_range(indices, -math.pi, math.pi)

    state_min, state_min_mask = build_interndata_canonical55(minimum_columns, prefix="states")
    state_max, state_max_mask = build_interndata_canonical55(maximum_columns, prefix="states")
    state_mask = state_min_mask & state_max_mask
    used_fallback = not state_mask.any()
    if used_fallback:
        state_min, state_max, state_mask = action_min, action_max, action_mask
    if state_mask.any():
        state_rows = max(
            (
                count
                for key, count in raw_counts.items()
                if key == "observation.state" or key.startswith("states.")
            ),
            default=1,
        )
        accumulators["state"].update(
            state_min,
            state_max,
            state_mask,
            rows=state_rows,
            source_fallback=used_fallback,
        )
        source_prefix = "actions" if used_fallback else "states"
        for indices in _intern_orientation_slices(minimum_columns, source_prefix):
            accumulators["state"].force_range(indices, -math.pi, math.pi)

    return {signal: accumulator.finalize() for signal, accumulator in accumulators.items()}


def read_filtered_episode_stats(path: Path, feature_keys: Iterable[str]) -> list[Dict[str, Any]]:
    """Decode only requested feature stats, skipping large image-stat objects."""
    decoder = json.JSONDecoder()
    markers = [(key, json.dumps(str(key), ensure_ascii=False)) for key in feature_keys]
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            filtered: Dict[str, Any] = {}
            for key, marker in markers:
                marker_pos = line.find(marker)
                if marker_pos < 0:
                    continue
                colon_pos = line.find(":", marker_pos + len(marker))
                if colon_pos < 0:
                    continue
                value_pos = colon_pos + 1
                while value_pos < len(line) and line[value_pos].isspace():
                    value_pos += 1
                value, _ = decoder.raw_decode(line, value_pos)
                if isinstance(value, Mapping) and "min" in value and "max" in value:
                    filtered[key] = value
            if filtered:
                rows.append({"stats": filtered})
    return rows


def _discover_task_roots(root: Path) -> list[Path]:
    task_roots: list[Path] = []
    for dir_path, dir_names, _ in os.walk(root):
        dir_names.sort()
        task_root = Path(dir_path)
        info_path = task_root / "meta" / "info.json"
        stats_path = task_root / "meta" / "episodes_stats.jsonl"
        if info_path.is_file() and stats_path.is_file():
            task_roots.append(task_root)
            dir_names[:] = []
    return sorted(task_roots)


def _merge_dataset_task_stats(
    task_stats: Iterable[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    accumulators = {
        "state": CanonicalStatsAccumulator(),
        "action": CanonicalStatsAccumulator(),
    }
    task_count = 0
    for stats in task_stats:
        task_count += 1
        for signal in ("state", "action"):
            accumulators[signal].update_finalized(stats[signal])
    result = {signal: accumulator.finalize() for signal, accumulator in accumulators.items()}
    for signal_stats in result.values():
        signal_stats["dataset_count"] = task_count
        signal_stats["stats_mode"] = "episode_minmax_with_physical_rpy_bounds"
    return result


def _build_task_stats_job(job: tuple[Path, str]) -> Dict[str, Dict[str, Any]]:
    task_root, dataset_name = job
    info = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {}) or {}
    if dataset_name == "robocoin":
        feature_keys = [key for key in ("observation.state", "action") if key in features]
    elif dataset_name == "interndata":
        feature_keys = [
            key
            for key in features
            if key == "action"
            or key == "observation.state"
            or key.startswith(("actions.", "states."))
        ]
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    rows = read_filtered_episode_stats(
        task_root / "meta" / "episodes_stats.jsonl",
        feature_keys,
    )
    if dataset_name == "robocoin":
        return build_robocoin_task_stats(info, rows)
    return build_interndata_task_stats(rows)


def build_dataset_stats(
    root: Path,
    dataset_name: str,
    *,
    workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    task_roots = _discover_task_roots(root)
    jobs = [(task_root, dataset_name) for task_root in task_roots]
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            task_results = list(executor.map(_build_task_stats_job, jobs))
    else:
        task_results = [_build_task_stats_job(job) for job in jobs]
    if not task_results:
        raise ValueError(f"No task statistics found under {root}")
    return _merge_dataset_task_stats(task_results)


def merge_stats_file(path: Path, updates: Mapping[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(updates)
    path.write_text(json.dumps(existing, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocoin-root", type=Path, required=True)
    parser.add_argument("--interndata-root", type=Path, required=True)
    parser.add_argument("--stat-path", type=Path, required=True)
    parser.add_argument(
        "--max-orientation-samples",
        type=int,
        default=250000,
        help="Reserved for CLI compatibility; quaternion-derived RPY uses [-pi, pi].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reserved for CLI compatibility; the current calculation is deterministic.",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    updates = {
        "robocoin": build_dataset_stats(
            args.robocoin_root, "robocoin", workers=args.workers
        ),
        "interndata": build_dataset_stats(
            args.interndata_root, "interndata", workers=args.workers
        ),
    }
    for dataset_stats in updates.values():
        for signal_stats in dataset_stats.values():
            signal_stats["workers_requested"] = int(args.workers)
    merge_stats_file(args.stat_path, updates)
    print(f"Wrote canonical55 stats for {', '.join(updates)} to {args.stat_path}")


if __name__ == "__main__":
    main()
