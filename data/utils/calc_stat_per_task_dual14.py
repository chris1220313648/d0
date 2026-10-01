#!/usr/bin/env python3
"""Generate resumable per-task native-14D min/max and q01/q99 statistics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.canonical55 import build_robocoin_dual_eef14_joint14
from data.lerobot.lerobot_dataset import LeRobotMotusDataset
from data.lerobot.lerobot_interndata_dataset import build_interndata_dual_eef14_joint14


DATASET_TYPES = {
    "robocoin": "lerobot_robocoin",
    "agibot": "lerobot_agibot",
    "interndata": "lerobot_interndata",
}


class ReservoirStats:
    """Exact min/max plus bounded deterministic priority sample for quantiles."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.minimum = np.full(14, np.inf, dtype=np.float64)
        self.maximum = np.full(14, -np.inf, dtype=np.float64)
        self.valid_count = np.zeros(14, dtype=np.int64)
        self.values = np.empty((0, 14), dtype=np.float32)
        self.keys = np.empty((0,), dtype=np.float64)
        self.rows = 0

    def update(self, values: Any, mask: Any, *, sample: bool = True) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1, 14)
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), values.shape).copy()
        mask &= np.isfinite(values)
        if not mask.any():
            return
        safe = np.where(mask, values, np.nan)
        for dim in range(14):
            valid = safe[:, dim][mask[:, dim]]
            if valid.size:
                self.minimum[dim] = min(self.minimum[dim], float(valid.min()))
                self.maximum[dim] = max(self.maximum[dim], float(valid.max()))
                self.valid_count[dim] += int(valid.size)
        self.rows += int(values.shape[0])
        if not sample:
            return
        new_keys = self.rng.random(values.shape[0])
        merged_values = np.concatenate((self.values, safe), axis=0)
        merged_keys = np.concatenate((self.keys, new_keys), axis=0)
        if merged_keys.size > self.capacity:
            keep = np.argpartition(merged_keys, self.capacity - 1)[: self.capacity]
            merged_values = merged_values[keep]
            merged_keys = merged_keys[keep]
        self.values, self.keys = merged_values, merged_keys

    def force_range(self, dims: Iterable[int], minimum: float, maximum: float) -> None:
        for dim in dims:
            if self.valid_count[dim] > 0:
                self.minimum[dim] = min(self.minimum[dim], minimum)
                self.maximum[dim] = max(self.maximum[dim], maximum)

    def finalize(self) -> Dict[str, Any]:
        active = self.valid_count > 0
        if not active.any():
            raise ValueError("No valid 14D values collected")
        q01 = np.zeros(14, dtype=np.float64)
        q99 = np.zeros(14, dtype=np.float64)
        for dim in np.flatnonzero(active):
            sample = self.values[:, dim]
            sample = sample[np.isfinite(sample)]
            q01[dim], q99[dim] = np.quantile(sample, (0.01, 0.99))
        minimum = np.where(active, self.minimum, 0.0)
        maximum = np.where(active, self.maximum, 0.0)
        return {
            "min": minimum.astype(np.float32).tolist(),
            "max": maximum.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
            "active_mask": active.tolist(),
            "valid_count": self.valid_count.tolist(),
            "rows_seen": self.rows,
            "quantile_sample_rows": int(self.values.shape[0]),
        }


def _seed(task_key: str, signal: str, seed: int) -> int:
    digest = hashlib.sha1(f"{task_key}:{signal}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _numpy_column(table: Any, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def _parquets(task_root: Path) -> list[Path]:
    paths = sorted((task_root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {task_root / 'data'}")
    return paths


def _build_robocoin(job: Mapping[str, Any], state: ReservoirStats, action: ReservoirStats) -> str:
    task_root = Path(job["task_root"])
    info = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    action_key = "action" if "action" in features else "actions"
    state_key = "observation.state" if "observation.state" in features else action_key
    state_names = features[state_key].get("names")
    action_names = features[action_key].get("names")
    if not state_names or not action_names:
        raise ValueError(f"RoboCOIN feature names missing in {task_root}")
    for path in _parquets(task_root):
        table = pq.read_table(path, columns=sorted({state_key, action_key}), memory_map=True)
        state_value, state_mask = build_robocoin_dual_eef14_joint14(
            torch.from_numpy(_numpy_column(table, state_key)), state_names, "states"
        )
        action_value, action_mask = build_robocoin_dual_eef14_joint14(
            torch.from_numpy(_numpy_column(table, action_key)), action_names, "actions"
        )
        state.update(state_value.numpy(), state_mask.numpy())
        action.update(action_value.numpy(), action_mask.numpy())
    return "named_eef_or_joint_fallback"


def _build_interndata(job: Mapping[str, Any], state: ReservoirStats, action: ReservoirStats) -> str:
    task_root = Path(job["task_root"])
    info = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
    wanted = [
        key
        for key, feature in (info.get("features", {}) or {}).items()
        if (key in {"action", "observation.state"} or key.startswith(("actions.", "states.")))
        and isinstance(feature, Mapping)
        and feature.get("dtype") not in {"video", "image"}
    ]
    for path in _parquets(task_root):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [key for key in wanted if key in available]
        table = pq.read_table(path, columns=columns, memory_map=True)
        batch = {key: table[key].to_pylist() for key in columns}
        state_value, state_mask = build_interndata_dual_eef14_joint14(batch, "states")
        action_value, action_mask = build_interndata_dual_eef14_joint14(batch, "actions")
        state.update(state_value.numpy(), state_mask.numpy())
        action.update(action_value.numpy(), action_mask.numpy())
    return "eef_or_joint_gripper_fallback"


def _fields(info: Mapping[str, Any], feature: str) -> Dict[str, tuple[int, ...]]:
    descriptions = info["features"][feature].get("field_descriptions", {})
    return {
        str(name): tuple(int(index) for index in desc.get("indices", []))
        for name, desc in descriptions.items()
    }


def _take(raw: np.ndarray, fields: Mapping[str, Sequence[int]], name: str, size: int) -> np.ndarray:
    indices = fields.get(name)
    if not indices or len(indices) < size:
        raise KeyError(f"Missing AgiBot field {name}")
    return raw[:, tuple(indices[:size])].astype(np.float64, copy=False)


def _agibot_state(raw: np.ndarray, fields: Mapping[str, Sequence[int]]) -> np.ndarray:
    joint = _take(raw, fields, "state/joint/position", 14)
    left = _take(raw, fields, "state/left_effector/position", 1)
    right = _take(raw, fields, "state/right_effector/position", 1)
    return np.concatenate((joint[:, :6], left, joint[:, 7:13], right), axis=1).astype(np.float32)


def _agibot_action_parts(raw: np.ndarray, fields: Mapping[str, Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    position = _take(raw, fields, "action/end/position", 6)
    quat = _take(raw, fields, "action/end/orientation", 8)
    left = _take(raw, fields, "action/left_effector/position", 1)
    right = _take(raw, fields, "action/right_effector/position", 1)
    direct = np.zeros((raw.shape[0], 14), dtype=np.float32)
    direct[:, 0:3], direct[:, 7:10] = position[:, 0:3], position[:, 3:6]
    direct[:, 6:7], direct[:, 13:14] = left, right
    return direct, quat


def _relative_rotvec(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    return (Rotation.from_quat(target) * Rotation.from_quat(base).inv()).as_rotvec()


def _build_agibot(job: Mapping[str, Any], state: ReservoirStats, action: ReservoirStats) -> str:
    task_root = Path(job["task_root"])
    info = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
    state_fields, action_fields = _fields(info, "observation.state"), _fields(info, "action")
    paths = _parquets(task_root)
    row_counts = [int(pq.ParquetFile(path).metadata.num_rows) for path in paths]
    horizon = int(job["action_chunk_size"])
    max_conditions = max(1, int(job["capacity"]) // max(1, horizon))
    valid_counts = [max(0, rows - horizon) for rows in row_counts]
    total_valid = sum(valid_counts)
    rng = np.random.default_rng(_seed(job["task_key"], "agibot_conditions", job["seed"]))

    for path, rows, valid_count in zip(paths, row_counts, valid_counts):
        table = pq.read_table(path, columns=["observation.state", "action"], memory_map=True)
        raw_state = _numpy_column(table, "observation.state")
        raw_action = _numpy_column(table, "action")
        state.update(_agibot_state(raw_state, state_fields), np.ones((raw_state.shape[0], 14), dtype=bool))
        direct, quat = _agibot_action_parts(raw_action, action_fields)
        direct_mask = np.zeros_like(direct, dtype=bool)
        direct_mask[:, [0, 1, 2, 6, 7, 8, 9, 13]] = True
        action.update(direct, direct_mask, sample=valid_count <= 0)
        if valid_count <= 0:
            continue
        quota = max(1, round(max_conditions * valid_count / max(1, total_valid)))
        conditions = np.arange(valid_count) if quota >= valid_count else np.sort(
            rng.choice(valid_count, size=quota, replace=False)
        )
        for offset in range(1, horizon + 1):
            target_index = conditions + offset
            values = direct[target_index].copy()
            values[:, 3:6] = _relative_rotvec(quat[target_index, 0:4], quat[conditions, 0:4])
            values[:, 10:13] = _relative_rotvec(quat[target_index, 4:8], quat[conditions, 4:8])
            action.update(values, np.ones_like(values, dtype=bool))
    action.force_range((3, 4, 5, 10, 11, 12), -math.pi, math.pi)
    return "dual_eef_relative_rotvec"


def _build_task(job: Mapping[str, Any]) -> Dict[str, Any]:
    state = ReservoirStats(job["capacity"], _seed(job["task_key"], "state", job["seed"]))
    action = ReservoirStats(job["capacity"], _seed(job["task_key"], "action", job["seed"]))
    layout = {
        "robocoin": _build_robocoin,
        "agibot": _build_agibot,
        "interndata": _build_interndata,
    }[job["dataset"]](job, state, action)
    return {
        "task_key": job["task_key"],
        "task_root": job["task_root"],
        "layout": layout,
        "state": state.finalize(),
        "action": action.finalize(),
    }


def _dataset_entries(config: Mapping[str, Any], selected: set[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    common = config["common"]
    for child in config["dataset"]["datasets"]:
        for dataset, dataset_type in DATASET_TYPES.items():
            if dataset not in selected or child.get("type") != dataset_type:
                continue
            params = child.get("params", {})
            child_common = {**common, **child.get("common", {})}
            result[dataset] = {
                "root": str(Path(params["root"]).resolve()),
                "task_discovery": params.get("task_discovery", {}),
                "action_chunk_size": int(child_common["num_video_frames"])
                * int(child_common["video_action_freq_ratio"]),
                "global_downsample_rate": int(child_common["global_downsample_rate"]),
            }
    missing = selected - set(result)
    if missing:
        raise ValueError(f"Datasets not found in config: {sorted(missing)}")
    return result


def _work_file(work_dir: Path, dataset: str, task_key: str) -> Path:
    digest = hashlib.sha1(task_key.encode()).hexdigest()
    return work_dir / dataset / f"{digest}.json"


def _valid_result(path: Path, task_key: str) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        return row.get("task_key") == task_key and all(
            len(row[signal][key]) == 14
            for signal in ("state", "action")
            for key in ("min", "max", "q01", "q99", "active_mask")
        )
    except Exception:
        return False


def _audit(datasets: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for dataset, payload in datasets.items():
        for task_key, task in payload["tasks"].items():
            for signal in ("state", "action"):
                stats = task[signal]
                for dim, (minimum, maximum, q01, q99, active) in enumerate(
                    zip(stats["min"], stats["max"], stats["q01"], stats["q99"], stats["active_mask"])
                ):
                    if not active:
                        continue
                    q_span = float(q99) - float(q01)
                    ratio = (float(maximum) - float(minimum)) / max(q_span, 1e-12)
                    rows.append({
                        "dataset": dataset,
                        "task_key": task_key,
                        "signal": signal,
                        "dim": dim,
                        "raw_to_q_span_ratio": ratio,
                        "min": minimum,
                        "max": maximum,
                        "q01": q01,
                        "q99": q99,
                    })
    rows.sort(key=lambda row: row["raw_to_q_span_ratio"], reverse=True)
    return {"largest_raw_to_quantile_spans": rows[:100]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/multidataset_lap_v5_vqa.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/utils/per_task_dual14_stats")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "outputs/per_task_dual14_stats_work")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_TYPES), default=sorted(DATASET_TYPES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-quantile-samples", type=int, default=1_000_000)
    parser.add_argument("--max-tasks", type=int, default=0, help="Per-dataset limit for smoke tests; 0 means all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selected = set(args.datasets)
    entries = _dataset_entries(config, selected)
    all_outputs: Dict[str, Any] = {}
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "output_format": "dual_eef14_joint14",
        "normalization_mode": "q01_q99",
        "config": str(args.config.resolve()),
        "datasets": {},
    }

    for dataset in args.datasets:
        entry = entries[dataset]
        root = Path(entry["root"])
        task_keys = LeRobotMotusDataset._discover_task_names(root, entry["task_discovery"])
        if args.max_tasks > 0:
            task_keys = task_keys[: args.max_tasks]
        jobs = []
        completed: Dict[str, Any] = {}
        for task_key in task_keys:
            work_path = _work_file(args.work_dir, dataset, task_key)
            if args.resume and _valid_result(work_path, task_key):
                completed[task_key] = json.loads(work_path.read_text(encoding="utf-8"))
                continue
            jobs.append({
                "dataset": dataset,
                "task_key": task_key,
                "task_root": str(root / task_key),
                "capacity": int(args.max_quantile_samples),
                "seed": int(args.seed),
                **entry,
            })
        print(f"{dataset}: discovered={len(task_keys)} resumed={len(completed)} pending={len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {executor.submit(_build_task, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                completed[result["task_key"]] = result
                _atomic_json(_work_file(args.work_dir, dataset, result["task_key"]), result)
                print(f"{dataset}: completed={len(completed)}/{len(task_keys)} task={result['task_key']}", flush=True)
        if set(completed) != set(task_keys):
            raise RuntimeError(f"{dataset} task coverage mismatch")
        payload = {
            "schema_version": 1,
            "dataset": dataset,
            "root": str(root),
            "output_format": "dual_eef14_joint14",
            "tasks": {key: completed[key] for key in sorted(completed)},
        }
        _atomic_json(args.output_dir / f"{dataset}.json", payload)
        all_outputs[dataset] = payload
        manifest["datasets"][dataset] = {"task_count": len(completed), "file": f"{dataset}.json"}

    _atomic_json(args.output_dir / "manifest.json", manifest)
    _atomic_json(args.output_dir / "audit.json", _audit(all_outputs))
    print(f"Wrote per-task dual14 stats to {args.output_dir}")


if __name__ == "__main__":
    main()
