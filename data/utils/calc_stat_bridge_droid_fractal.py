#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_DATASET_ROOTS = {
    "bridge": "data/bridge/bridge_dataset",
    "droid": "data/droid/droid_dataset",
    "fractal": "data/fractal/fractal_dataset",
}

VALID_DATASETS = ("bridge", "droid", "fractal")
VALID_SIGNALS = ("qpos", "epos")


def _worker_init() -> None:
    # Avoid per-process over-subscription from BLAS/OpenMP backends.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _resolve_signal_tensor(obj: Any, signal: str) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        tensor = obj
    elif isinstance(obj, dict):
        candidates = [signal, "joint_action", "action", "actions", "data"]
        tensor = None
        for key in candidates:
            value = obj.get(key, None)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                tensor = value
                break
            try:
                tensor = torch.as_tensor(value)
                break
            except Exception:
                continue
        if tensor is None:
            raise TypeError(f"Unsupported dict payload keys: {list(obj.keys())[:8]}")
    else:
        tensor = torch.as_tensor(obj)

    if tensor.ndim != 2:
        raise ValueError(f"Tensor must be 2D [T, D], got shape={tuple(tensor.shape)}")
    return tensor.to(torch.float32)


def _process_single_file(args: Tuple[str, str]) -> Tuple[str, bool, Optional[np.ndarray], Optional[np.ndarray], Optional[int], str]:
    file_path, signal = args
    try:
        payload = torch.load(file_path, map_location="cpu")
        tensor = _resolve_signal_tensor(payload, signal)

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Contains NaN or Inf")

        file_min = tensor.min(dim=0).values.numpy()
        file_max = tensor.max(dim=0).values.numpy()
        action_dim = int(tensor.shape[1])
        return file_path, True, file_min, file_max, action_dim, ""
    except Exception as e:
        return file_path, False, None, None, None, f"{type(e).__name__}: {e}"


def collect_signal_files(dataset_root: Path, signal: str) -> List[str]:
    files: List[str] = []
    skip_names = {"videos", "instructions", "umt5_wan", ".git", "__pycache__"}

    for current_root, dirnames, filenames in os.walk(dataset_root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in skip_names and not d.startswith(".")]
        if Path(current_root).name != signal:
            continue
        for fn in filenames:
            if fn.endswith(".pt"):
                files.append(str(Path(current_root) / fn))
    files.sort()
    return files


def compute_stats_for_signal(
    dataset_name: str,
    dataset_root: Path,
    signal: str,
    num_workers: int,
    spot_check_count: int,
    max_files: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    start_time = time.time()
    files = collect_signal_files(dataset_root, signal)
    total_scanned = len(files)

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    global_min: Optional[np.ndarray] = None
    global_max: Optional[np.ndarray] = None
    valid_file_count = 0
    expected_dim: Optional[int] = None
    valid_files: List[str] = []
    errors: List[str] = []

    worker_args = [(p, signal) for p in files]

    with mp.Pool(processes=num_workers, initializer=_worker_init) as pool:
        iterator = pool.imap_unordered(_process_single_file, worker_args, chunksize=32)
        for file_path, ok, file_min, file_max, action_dim, err_msg in tqdm(
            iterator, total=len(worker_args), desc=f"{dataset_name}/{signal}"
        ):
            if not ok:
                errors.append(f"{file_path}\t{err_msg}")
                continue

            if expected_dim is None:
                expected_dim = action_dim
            elif action_dim != expected_dim:
                errors.append(
                    f"{file_path}\tValueError: Inconsistent action_dim={action_dim}, expected={expected_dim}"
                )
                continue

            if global_min is None:
                global_min = file_min
                global_max = file_max
            else:
                global_min = np.minimum(global_min, file_min)
                global_max = np.maximum(global_max, file_max)

            valid_file_count += 1
            valid_files.append(file_path)

    # Spot-check random valid files: their local min/max must be within global bounds.
    if global_min is not None and valid_files and spot_check_count > 0:
        sample_files = random.sample(valid_files, min(spot_check_count, len(valid_files)))
        tol = 1e-6
        for p in sample_files:
            try:
                payload = torch.load(p, map_location="cpu")
                tensor = _resolve_signal_tensor(payload, signal)
                local_min = tensor.min(dim=0).values.numpy()
                local_max = tensor.max(dim=0).values.numpy()
                if np.any(local_min < (global_min - tol)) or np.any(local_max > (global_max + tol)):
                    errors.append(f"{p}\tSpotCheckError: local min/max out of global bounds")
            except Exception as e:
                errors.append(f"{p}\tSpotCheckLoadError: {type(e).__name__}: {e}")

    elapsed = time.time() - start_time
    stat = {
        "min": global_min.tolist() if global_min is not None else [],
        "max": global_max.tolist() if global_max is not None else [],
        "file_count": valid_file_count,
        "total_files_scanned": total_scanned,
        "action_dim": int(expected_dim) if expected_dim is not None else 0,
        "processing_time_seconds": elapsed,
        "num_processes_used": int(num_workers),
    }
    return stat, errors


def merge_stats_to_json(stat_path: Path, updates: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    if stat_path.exists():
        with open(stat_path, "r", encoding="utf-8") as f:
            stat_json = json.load(f)
    else:
        stat_json = {}

    for dataset_name, signal_map in updates.items():
        if dataset_name not in stat_json or not isinstance(stat_json.get(dataset_name), dict):
            stat_json[dataset_name] = {}
        for signal_name, values in signal_map.items():
            stat_json[dataset_name][signal_name] = values

    stat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stat_json, f, indent=4, ensure_ascii=False)


def write_error_log(log_path: Path, dataset_name: str, signal_name: str, errors: List[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{dataset_name}/{signal_name} stat build errors\n")
        f.write("=" * 90 + "\n")
        if not errors:
            f.write("No errors.\n")
            return
        for line in errors:
            f.write(line + "\n")


def write_dataset_error_log(log_path: Path, dataset_name: str, signal_errors: Dict[str, List[str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{dataset_name} stat build errors\n")
        f.write("=" * 90 + "\n")
        total = sum(len(v) for v in signal_errors.values())
        f.write(f"Total errors: {total}\n\n")
        for signal_name in sorted(signal_errors.keys()):
            errors = signal_errors[signal_name]
            f.write(f"[{signal_name}] errors={len(errors)}\n")
            if not errors:
                f.write("No errors.\n\n")
                continue
            for line in errors:
                f.write(f"[{signal_name}] {line}\n")
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build qpos/epos min-max stats for bridge/droid/fractal.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", *VALID_DATASETS],
        help="Which dataset to process. Use 'all' or run separately per dataset.",
    )
    parser.add_argument(
        "--signal",
        type=str,
        default="both",
        choices=["both", *VALID_SIGNALS],
        help="Which signal to process. Use 'both' or run separately per signal.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional override root for a single dataset run. If set, --dataset must not be 'all'.",
    )
    parser.add_argument(
        "--stat_path",
        type=str,
        default="data/utils/stat.json",
        help="Path to stat.json to merge updates into.",
    )
    parser.add_argument(
        "--error_log_dir",
        type=str,
        default="data/utils",
        help="Directory for output error logs.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Multiprocessing workers.")
    parser.add_argument(
        "--spot_check_count",
        type=int,
        default=10,
        help="Random valid files to spot-check against global bounds.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Debug option: limit files per (dataset,signal). 0 means no limit.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for spot-check sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.dataset == "all":
        dataset_list = list(VALID_DATASETS)
        if args.dataset_root is not None:
            raise ValueError("--dataset_root can only be used when --dataset is a single dataset.")
    else:
        dataset_list = [args.dataset]

    if args.signal == "both":
        signal_list = list(VALID_SIGNALS)
    else:
        signal_list = [args.signal]

    stat_path = Path(args.stat_path).resolve()
    error_log_dir = Path(args.error_log_dir).resolve()
    max_files = args.max_files if args.max_files > 0 else None

    merged_updates: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dataset_name in dataset_list:
        if args.dataset_root is not None:
            root = Path(args.dataset_root).resolve()
        else:
            root = (Path.cwd() / DEFAULT_DATASET_ROOTS[dataset_name]).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        merged_updates.setdefault(dataset_name, {})
        dataset_error_map: Dict[str, List[str]] = {}
        for signal_name in signal_list:
            print(f"\n[Run] dataset={dataset_name}, signal={signal_name}, root={root}")
            stat, errors = compute_stats_for_signal(
                dataset_name=dataset_name,
                dataset_root=root,
                signal=signal_name,
                num_workers=args.num_workers,
                spot_check_count=args.spot_check_count,
                max_files=max_files,
            )

            merged_updates[dataset_name][signal_name] = stat
            dataset_error_map[signal_name] = errors
            log_name = f"{dataset_name}_{signal_name}_stat_errors.log"
            write_error_log(error_log_dir / log_name, dataset_name, signal_name, errors)

            print(
                f"[Done] {dataset_name}/{signal_name}: "
                f"valid={stat['file_count']}, scanned={stat['total_files_scanned']}, "
                f"action_dim={stat['action_dim']}, errors={len(errors)}"
            )

        write_dataset_error_log(
            error_log_dir / f"{dataset_name}_stat_errors.log",
            dataset_name,
            dataset_error_map,
        )

    merge_stats_to_json(stat_path, merged_updates)
    print(f"\nMerged stats written to: {stat_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
