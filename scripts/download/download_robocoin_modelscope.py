#!/usr/bin/env python3
"""Download RoboCOIN datasets from ModelScope in batches."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen


DEFAULT_TARGET_DIR = Path("/root/nasbak/cjy/robot_raw/robocoin")
DEFAULT_BATCH_SIZE = 50
CONSOLIDATED_URL = (
    "https://huggingface.co/datasets/RoboCOIN/pageAssets/resolve/main/"
    "info/consolidated_datasets.json"
)
HUB_NAMES_URL = (
    "https://huggingface.co/datasets/RoboCOIN/pageAssets/resolve/main/"
    "info/hub_dataset_names.json"
)

def fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_dataset_names(
    datasets: dict[str, Any],
    hub_names: dict[str, Any],
    hub: str = "modelscope",
) -> list[str]:
    mapping = hub_names.get(hub, {})
    if not isinstance(mapping, dict):
        mapping = {}
    return [str(mapping.get(name, name)) for name in datasets.keys()]


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def parse_size_gib(size_text: str) -> float:
    parts = size_text.strip().split()
    if len(parts) < 2:
        return float("inf")
    try:
        value = float(parts[0])
    except ValueError:
        return float("inf")
    unit = parts[1].lower()
    multipliers = {
        "tb": 1024.0,
        "tib": 1024.0,
        "gb": 1.0,
        "gib": 1.0,
        "mb": 1.0 / 1024.0,
        "mib": 1.0 / 1024.0,
    }
    return value * multipliers.get(unit, float("inf"))


def select_smoke_dataset_names(
    datasets: dict[str, Any],
    hub_names: dict[str, Any],
    count: int = 2,
) -> list[str]:
    mapping = hub_names.get("modelscope", {})
    if not isinstance(mapping, dict):
        mapping = {}
    ranked = sorted(
        datasets.items(),
        key=lambda item: (parse_size_gib(str(item[1].get("dataset_size", ""))), item[0]),
    )
    return [str(mapping.get(name, name)) for name, _ in ranked[:count]]


def build_download_command(dataset_names: list[str], target_dir: Path) -> list[str]:
    return [
        "robocoin-download",
        "--hub",
        "modelscope",
        "--ds_lists",
        *dataset_names,
        "--target-dir",
        str(target_dir),
    ]


def log_line(log_file, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    print(line, file=log_file, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RoboCOIN datasets from ModelScope."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Download target directory. Default: {DEFAULT_TARGET_DIR}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Datasets per robocoin-download invocation. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Download the two README example datasets only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned batches without running robocoin-download.",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="1-based batch index to start from. Default: 1",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional maximum number of batches to run.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path. Default: <target-dir>/download_robocoin_modelscope.log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_batch <= 0:
        raise ValueError(f"start_batch must be positive, got {args.start_batch}")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError(f"max_batches must be positive, got {args.max_batches}")

    if not args.dry_run and shutil.which("robocoin-download") is None:
        print(
            "[ERROR] robocoin-download not found. Install with: "
            "python -m pip install -U robocoin modelscope",
            file=sys.stderr,
        )
        return 127

    args.target_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or args.target_dir / "download_robocoin_modelscope.log"

    datasets = fetch_json(CONSOLIDATED_URL)
    hub_names = fetch_json(HUB_NAMES_URL)
    if args.smoke:
        dataset_names = select_smoke_dataset_names(datasets, hub_names, count=2)
        source_summary = "smallest current RoboCOIN DataManager smoke datasets"
    else:
        dataset_names = resolve_dataset_names(datasets, hub_names, hub="modelscope")
        source_summary = f"RoboCOIN DataManager index ({len(datasets)} datasets)"

    all_batches = list(batched(dataset_names, args.batch_size))
    selected_batches = all_batches[args.start_batch - 1 :]
    if args.max_batches is not None:
        selected_batches = selected_batches[: args.max_batches]

    with log_path.open("a", encoding="utf-8") as log_file:
        log_line(log_file, f"source={source_summary}")
        log_line(log_file, f"target_dir={args.target_dir}")
        log_line(log_file, f"total_datasets={len(dataset_names)}")
        log_line(log_file, f"batch_size={args.batch_size}")
        log_line(log_file, f"total_batches={len(all_batches)}")
        log_line(log_file, f"start_batch={args.start_batch}")
        if args.dry_run:
            log_line(log_file, "mode=dry-run")
        else:
            log_line(log_file, "mode=download")

        for offset, batch in enumerate(selected_batches):
            batch_number = args.start_batch + offset
            command = build_download_command(batch, args.target_dir)
            log_line(
                log_file,
                f"batch={batch_number}/{len(all_batches)} datasets={len(batch)} "
                f"first={batch[0]} last={batch[-1]}",
            )
            if args.dry_run:
                log_line(log_file, "command=" + json.dumps(command, ensure_ascii=False))
                continue

            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_line(log_file, f"batch={batch_number} returncode={completed.returncode}")
            if completed.returncode != 0:
                log_line(
                    log_file,
                    "failed_batch_datasets="
                    + json.dumps(batch, ensure_ascii=False),
                )
                return completed.returncode

        log_line(log_file, "download script finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
