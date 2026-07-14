#!/usr/bin/env python3
"""Resume RoboCOIN ModelScope downloads and continue past failed datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen


CONSOLIDATED_URL = (
    "https://huggingface.co/datasets/RoboCOIN/pageAssets/resolve/main/"
    "info/consolidated_datasets.json"
)
HUB_NAMES_URL = (
    "https://huggingface.co/datasets/RoboCOIN/pageAssets/resolve/main/"
    "info/hub_dataset_names.json"
)


def log(message: str) -> None:
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    print(f"[{timestamp}] {message}", flush=True)


def fetch_json(url: str, attempts: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network fetch; log exact transient failure
            last_error = exc
            if attempt == attempts:
                break
            wait_seconds = 10 * attempt
            log(f"fetch failed attempt={attempt}/{attempts} url={url} error={exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last_error}") from last_error


def resolve_modelscope_names() -> list[str]:
    datasets = fetch_json(CONSOLIDATED_URL)
    hub_names = fetch_json(HUB_NAMES_URL)
    mapping = hub_names.get("modelscope", {})
    if not isinstance(mapping, dict):
        mapping = {}
    return [str(mapping.get(name, name)) for name in datasets.keys()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=401)
    parser.add_argument("--skip", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_index <= 0:
        raise ValueError(f"start-index must be positive, got {args.start_index}")
    if shutil.which("robocoin-download") is None:
        log("ERROR robocoin-download not found")
        return 127

    args.target_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    log("resume_skip_failed_start")
    log(f"target_dir={args.target_dir}")
    log(f"start_index_1based={args.start_index}")
    log("skip=" + json.dumps(sorted(skip), ensure_ascii=False))

    names = resolve_modelscope_names()
    selected = [
        (index, name)
        for index, name in enumerate(names, 1)
        if index >= args.start_index and name not in skip
    ]
    log(f"total_datasets={len(names)}")
    log(f"selected_datasets={len(selected)}")

    completed = 0
    failed: list[dict[str, Any]] = []
    for position, (index, name) in enumerate(selected, 1):
        log(f"dataset {position}/{len(selected)} index={index} name={name}")
        command = [
            "robocoin-download",
            "--hub",
            "modelscope",
            "--ds_lists",
            name,
            "--target-dir",
            str(args.target_dir),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            completed += 1
            log(f"OK index={index} name={name}")
        else:
            failed.append({"index": index, "name": name, "returncode": result.returncode})
            log(f"FAILED index={index} name={name} returncode={result.returncode}; continuing")

    log(f"resume_skip_failed_finished completed={completed} failed={len(failed)}")
    if failed:
        log("failed_datasets=" + json.dumps(failed, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
