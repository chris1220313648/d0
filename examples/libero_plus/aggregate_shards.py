#!/usr/bin/env python3
"""Merge and validate task-disjoint LIBERO-plus evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate_shards(
    suite_dir: Path,
    expected_task_count: int | None,
    allow_partial: bool = False,
) -> dict:
    shard_paths = sorted((suite_dir / "shards").glob("*/results.json"))
    if not shard_paths:
        raise ValueError(f"No shard results found under {suite_dir / 'shards'}")

    suite_name = None
    tasks_by_id: dict[int, dict] = {}
    sources = []
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        shard_suite = shard.get("suite")
        if suite_name is None:
            suite_name = shard_suite
        elif shard_suite != suite_name:
            raise ValueError(
                f"Suite mismatch in {shard_path}: {shard_suite!r} != {suite_name!r}"
            )

        rows = shard.get("tasks")
        if not isinstance(rows, list):
            raise ValueError(f"Invalid tasks list in {shard_path}")
        for row in rows:
            task_id = int(row.get("task_id", -1))
            if task_id < 0:
                raise ValueError(f"Invalid task_id in {shard_path}: {task_id}")
            if task_id in tasks_by_id:
                raise ValueError(
                    f"Duplicate task_id {task_id} in {shard_path} and "
                    f"{tasks_by_id[task_id]['_source']}"
                )
            successes = int(row.get("successes", -1))
            episodes = int(row.get("episodes", -1))
            if episodes <= 0 or not 0 <= successes <= episodes:
                raise ValueError(
                    f"Invalid successes/episodes for task {task_id} in {shard_path}"
                )
            tasks_by_id[task_id] = {**row, "_source": str(shard_path)}
        sources.append(str(shard_path))

    task_ids = sorted(tasks_by_id)
    if expected_task_count is not None:
        expected_ids = set(range(expected_task_count))
        actual_ids = set(task_ids)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if extra:
            raise ValueError(f"Task IDs exceed expected range: {extra[:20]}")
        if missing and not allow_partial:
            raise ValueError(
                f"Missing {len(missing)} task IDs; first missing IDs: {missing[:20]}"
            )

    categories: dict[str, dict[str, int]] = {}
    tasks = []
    total_successes = 0
    total_episodes = 0
    for task_id in task_ids:
        row = dict(tasks_by_id[task_id])
        row.pop("_source")
        category = str(row["category"])
        category_result = categories.setdefault(
            category, {"successes": 0, "episodes": 0}
        )
        category_result["successes"] += int(row["successes"])
        category_result["episodes"] += int(row["episodes"])
        total_successes += int(row["successes"])
        total_episodes += int(row["episodes"])
        tasks.append(row)

    return {
        "suite": suite_name,
        "tasks": tasks,
        "categories": categories,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "aggregation": {
            "expected_task_count": expected_task_count,
            "complete": expected_task_count is None or len(task_ids) == expected_task_count,
            "shards": sources,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_shards(
        suite_dir=args.suite_dir,
        expected_task_count=args.expected_task_count,
        allow_partial=args.allow_partial,
    )
    output_path = args.output or args.suite_dir / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    print(
        f"Aggregated {len(result['tasks'])} tasks / {result['total_episodes']} episodes: "
        f"success_rate={result['success_rate']:.6f} -> {output_path}"
    )


if __name__ == "__main__":
    main()
