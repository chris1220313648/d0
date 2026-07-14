#!/usr/bin/env python3
"""Prepare RoboCOIN LeRobot datasets for LAP training.

The script creates per-episode language action files from RoboCOIN's annotation
columns and writes `language_action_path` back to `meta/episodes.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pyarrow.parquet as pq


DEFAULT_ROOT = Path("/root/nasbak/cjy/robot_raw/robocoin/RoboCOIN")
REQUIRED_ACTION_COLUMNS = (
    "eef_direction_action",
    "eef_velocity_action",
    "eef_acc_mag_action",
    "gripper_mode_action",
    "gripper_activity_action",
)


@dataclass(frozen=True)
class GenerationStats:
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
    candidates: List[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "meta" / "info.json").exists() and (path / "data").exists() and (path / "videos").exists():
            candidates.append(path)
    return candidates


def _load_annotation_map(path: Path, index_key: str, value_key: str) -> Dict[int, str]:
    if not path.exists():
        return {}
    mapping: Dict[int, str] = {}
    for row in load_jsonlines(path):
        if index_key in row and value_key in row:
            mapping[int(row[index_key])] = str(row[value_key])
    return mapping


def load_annotation_maps(dataset_root: Path) -> Dict[str, Dict[int, str]]:
    ann_dir = dataset_root / "annotations"
    return {
        "subtask": _load_annotation_map(ann_dir / "subtask_annotations.jsonl", "subtask_index", "subtask"),
        "direction": _load_annotation_map(ann_dir / "eef_direction_annotation.jsonl", "eef_direction_index", "eef_direction"),
        "velocity": _load_annotation_map(ann_dir / "eef_velocity_annotation.jsonl", "eef_velocity_index", "eef_velocity"),
        "acc": _load_annotation_map(ann_dir / "eef_acc_mag_annotation.jsonl", "eef_acc_mag_index", "eef_acc_mag"),
        "gripper_mode": _load_annotation_map(ann_dir / "gripper_mode_annotation.jsonl", "gripper_mode_index", "gripper_mode"),
        "gripper_activity": _load_annotation_map(
            ann_dir / "gripper_activity_annotation.jsonl",
            "gripper_activity_index",
            "gripper_activity",
        ),
    }


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _lookup(mapping: Mapping[int, str], values: Sequence[int], index: int, default: str = "unknown") -> str:
    if index >= len(values):
        return default
    return mapping.get(int(values[index]), default)


def _direction_phrase(direction: str) -> str:
    normalized = direction.strip().lower()
    if normalized in {"still", "none", "unknown", "null", ""}:
        return "hold position"
    return f"move {normalized}"


def _velocity_phrase(velocity: str) -> str:
    normalized = velocity.strip().lower()
    if normalized in {"still", "constant", "none", "unknown", "null", ""}:
        return "constant speed"
    return f"at {normalized} speed"


def _acc_phrase(acc: str) -> str:
    normalized = acc.strip().lower()
    if normalized in {"constant", "still", "none", "unknown", "null", ""}:
        return "constant acceleration"
    return normalized


def _gripper_mode_phrase(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in {"unknown", "null", ""}:
        return "gripper state unknown"
    return f"gripper {normalized}"


def _format_arm(
    arm_label: str,
    arm_idx: int,
    row: Mapping[str, Sequence[int]],
    annotations: Mapping[str, Mapping[int, str]],
) -> str:
    direction = _direction_phrase(_lookup(annotations["direction"], row["eef_direction_action"], arm_idx))
    velocity = _velocity_phrase(_lookup(annotations["velocity"], row["eef_velocity_action"], arm_idx))
    acc = _acc_phrase(_lookup(annotations["acc"], row["eef_acc_mag_action"], arm_idx))
    gripper_mode = _gripper_mode_phrase(_lookup(annotations["gripper_mode"], row["gripper_mode_action"], arm_idx))
    gripper_activity = _lookup(annotations["gripper_activity"], row["gripper_activity_action"], arm_idx)
    motion = f"{direction} {velocity}" if velocity.startswith("at ") else f"{direction}, {velocity}"
    return f"{arm_label}: {motion}, {acc}, {gripper_mode}, {gripper_activity}."


def build_language_action_lines(
    frame_rows: Iterable[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[int, str]],
) -> List[str]:
    lines: List[str] = []
    for raw_row in frame_rows:
        row = {key: _as_int_list(raw_row.get(key)) for key in REQUIRED_ACTION_COLUMNS}
        subtask_values = _as_int_list(raw_row.get("subtask_annotation"))

        max_arms = max((len(row[key]) for key in REQUIRED_ACTION_COLUMNS), default=0)
        arm_labels = ["Left arm", "Right arm"]
        parts: List[str] = []
        if subtask_values and annotations.get("subtask"):
            subtask = _lookup(annotations["subtask"], subtask_values, 0, default="")
            if subtask:
                parts.append(f"Current subtask: {subtask}.")

        for arm_idx in range(max_arms):
            label = arm_labels[arm_idx] if arm_idx < len(arm_labels) else f"Arm {arm_idx + 1}"
            parts.append(_format_arm(label, arm_idx, row, annotations))

        lines.append(" ".join(parts).strip() or "Robot continues the current task.")
    return lines


def _read_episode_rows(parquet_path: Path) -> List[Dict[str, Any]]:
    columns = list(REQUIRED_ACTION_COLUMNS) + ["subtask_annotation"]
    schema_names = set(pq.read_schema(parquet_path).names)
    missing = [column for column in REQUIRED_ACTION_COLUMNS if column not in schema_names]
    if missing:
        raise KeyError(f"Missing RoboCOIN annotation columns in {parquet_path}: {missing}")
    read_columns = [column for column in columns if column in schema_names]
    table = pq.read_table(parquet_path, columns=read_columns)
    return [
        {column: table[column][idx].as_py() for column in read_columns}
        for idx in range(table.num_rows)
    ]


def ensure_language_action(
    dataset_root: Path,
    folder_name: str = "language_action",
    overwrite: bool = False,
) -> GenerationStats:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = load_jsonlines(episodes_path)
    annotations = load_annotation_maps(dataset_root)
    out_dir = dataset_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_files = {path.stem: path for path in (dataset_root / "data").glob("chunk-*/episode_*.parquet")}

    updated = 0
    skipped = 0
    for row in episodes:
        ep_idx = int(row["episode_index"])
        stem = f"episode_{ep_idx:06d}"
        rel_path = f"{folder_name}/{stem}.txt"
        out_path = dataset_root / rel_path
        if out_path.exists() and not overwrite:
            row["language_action_path"] = rel_path
            skipped += 1
            continue

        parquet_path = data_files.get(stem)
        if parquet_path is None:
            raise FileNotFoundError(f"Missing parquet for {stem} under {dataset_root / 'data'}")
        lines = build_language_action_lines(_read_episode_rows(parquet_path), annotations)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        row["language_action_path"] = rel_path
        updated += 1

    write_jsonlines_atomic(episodes_path, episodes)
    return GenerationStats(updated=updated, skipped=skipped)


def write_failures(path: Path, failures: Sequence[Mapping[str, str]]) -> None:
    if not failures:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RoboCOIN language_action files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--language-action-dir-name", type=str, default="language_action")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-language-action", action="store_true")
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_ROOT.parent / "prepare_robocoin_failures.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = discover_dataset_roots(args.root)
    if args.limit > 0:
        roots = roots[: args.limit]

    print(f"root={args.root}", flush=True)
    print(f"datasets={len(roots)}", flush=True)
    for idx, dataset_root in enumerate(roots, start=1):
        print(f"[{idx}/{len(roots)}] {dataset_root}", flush=True)

    if args.dry_run:
        return 0

    failures: List[Dict[str, str]] = []
    total_updated = 0
    total_skipped = 0
    for idx, dataset_root in enumerate(roots, start=1):
        try:
            stats = ensure_language_action(
                dataset_root=dataset_root,
                folder_name=args.language_action_dir_name,
                overwrite=bool(args.overwrite_language_action),
            )
            total_updated += stats.updated
            total_skipped += stats.skipped
            print(
                f"[{idx}/{len(roots)}] prepared {dataset_root} "
                f"updated={stats.updated} skipped={stats.skipped}",
                flush=True,
            )
        except Exception as err:  # noqa: BLE001
            row = {"path": str(dataset_root), "error": repr(err)}
            failures.append(row)
            write_failures(args.failure_log, failures)
            print(f"[{idx}/{len(roots)}] FAILED {row}", flush=True)

    write_failures(args.failure_log, failures)
    print(
        f"done datasets={len(roots)} updated={total_updated} "
        f"skipped={total_skipped} failures={len(failures)} failure_log={args.failure_log}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
