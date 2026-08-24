#!/usr/bin/env python3
"""Prepare an aggregated LeRobot v3 OLA dataset for Motus LAP training.

The script never rewrites source parquet/video files. It validates the v3
schema and writes versioned Motus sidecars under ``<dataset>/motus``:

* ``schema.json``: exact state/action/camera contract
* ``stats.json``: state and relative-action normalization statistics (single root)
* ``language_action/episode_XXXXXX.jsonl``: one LAP answer per source frame
* ``t5_embeddings/task_XXXXXX.pt``: optional offline WAN UMT5 cache
* ``preparation_report.json``: counts and validation summary

Multiple ``--root`` values share one explicitly selected stats file while
keeping LAP/T5 sidecars isolated inside each dataset root.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    from data.lerobot.lerobot_ola_v3_dataset import (
        absolute_eef_to_relative_rpy,
        normalize_joint_grippers,
    )
    from data.lerobot.ola_dataset_manifest import load_manifest
    from data.lerobot.add_t5_cache_to_lerobot_dataset import _encode_t5, _init_wan_t5_encoder
    from data.robotwin2.robotwin_data_convert.robotwin_generate_language_action import _summarize_window
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data.lerobot.lerobot_ola_v3_dataset import (
        absolute_eef_to_relative_rpy,
        normalize_joint_grippers,
    )
    from data.lerobot.ola_dataset_manifest import load_manifest
    from data.lerobot.add_t5_cache_to_lerobot_dataset import _encode_t5, _init_wan_t5_encoder
    from data.robotwin2.robotwin_data_convert.robotwin_generate_language_action import _summarize_window


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonlines_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_episode_metadata(root: Path) -> Dict[int, Dict[str, Any]]:
    import pyarrow.parquet as pq

    episodes: Dict[int, Dict[str, Any]] = {}
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta' / 'episodes'}")
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            episodes[int(row["episode_index"])] = row
    return episodes


def _parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No OLA v3 data parquet under {root / 'data'}")
    return paths


def _to_numpy_column(table: Any, key: str, dtype: np.dtype) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Column {key!r} not found in {table.column_names}")
    return np.asarray(table[key].to_pylist(), dtype=dtype)


def _stats(values: np.ndarray) -> Dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"Expected [N,14] stats input, got {values.shape}")
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
        "count": int(values.shape[0]),
    }


def _language_action_rows(
    relative_actions: np.ndarray,
    frame_indices: np.ndarray,
    window_size: int,
) -> list[Dict[str, Any]]:
    """Describe exactly the same future delta-EEF chunk used by action loss.

    Annotation row ``t`` summarizes model actions ``t+1 .. t+window_size``.
    Tail rows are retained for completeness, although the loader never samples
    conditions without a full future chunk.
    """
    rows: list[Dict[str, Any]] = []
    total = int(relative_actions.shape[0])
    for local_index in range(total):
        start = min(local_index + 1, total - 1)
        stop = min(local_index + 1 + int(window_size), total)
        window = relative_actions[start:stop]
        if window.shape[0] == 0:
            window = relative_actions[-1:]
        text = _summarize_window(
            window_actions=window,
            input_mode="delta",
            input_dir_name="action",
            quat_order="xyzw",
        )
        rows.append(
            {
                "frame_index": int(frame_indices[local_index]),
                "start_frame_index": int(frame_indices[start]),
                "end_frame_index": int(frame_indices[min(stop - 1, total - 1)]),
                "text": text,
            }
        )
    return rows


def _schema(args: argparse.Namespace, info: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "version": "ola_motus_v2",
        "source_format": "lerobot_v3",
        "fps": int(info.get("fps", 30)),
        "state": {
            "key": args.state_key,
            "dim": 14,
            "layout": [
                "left_joint_1_native",
                "left_joint_2_native",
                "left_joint_3_native",
                "left_joint_4_native",
                "left_joint_5_native",
                "left_joint_6_native",
                "left_gripper_0_1",
                "right_joint_1_native",
                "right_joint_2_native",
                "right_joint_3_native",
                "right_joint_4_native",
                "right_joint_5_native",
                "right_joint_6_native",
                "right_gripper_0_1",
            ],
            "joint_unit": "dataset_native",
        },
        "eef_state": {
            "key": args.eef_state_key,
            "dim": 16,
            "layout": "left_xyz_quat_xyzw_gripper__right_xyz_quat_xyzw_gripper",
            "coordinate_frame": "robot_base",
        },
        "absolute_action": {
            "key": args.absolute_action_key,
            "dim": 16,
            "layout": "left_xyz_quat_xyzw_gripper__right_xyz_quat_xyzw_gripper",
            "coordinate_frame": "robot_base",
        },
        "model_action": {
            "dim": 14,
            "layout": "left_dxyz_drpy_xyz_gripper__right_dxyz_drpy_xyz_gripper",
            "coordinate_frame": "robot_base",
            "rotation_unit": "rad",
            "delta_reference": "same_frame_measured_eef",
        },
        "gripper": {
            "closed_raw": float(args.gripper_closed_value),
            "open_raw": float(args.gripper_open_value),
            "normalized_range": [0.0, 1.0],
            "language_open_threshold": 0.5,
        },
        "sampling": {
            "global_downsample_rate": 1,
            "video_action_freq_ratio": int(args.window_size // args.num_video_frames),
            "action_chunk_size": int(args.window_size),
            "num_video_frames": int(args.num_video_frames),
        },
        "cameras": {
            "keys": list(args.camera_keys),
            "layout": "right_front_top__left_wrist_bottom_left__right_wrist_bottom_right",
            "layout_ratio": "front_2_over_3__wrists_1_over_3",
        },
        "language_action": {
            "source": "model_delta_action",
            "condition_offset": 1,
            "window_size": int(args.window_size),
        },
    }


def _prepare_root(
    args: argparse.Namespace,
    root: Path,
    excluded_episode_indices: Sequence[int] = (),
    task_text_filter: str | None = None,
) -> tuple[Dict[str, Any], np.ndarray | None, np.ndarray, Path]:
    import pyarrow.parquet as pq

    root = root.expanduser().resolve()
    info = _read_json(root / "meta" / "info.json")
    if str(info.get("codebase_version")) != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')}")
    features = info.get("features", {})
    for key, expected_dim in (
        (args.state_key, 14),
        (args.eef_state_key, 16),
        (args.absolute_action_key, 16),
    ):
        if key not in features:
            if key == args.state_key and args.allow_missing_joint:
                logger.warning("Joint state column %s is missing; stats will not be written", key)
                continue
            raise KeyError(f"Required OLA feature {key!r} missing from meta/info.json")
        if int(features[key]["shape"][-1]) != expected_dim:
            raise ValueError(f"Feature {key!r} must be {expected_dim}D, got {features[key]['shape']}")
    for key in args.camera_keys:
        if key not in features or features[key].get("dtype") != "video":
            raise KeyError(f"Required synchronized OLA camera missing: {key}")

    episode_meta = _load_episode_metadata(root)
    excluded = {int(index) for index in excluded_episode_indices}
    selected_episode_indices = {
        index
        for index, metadata in episode_meta.items()
        if index not in excluded
        and (
            task_text_filter is None
            or task_text_filter in {str(task) for task in (metadata.get("tasks") or [])}
        )
    }
    if not selected_episode_indices:
        raise ValueError(
            f"No episodes remain for {root} after exclusions={sorted(excluded)} "
            f"and task_text_filter={task_text_filter!r}"
        )
    motus_root = root / args.motus_dir_name
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    task_text: Dict[int, str] = {}
    episodes_processed = 0
    frames_processed = 0
    lap_written = 0

    columns = [
        args.eef_state_key,
        args.absolute_action_key,
        "episode_index",
        "frame_index",
        "task_index",
    ]
    have_joint = args.state_key in features
    if have_joint:
        columns.append(args.state_key)

    for parquet_path in _parquet_paths(root):
        table = pq.read_table(parquet_path, columns=columns, memory_map=True)
        episode_ids = _to_numpy_column(table, "episode_index", np.int64).reshape(-1)
        frame_ids_all = _to_numpy_column(table, "frame_index", np.int64).reshape(-1)
        task_ids_all = _to_numpy_column(table, "task_index", np.int64).reshape(-1)
        for episode_index in np.unique(episode_ids):
            if int(episode_index) not in selected_episode_indices:
                continue
            mask = episode_ids == episode_index
            order = np.argsort(frame_ids_all[mask], kind="stable")
            frame_indices = frame_ids_all[mask][order]
            task_indices = task_ids_all[mask][order]
            measured = _to_numpy_column(table, args.eef_state_key, np.float32)[mask][order]
            commanded = _to_numpy_column(table, args.absolute_action_key, np.float32)[mask][order]
            relative = absolute_eef_to_relative_rpy(
                torch.from_numpy(measured),
                torch.from_numpy(commanded),
                gripper_closed_value=args.gripper_closed_value,
                gripper_open_value=args.gripper_open_value,
            ).numpy()
            action_values.append(relative)

            if have_joint:
                joint = _to_numpy_column(table, args.state_key, np.float32)[mask][order]
                joint = normalize_joint_grippers(
                    torch.from_numpy(joint),
                    gripper_closed_value=args.gripper_closed_value,
                    gripper_open_value=args.gripper_open_value,
                ).numpy()
                state_values.append(joint)

            meta = episode_meta.get(int(episode_index), {})
            tasks = meta.get("tasks") or []
            if tasks:
                task_text.setdefault(int(task_indices[0]), str(tasks[0]))

            if args.write_language_action:
                out_path = motus_root / "language_action" / f"episode_{int(episode_index):06d}.jsonl"
                if args.overwrite or not out_path.exists():
                    rows = _language_action_rows(relative, frame_indices, args.window_size)
                    _write_jsonlines_atomic(out_path, rows)
                    lap_written += 1

            episodes_processed += 1
            frames_processed += int(mask.sum())

    _write_json_atomic(motus_root / "schema.json", _schema(args, info))

    t5_written = 0
    if args.encode_t5:
        if not task_text:
            raise ValueError("No task instructions found in v3 episode metadata")
        device = getattr(args, "_shared_t5_device", None)
        encoder = getattr(args, "_shared_t5_encoder", None)
        if encoder is None:
            device = args.t5_device or ("cuda" if torch.cuda.is_available() else "cpu")
            encoder = _init_wan_t5_encoder(args.wan_path, device=device, text_len=args.t5_text_len)
        t5_dir = motus_root / "t5_embeddings"
        t5_dir.mkdir(parents=True, exist_ok=True)
        for task_index, instruction in sorted(task_text.items()):
            path = t5_dir / f"task_{task_index:06d}.pt"
            if path.exists() and not args.overwrite:
                continue
            torch.save(_encode_t5(encoder, instruction, device=device), path)
            t5_written += 1

    report = {
        "root": str(root),
        "episodes_processed": episodes_processed,
        "frames_processed": frames_processed,
        "language_action_files_written": lap_written,
        "t5_embeddings_written": t5_written,
        "joint_state_available": have_joint,
        "task_count": len(task_text),
        "excluded_episode_indices": sorted(excluded),
        "task_text_filter": task_text_filter,
        "source_episode_count": len(episode_meta),
        "source_files_modified": False,
    }
    state_array = np.concatenate(state_values, axis=0) if state_values else None
    action_array = np.concatenate(action_values, axis=0)
    return report, state_array, action_array, motus_root


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = getattr(args, "manifest", None)
    raw_roots = getattr(args, "roots", None)
    manifest_datasets = None
    if manifest_path:
        if raw_roots:
            raise ValueError("Use either --manifest or --root, not both")
        resolved = load_manifest(manifest_path)
        manifest_datasets = resolved.datasets
        raw_roots = [item.path for item in manifest_datasets]
        configured_stats_path = getattr(args, "shared_stats_path", None)
        if configured_stats_path and Path(configured_stats_path).expanduser().resolve() != resolved.stats_path:
            raise ValueError("--shared-stats-path conflicts with the manifest stats path")
        args.shared_stats_path = str(resolved.stats_path)
    if raw_roots is None:
        legacy_root = getattr(args, "root", None)
        raw_roots = [legacy_root] if isinstance(legacy_root, (str, Path)) else legacy_root
    if not raw_roots:
        raise ValueError("At least one --root is required")

    roots = [Path(root).expanduser().resolve() for root in raw_roots]
    shared_stats_path = getattr(args, "shared_stats_path", None)
    if args.write_stats and len(roots) > 1 and not shared_stats_path:
        raise ValueError("Multiple --root values require --shared-stats-path when stats are enabled")
    if args.encode_t5:
        device = args.t5_device or ("cuda" if torch.cuda.is_available() else "cpu")
        args._shared_t5_device = device
        args._shared_t5_encoder = _init_wan_t5_encoder(
            args.wan_path,
            device=device,
            text_len=args.t5_text_len,
        )
    if manifest_datasets is None:
        prepared = [_prepare_root(args, root) for root in roots]
    else:
        prepared = [
            _prepare_root(
                args,
                root,
                excluded_episode_indices=item.excluded_episode_indices,
                task_text_filter=item.task_text,
            )
            for root, item in zip(roots, manifest_datasets)
        ]
    reports = [entry[0] for entry in prepared]
    state_arrays = [entry[1] for entry in prepared if entry[1] is not None]
    action_arrays = [entry[2] for entry in prepared]

    stats_path: Path | None = None
    if args.write_stats:
        if len(state_arrays) != len(roots):
            raise ValueError("Cannot write normalization stats when a dataset has no joint state")
        if shared_stats_path:
            stats_path = Path(shared_stats_path).expanduser().resolve()
        elif len(roots) == 1:
            stats_path = prepared[0][3] / "stats.json"
        stats = {
            "version": "ola_motus_shared_v1",
            "dataset_roots": [str(root) for root in roots],
            "state": _stats(np.concatenate(state_arrays, axis=0)),
            "action": _stats(np.concatenate(action_arrays, axis=0)),
        }
        _write_json_atomic(stats_path, stats)

    for report, (_, _, _, motus_root) in zip(reports, prepared):
        report["normalization_stats_path"] = str(stats_path) if stats_path else None
        report["shared_normalization_stats"] = bool(stats_path and len(roots) > 1)
        _write_json_atomic(motus_root / "preparation_report.json", report)
        logger.info("OLA preparation complete: %s", report)

    if len(reports) == 1:
        return reports[0]
    return {
        "dataset_count": len(reports),
        "dataset_roots": [str(root) for root in roots],
        "normalization_stats_path": str(stats_path) if stats_path else None,
        "reports": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        dest="roots",
        nargs="+",
        default=None,
        help="One or more LeRobot v3 dataset roots",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Collection or mixture manifest; mutually exclusive with --root",
    )
    parser.add_argument(
        "--shared-stats-path",
        default=None,
        help="Required for multi-root stats; every child loader should use this same file",
    )
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--eef-state-key", default="observation.ee_pose")
    parser.add_argument("--absolute-action-key", default="action.ee_pose")
    parser.add_argument(
        "--camera-keys",
        nargs=3,
        default=[
            "observation.images.right_front",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ],
        metavar=("FRONT", "LEFT_WRIST", "RIGHT_WRIST"),
    )
    parser.add_argument("--gripper-closed-value", type=float, default=0.0)
    parser.add_argument("--gripper-open-value", type=float, default=100.0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--motus-dir-name", default="motus")
    parser.add_argument("--allow-missing-joint", action="store_true")
    parser.add_argument("--write-language-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encode-t5", action="store_true")
    parser.add_argument("--wan-path", default=None)
    parser.add_argument("--t5-device", default=None)
    parser.add_argument("--t5-text-len", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.encode_t5 and not args.wan_path:
        raise ValueError("--wan-path is required with --encode-t5")
    if args.window_size <= 0 or args.window_size % 2 != 0:
        raise ValueError("--window-size must be a positive even number")
    if args.num_video_frames <= 0 or args.window_size % args.num_video_frames != 0:
        raise ValueError("--window-size must be divisible by --num-video-frames")
    prepare(args)


if __name__ == "__main__":
    main()
