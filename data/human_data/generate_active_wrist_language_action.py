#!/usr/bin/env python3
"""Generate sidecar language_action files for ego active-wrist manifests."""

from __future__ import annotations

import argparse
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq


LOGGER = logging.getLogger(__name__)

EGO3_MANIFESTS = (
    "/root/nasbak2/shuai/ready_data_egoverse/processed/egoverse_active_wrist/manifests/train.jsonl",
    "/root/nasbak2/shuai/ready_data_egoverse/processed/egoverse_active_wrist/manifests/val.jsonl",
    "/root/nasbak/shuai/ready_data/processed/egodex_active_wrist/manifests/train.jsonl",
    "/root/nasbak/shuai/ready_data/processed/egodex_active_wrist/manifests/val.jsonl",
    "/root/nasbak/shuai/ready_data/processed/vitra_active_wrist/manifests/train.jsonl",
    "/root/nasbak/shuai/ready_data/processed/vitra_active_wrist/manifests/val.jsonl",
)


@dataclass
class ManifestStats:
    manifest: str
    output_root: str
    rows_seen: int = 0
    files_written: int = 0
    files_skipped: int = 0
    rows_failed: int = 0
    lines_written: int = 0
    failures: list[dict[str, str]] | None = None

    def add_failure(self, row_id: str, error: Exception) -> None:
        self.rows_failed += 1
        if self.failures is None:
            self.failures = []
        self.failures.append({"id": str(row_id), "error": str(error)})


@dataclass
class ParquetData:
    frame_indices: np.ndarray
    states: np.ndarray
    episode_indices: np.ndarray | None = None


@dataclass
class ManifestProcessConfig:
    manifest: str
    output_root: str
    split_output_root: str
    window_size: int
    min_translation_cm: float
    min_rotation_deg: float


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return quat / norm


def _quat_inverse_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    out = quat.copy()
    out[..., :3] *= -1.0
    return out


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        axis=-1,
    )


def _quat_xyzw_to_rpy_xyz(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    qx, qy, qz, qw = np.moveaxis(quat, -1, 0)
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.stack([roll, pitch, yaw], axis=-1)


def _format_int(value: float) -> str:
    return str(int(round(value)))


def _summarize_wrist_delta(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    *,
    min_translation_cm: float,
    min_rotation_deg: float,
) -> list[str]:
    dxyz = np.asarray(end_pose[:3], dtype=np.float64) - np.asarray(start_pose[:3], dtype=np.float64)
    start_quat = _normalize_quat_xyzw(np.asarray(start_pose[3:7], dtype=np.float64))
    end_quat = _normalize_quat_xyzw(np.asarray(end_pose[3:7], dtype=np.float64))
    delta_quat = _quat_multiply_xyzw(end_quat, _quat_inverse_xyzw(start_quat))
    drpy = _quat_xyzw_to_rpy_xyz(delta_quat)

    dx_cm, dy_cm, dz_cm = dxyz * 100.0
    droll_deg, dpitch_deg, dyaw_deg = drpy * 180.0 / math.pi
    parts: list[str] = []

    if abs(dx_cm) >= min_translation_cm:
        parts.append(f"move {'forward' if dx_cm > 0 else 'back'} {_format_int(abs(dx_cm))} cm")
    if abs(dz_cm) >= min_translation_cm:
        parts.append(f"move {'up' if dz_cm > 0 else 'down'} {_format_int(abs(dz_cm))} cm")
    if abs(dy_cm) >= min_translation_cm:
        parts.append(f"move {'left' if dy_cm > 0 else 'right'} {_format_int(abs(dy_cm))} cm")

    if abs(droll_deg) >= min_rotation_deg:
        parts.append(f"tilt {'left' if droll_deg > 0 else 'right'} {_format_int(abs(droll_deg))} degrees")
    if abs(dpitch_deg) >= min_rotation_deg:
        parts.append(f"tilt {'back' if dpitch_deg > 0 else 'forward'} {_format_int(abs(dpitch_deg))} degrees")
    if abs(dyaw_deg) >= min_rotation_deg:
        parts.append(
            f"rotate {'counterclockwise' if dyaw_deg > 0 else 'clockwise'} "
            f"{_format_int(abs(dyaw_deg))} degrees"
        )
    return parts


def build_active_wrist_language_action_lines(
    states: np.ndarray,
    *,
    window_size: int = 64,
    min_translation_cm: float = 0.5,
    min_rotation_deg: float = 2.0,
) -> list[str]:
    """Build one language-action line per active21 state row."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 21:
        raise ValueError(f"active wrist states must be [T,21], got {states.shape}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    lines: list[str] = []
    last_idx = states.shape[0] - 1
    for idx in range(states.shape[0]):
        end_idx = min(idx + int(window_size), last_idx)
        left_parts = _summarize_wrist_delta(
            states[idx, 7:14],
            states[end_idx, 7:14],
            min_translation_cm=min_translation_cm,
            min_rotation_deg=min_rotation_deg,
        )
        right_parts = _summarize_wrist_delta(
            states[idx, 14:21],
            states[end_idx, 14:21],
            min_translation_cm=min_translation_cm,
            min_rotation_deg=min_rotation_deg,
        )
        if not left_parts and not right_parts:
            lines.append("hold position")
            continue
        left_text = ", ".join(left_parts) if left_parts else "hold position"
        right_text = ", ".join(right_parts) if right_parts else "hold position"
        lines.append(f"Left wrist: {left_text}. Right wrist: {right_text}")
    return lines


def _infer_processed_root(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "manifests":
        return manifest_path.parent.parent
    return manifest_path.parent


def _infer_split(manifest_path: Path) -> str:
    stem = manifest_path.name.split(".", 1)[0]
    if stem.startswith("train"):
        return "train"
    if stem.startswith("val"):
        return "val"
    return stem


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_parquet_data(parquet_path: str | Path) -> ParquetData:
    parquet_file = pq.ParquetFile(parquet_path)
    available_columns = set(parquet_file.schema_arrow.names)
    columns = ["frame_index", "observation.state"]
    if "episode_index" in available_columns:
        columns.append("episode_index")
    table = pq.read_table(parquet_path, columns=columns, memory_map=True)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 21:
        raise ValueError(f"observation.state must be [T,21], got {states.shape}")
    episode_indices = None
    if "episode_index" in table.column_names:
        episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)
    return ParquetData(frame_indices=frame_indices, states=states, episode_indices=episode_indices)


def _select_segment_states(row: dict[str, Any], parquet_data: ParquetData) -> np.ndarray:
    frame_indices = parquet_data.frame_indices
    states = parquet_data.states
    used_episode_filter = False
    if parquet_data.episode_indices is not None and row.get("episode_index") is not None:
        episode_mask = parquet_data.episode_indices == int(row["episode_index"])
        if episode_mask.any():
            states = states[episode_mask]
            frame_indices = frame_indices[episode_mask]
            used_episode_filter = True

    from_index = row.get("dataset_from_index")
    to_index = row.get("dataset_to_index")
    used_dataset_indices = from_index is not None and to_index is not None
    if used_dataset_indices and not used_episode_filter:
        start = max(0, int(from_index))
        end = min(len(states), int(to_index))
        states = states[start:end]
        frame_indices = frame_indices[start:end]

    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    mask = (frame_indices >= start_frame) & (frame_indices < end_frame)
    selected = states[mask]
    if selected.size == 0 and used_dataset_indices:
        local_start = max(0, start_frame)
        local_end = min(len(states), end_frame)
        selected = states[local_start:local_end]
    if selected.size == 0:
        raise ValueError(f"no frames selected for start_frame={start_frame}, end_frame={end_frame}")
    return selected


def _empty_manifest_stats(manifest: str, output_root: str) -> ManifestStats:
    return ManifestStats(manifest=str(manifest), output_root=str(output_root), failures=[])


def _merge_manifest_stats(target: ManifestStats, source: ManifestStats) -> None:
    target.rows_seen += source.rows_seen
    target.files_written += source.files_written
    target.files_skipped += source.files_skipped
    target.rows_failed += source.rows_failed
    target.lines_written += source.lines_written
    if source.failures:
        if target.failures is None:
            target.failures = []
        target.failures.extend(source.failures)


def _process_parquet_group(
    parquet_path: str | None,
    grouped_rows: list[dict[str, Any]],
    cfg: ManifestProcessConfig,
) -> ManifestStats:
    stats = _empty_manifest_stats(cfg.manifest, cfg.output_root)
    if parquet_path is None or not grouped_rows:
        return stats
    try:
        parquet_data = _read_parquet_data(parquet_path)
    except Exception as error:  # noqa: BLE001
        for grouped_row in grouped_rows:
            stats.add_failure(
                str(grouped_row.get("id") or f"row_{grouped_row.get('_row_index', 0):08d}"),
                error,
            )
        LOGGER.warning("Failed to read parquet %s for %s rows: %s", parquet_path, len(grouped_rows), error)
        return stats

    split_output_root = Path(cfg.split_output_root)
    for grouped_row in grouped_rows:
        row_id = str(grouped_row.get("id") or f"row_{grouped_row.get('_row_index', 0):08d}")
        output_path = split_output_root / f"{row_id}.txt"
        try:
            states = _select_segment_states(grouped_row, parquet_data)
            lines = build_active_wrist_language_action_lines(
                states,
                window_size=cfg.window_size,
                min_translation_cm=cfg.min_translation_cm,
                min_rotation_deg=cfg.min_rotation_deg,
            )
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats.files_written += 1
            stats.lines_written += len(lines)
        except Exception as error:  # noqa: BLE001
            stats.add_failure(row_id, error)
            LOGGER.warning("Failed to process %s row %s: %s", cfg.manifest, row_id, error)
    return stats


def process_manifest(
    manifest_path: str | Path,
    *,
    output_root: str | Path | None = None,
    split: str | None = None,
    window_size: int = 64,
    min_translation_cm: float = 0.5,
    min_rotation_deg: float = 2.0,
    overwrite: bool = False,
    max_rows: int | None = None,
    workers: int = 1,
) -> ManifestStats:
    manifest_path = Path(manifest_path).expanduser().resolve()
    split = split or _infer_split(manifest_path)
    processed_root = _infer_processed_root(manifest_path)
    output_root = Path(output_root).expanduser().resolve() if output_root else processed_root / "language_action"
    split_output_root = output_root / split
    split_output_root.mkdir(parents=True, exist_ok=True)

    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    cfg = ManifestProcessConfig(
        manifest=str(manifest_path),
        output_root=str(output_root),
        split_output_root=str(split_output_root),
        window_size=window_size,
        min_translation_cm=min_translation_cm,
        min_rotation_deg=min_rotation_deg,
    )
    stats = _empty_manifest_stats(str(manifest_path), str(output_root))
    groups: list[tuple[str | None, list[dict[str, Any]]]] = []

    def flush_group(parquet_path: str | None, grouped_rows: list[dict[str, Any]]) -> None:
        if parquet_path is None or not grouped_rows:
            return
        groups.append((parquet_path, list(grouped_rows)))

    current_path: str | None = None
    current_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(_load_jsonl(manifest_path)):
        if max_rows is not None and row_idx >= max_rows:
            break
        stats.rows_seen += 1
        row["_row_index"] = row_idx
        row_id = str(row.get("id") or f"row_{row_idx:08d}")
        output_path = split_output_root / f"{row_id}.txt"
        if output_path.exists() and not overwrite:
            stats.files_skipped += 1
            continue
        parquet_path = str(row["data_parquet"])
        if current_path is not None and parquet_path != current_path:
            flush_group(current_path, current_rows)
            current_rows = []
        current_path = parquet_path
        current_rows.append(row)
    flush_group(current_path, current_rows)
    if workers == 1 or len(groups) <= 1:
        for parquet_path, grouped_rows in groups:
            _merge_manifest_stats(stats, _process_parquet_group(parquet_path, grouped_rows, cfg))
        return stats

    with ProcessPoolExecutor(max_workers=min(workers, len(groups))) as executor:
        futures = [
            executor.submit(_process_parquet_group, parquet_path, grouped_rows, cfg)
            for parquet_path, grouped_rows in groups
        ]
        for future in as_completed(futures):
            _merge_manifest_stats(stats, future.result())
    return stats


def _read_segment_states(row: dict[str, Any]) -> np.ndarray:
    """Compatibility helper for tests and ad-hoc inspection."""
    return _select_segment_states(row, _read_parquet_data(row["data_parquet"]))


def _write_reports(stats_by_root: dict[Path, list[ManifestStats]], report_root: str | Path | None = None) -> None:
    if report_root is not None:
        root = Path(report_root).expanduser().resolve()
        report_dir = root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "active_wrist_language_action_summary.json"
        stats_items = [item for root_items in stats_by_root.values() for item in root_items]
        payload = {
            "stats": [asdict(item) for item in stats_items],
            "totals": {
                "rows_seen": sum(item.rows_seen for item in stats_items),
                "files_written": sum(item.files_written for item in stats_items),
                "files_skipped": sum(item.files_skipped for item in stats_items),
                "rows_failed": sum(item.rows_failed for item in stats_items),
                "lines_written": sum(item.lines_written for item in stats_items),
            },
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote report: %s", report_path)
        return

    for processed_root, stats_items in stats_by_root.items():
        report_dir = processed_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "active_wrist_language_action_summary.json"
        payload = {
            "stats": [asdict(item) for item in stats_items],
            "totals": {
                "rows_seen": sum(item.rows_seen for item in stats_items),
                "files_written": sum(item.files_written for item in stats_items),
                "files_skipped": sum(item.files_skipped for item in stats_items),
                "rows_failed": sum(item.rows_failed for item in stats_items),
                "lines_written": sum(item.lines_written for item in stats_items),
            },
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote report: %s", report_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["ego3"], help="Use default EgoVerse/EgoDex/VITRA active-wrist manifests.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest jsonl path. Can be repeated.")
    parser.add_argument("--output-root", help="Optional shared output root. Defaults to each processed_root/language_action.")
    parser.add_argument("--window-size", type=int, default=64, help="Language-action horizon in frames.")
    parser.add_argument("--min-translation-cm", type=float, default=0.5)
    parser.add_argument("--min-rotation-deg", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows", type=int, help="Smoke-test limit per manifest.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel parquet-group workers per manifest.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error(f"--workers must be positive, got {args.workers}")
    return args


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(asctime)s %(levelname)s %(message)s")

    manifests = [Path(path) for path in args.manifest]
    if args.preset == "ego3":
        manifests.extend(Path(path) for path in EGO3_MANIFESTS)
    if not manifests:
        raise SystemExit("No manifests provided. Use --preset ego3 or --manifest PATH.")

    stats_by_root: dict[Path, list[ManifestStats]] = defaultdict(list)
    for manifest in manifests:
        if not manifest.exists():
            LOGGER.warning("Skip missing manifest: %s", manifest)
            continue
        processed_root = _infer_processed_root(manifest.expanduser().resolve())
        stats = process_manifest(
            manifest,
            output_root=args.output_root,
            split=_infer_split(manifest),
            window_size=args.window_size,
            min_translation_cm=args.min_translation_cm,
            min_rotation_deg=args.min_rotation_deg,
            overwrite=args.overwrite,
            max_rows=args.max_rows,
            workers=args.workers,
        )
        stats_by_root[processed_root].append(stats)
        LOGGER.info(
            "%s rows=%d written=%d skipped=%d failed=%d lines=%d",
            manifest,
            stats.rows_seen,
            stats.files_written,
            stats.files_skipped,
            stats.rows_failed,
            stats.lines_written,
        )
    _write_reports(stats_by_root, report_root=args.output_root)


if __name__ == "__main__":
    main()
