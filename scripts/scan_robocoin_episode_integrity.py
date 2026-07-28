#!/usr/bin/env python3
"""Read-only, resumable integrity scanner for RoboCOIN LeRobot episodes."""

from __future__ import annotations

import json
import traceback as traceback_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data.canonical55 import map_robocoin_named_vector


DEFAULT_ROOT = Path("/root/nas/code/d0/data/robot_data/robocoin")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/motus-multidataset_lap_v2/robocoin_full_scan"
)
DEFAULT_TOLERANCE_S = 0.0001


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class TaskFailure:
    task: str
    code: str
    message: str
    traceback: str | None = None


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    episode_index: int
    status: str
    elapsed_s: float
    parquet: dict[str, object] | None
    videos: tuple[dict[str, object], ...]
    issues: tuple[Issue, ...]
    traceback: str | None = None


@dataclass(frozen=True)
class EpisodeSpec:
    root: str
    task: str
    episode_index: int
    declared_length: int
    fps: float
    tolerance_s: float
    parquet_path: str
    videos: tuple[tuple[str, str], ...]
    action_key: str
    action_names: tuple[str, ...]
    state_key: str
    state_names: tuple[str, ...]
    language_action_path: str | None
    language_action_declared: bool
    t5_embedding_path: str | None
    t5_embedding_declared: bool


@dataclass(frozen=True)
class ParquetMeasurement:
    path: str
    rows: int
    timestamp_min: float | None
    timestamp_max: float | None
    frame_index_min: int | None
    frame_index_max: int | None


def resolve_episode_path(
    task_root: Path,
    template: str,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    """Expand a LeRobot path template and keep it inside the task root."""
    if chunks_size <= 0:
        raise ValueError(f"chunks_size must be positive, got {chunks_size}")
    relative = template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    root = task_root.resolve()
    candidate = (task_root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"resolved path is outside task root: {candidate}")
    return candidate


def _feature_names(features: dict[str, Any], key: str) -> tuple[str, ...]:
    feature = features.get(key, {})
    names = feature.get("names", ()) if isinstance(feature, dict) else ()
    if not isinstance(names, (list, tuple)):
        return ()
    return tuple(str(name) for name in names)


def _auxiliary_path(
    task_root: Path,
    raw_path: object,
    *,
    fallback: str | None = None,
) -> str | None:
    value = raw_path if raw_path is not None else fallback
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = task_root / path
    path = path.resolve()
    if not path.is_relative_to(task_root.resolve()):
        raise ValueError(f"auxiliary path is outside task root: {path}")
    return str(path)


def _discover_task(task_root: Path) -> list[EpisodeSpec]:
    info_path = task_root / "meta" / "info.json"
    episodes_path = task_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise ValueError("meta/info.json and meta/episodes.jsonl are required")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    chunks_size = int(info["chunks_size"])
    data_template = str(info["data_path"])
    video_template = str(info["video_path"])
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("features must be an object")

    action_key = "action" if "action" in features else "actions"
    if action_key not in features:
        raise ValueError("features must declare action or actions")
    state_key = "observation.state" if "observation.state" in features else action_key
    video_keys = sorted(
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    )
    if not video_keys:
        raise ValueError("features must declare at least one video")

    specs: list[EpisodeSpec] = []
    seen: set[int] = set()
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_index = int(episode["episode_index"])
            if episode_index in seen:
                raise ValueError(f"duplicate episode_index {episode_index}")
            seen.add(episode_index)
            declared_length = int(episode["length"])
            declared_language = episode.get("language_action_path") is not None
            declared_t5 = episode.get("t5_embedding_path") is not None
            parquet_path = resolve_episode_path(
                task_root,
                data_template,
                episode_index,
                chunks_size,
            )
            videos = tuple(
                (
                    key,
                    str(
                        resolve_episode_path(
                            task_root,
                            video_template,
                            episode_index,
                            chunks_size,
                            key,
                        )
                    ),
                )
                for key in video_keys
            )
            specs.append(
                EpisodeSpec(
                    root=str(task_root.resolve()),
                    task=task_root.name,
                    episode_index=episode_index,
                    declared_length=declared_length,
                    fps=fps,
                    tolerance_s=DEFAULT_TOLERANCE_S,
                    parquet_path=str(parquet_path),
                    videos=videos,
                    action_key=action_key,
                    action_names=_feature_names(features, action_key),
                    state_key=state_key,
                    state_names=_feature_names(features, state_key),
                    language_action_path=_auxiliary_path(
                        task_root,
                        episode.get("language_action_path"),
                        fallback=f"language_action/episode_{episode_index:06d}.txt",
                    ),
                    language_action_declared=declared_language,
                    t5_embedding_path=_auxiliary_path(
                        task_root,
                        episode.get("t5_embedding_path"),
                    ),
                    t5_embedding_declared=declared_t5,
                )
            )
    return specs


def discover_episodes(
    root: Path,
    task_filter: str | None,
    episode_filter: int | None,
) -> tuple[list[EpisodeSpec], list[TaskFailure]]:
    """Discover all episode descriptors while retaining task-level failures."""
    root = root.resolve()
    episodes: list[EpisodeSpec] = []
    failures: list[TaskFailure] = []
    for task_root in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        if task_filter is not None and task_root.name != task_filter:
            continue
        try:
            task_episodes = _discover_task(task_root)
        except Exception as exc:
            failures.append(
                TaskFailure(
                    task=task_root.name,
                    code="invalid_task_metadata",
                    message=str(exc),
                    traceback=traceback_module.format_exc(),
                )
            )
            continue
        for episode in task_episodes:
            if episode_filter is None or episode.episode_index == episode_filter:
                episodes.append(episode)
    episodes.sort(key=lambda item: (item.task, item.episode_index))
    return episodes, failures


def _issue(code: str, message: str, path: Path | str | None = None) -> Issue:
    return Issue(code=code, message=message, path=str(path) if path is not None else None)


def _numeric_matrix(table: pa.Table, key: str) -> np.ndarray:
    values = table[key].to_pylist()
    if any(value is None for value in values):
        raise ValueError(f"{key} contains null rows")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{key} must be a rank-2 vector column, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains non-finite values")
    return array


def validate_canonical55(spec: EpisodeSpec, table: pa.Table) -> list[Issue]:
    """Validate the same named RoboCOIN-to-canonical55 mapping used in training."""
    issues: list[Issue] = []
    for key, names in (
        (spec.state_key, spec.state_names),
        (spec.action_key, spec.action_names),
    ):
        if key not in table.column_names:
            continue
        try:
            array = _numeric_matrix(table, key)
            if len(names) > array.shape[-1]:
                raise ValueError(
                    f"{key} has width {array.shape[-1]} but {len(names)} names"
                )
            mapped, mask = map_robocoin_named_vector(
                torch.from_numpy(array),
                names,
            )
            if mapped.shape[-1] != 55 or mask.shape != mapped.shape:
                raise ValueError(
                    f"{key} mapping returned mapped={tuple(mapped.shape)}, "
                    f"mask={tuple(mask.shape)}"
                )
            if not bool(mask.any()):
                raise ValueError(f"{key} names map to no canonical55 slots")
        except Exception as exc:
            issues.append(
                _issue(
                    "canonical55_mapping_error",
                    f"{key}: {exc}",
                    spec.parquet_path,
                )
            )
    return issues


def validate_parquet(
    spec: EpisodeSpec,
) -> tuple[ParquetMeasurement | None, list[float], list[Issue]]:
    """Read an episode parquet completely and validate loader-required columns."""
    path = Path(spec.parquet_path)
    if not path.is_file():
        return None, [], [_issue("missing_parquet", "parquet does not exist", path)]
    try:
        table = pq.read_table(path)
    except Exception as exc:
        return None, [], [_issue("invalid_parquet", str(exc), path)]

    rows = table.num_rows
    issues: list[Issue] = []
    if rows == 0:
        issues.append(_issue("empty_parquet", "parquet has zero rows", path))
    if rows != spec.declared_length:
        issues.append(
            _issue(
                "episode_length_mismatch",
                f"declared={spec.declared_length}, parquet={rows}",
                path,
            )
        )

    required_scalar = ("timestamp", "frame_index", "episode_index")
    for key in required_scalar:
        if key not in table.column_names:
            issues.append(_issue("schema_mismatch", f"missing column {key}", path))
    if spec.action_key not in table.column_names:
        issues.append(
            _issue(
                "missing_action",
                f"missing action column {spec.action_key}",
                path,
            )
        )
    if spec.state_key not in table.column_names:
        issues.append(
            _issue(
                "schema_mismatch",
                f"missing state column {spec.state_key}",
                path,
            )
        )

    timestamps: list[float] = []
    timestamp_min = None
    timestamp_max = None
    if "timestamp" in table.column_names:
        try:
            timestamp_array = np.asarray(
                table["timestamp"].to_pylist(),
                dtype=np.float64,
            ).reshape(-1)
            if (
                len(timestamp_array) != rows
                or not np.isfinite(timestamp_array).all()
                or np.any(np.diff(timestamp_array) < 0)
            ):
                raise ValueError("timestamps must be finite and monotonic")
            timestamps = timestamp_array.tolist()
            if len(timestamp_array):
                timestamp_min = float(timestamp_array[0])
                timestamp_max = float(timestamp_array[-1])
        except Exception as exc:
            issues.append(_issue("invalid_timestamp", str(exc), path))

    frame_index_min = None
    frame_index_max = None
    if "frame_index" in table.column_names:
        try:
            frame_indices = np.asarray(
                table["frame_index"].to_pylist(),
                dtype=np.int64,
            ).reshape(-1)
            if not np.array_equal(frame_indices, np.arange(rows, dtype=np.int64)):
                raise ValueError("frame_index must be contiguous from zero")
            if len(frame_indices):
                frame_index_min = int(frame_indices[0])
                frame_index_max = int(frame_indices[-1])
        except Exception as exc:
            issues.append(_issue("invalid_frame_index", str(exc), path))

    if "episode_index" in table.column_names:
        try:
            episode_indices = np.asarray(
                table["episode_index"].to_pylist(),
                dtype=np.int64,
            ).reshape(-1)
            if len(episode_indices) != rows or not np.all(
                episode_indices == spec.episode_index
            ):
                raise ValueError(
                    f"rows do not all belong to episode {spec.episode_index}"
                )
        except Exception as exc:
            issues.append(_issue("episode_index_mismatch", str(exc), path))

    for key, names in (
        (spec.state_key, spec.state_names),
        (spec.action_key, spec.action_names),
    ):
        if key not in table.column_names:
            continue
        try:
            array = _numeric_matrix(table, key)
            if array.shape[0] != rows:
                raise ValueError(f"{key} has {array.shape[0]} rows, expected {rows}")
            if len(names) > array.shape[-1]:
                raise ValueError(
                    f"{key} has width {array.shape[-1]} but {len(names)} names"
                )
        except Exception as exc:
            issues.append(_issue("schema_mismatch", str(exc), path))

    issues.extend(validate_canonical55(spec, table))
    measurement = ParquetMeasurement(
        path=str(path),
        rows=rows,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        frame_index_min=frame_index_min,
        frame_index_max=frame_index_max,
    )
    return measurement, timestamps, issues


def validate_auxiliary(spec: EpisodeSpec) -> list[Issue]:
    """Check optional LAP text and declared T5 artifacts without model loading."""
    issues: list[Issue] = []
    if spec.language_action_path is not None:
        path = Path(spec.language_action_path)
        if not path.is_file():
            if spec.language_action_declared:
                issues.append(
                    _issue(
                        "missing_declared_language_action",
                        "declared language action does not exist",
                        path,
                    )
                )
        else:
            try:
                lines = [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if not lines:
                    raise ValueError("language action has no non-empty lines")
            except Exception as exc:
                issues.append(_issue("invalid_language_action", str(exc), path))

    if spec.t5_embedding_declared and spec.t5_embedding_path is not None:
        path = Path(spec.t5_embedding_path)
        if not path.is_file():
            issues.append(
                _issue(
                    "missing_declared_t5_embedding",
                    "declared T5 embedding does not exist",
                    path,
                )
            )
        else:
            try:
                if path.stat().st_size <= 0:
                    raise ValueError("T5 embedding file is empty")
                with path.open("rb") as handle:
                    handle.read(1)
            except Exception as exc:
                issues.append(_issue("invalid_t5_embedding", str(exc), path))
    return issues
