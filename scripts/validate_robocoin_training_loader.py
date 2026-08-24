#!/usr/bin/env python3
"""Deterministically validate every RoboCOIN episode through the training loader."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback as traceback_module
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.canonical55 import Canonical55Dataset
from data.dataset import collate_fn, create_dataset
from data.lerobot.lerobot_robocoin_dataset import (
    LeRobotRoboCOINDataset,
    RoboCOINEpisodeTarget,
)
from data.lerobot.lerobot_dataset import LeRobotMotusDataset


DEFAULT_CONFIG = Path("configs/robocoin_lap_v2.yaml")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/motus-robocoin_lap_v2/robocoin_training_loader_full_validation"
)
REPORT_FILENAMES = (
    "episode_results.jsonl",
    "good_episodes.jsonl",
    "bad_episodes.jsonl",
    "task_failures.jsonl",
    "summary.json",
    "validation.log",
)
LOGGER = logging.getLogger("robocoin_training_loader_validation")


@dataclass(frozen=True)
class ValidationResult:
    task: str
    episode_index: int
    status: str
    elapsed_s: float
    condition_frame_idx: int | str
    stage: str | None
    error_type: str | None
    message: str | None
    traceback: str | None
    measurements: dict[str, Any]


class _SingleSampleDataset(Dataset):
    def __init__(self) -> None:
        self.current: Mapping[str, Any] | None = None

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _idx: int) -> Mapping[str, Any]:
        if self.current is None:
            raise RuntimeError("No raw sample is available for canonicalization")
        return self.current


class CanonicalizeLoadedSample:
    """Reuse an initialized Canonical55Dataset for an explicitly loaded sample."""

    def __init__(self, canonical_dataset: Canonical55Dataset) -> None:
        self.proxy = _SingleSampleDataset()
        self.canonical_dataset = canonical_dataset
        self.canonical_dataset.dataset = self.proxy

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        self.proxy.current = sample
        return self.canonical_dataset[0]


def _tensor_shapes(value: Any) -> Any:
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, Mapping):
        return {str(key): _tensor_shapes(item) for key, item in value.items()}
    return None


def _assert_finite(value: Any, path: str = "batch") -> None:
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"non-finite tensor at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")


def _validate_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    required = ("first_frame", "video_frames", "action_sequence", "initial_state")
    missing = [key for key in required if batch.get(key) is None]
    if missing:
        raise KeyError(f"collated batch is missing required values: {missing}")
    if batch["action_sequence"].shape[-1] != 55:
        raise ValueError(
            f"canonical action dim must be 55, got {tuple(batch['action_sequence'].shape)}"
        )
    if batch["initial_state"].shape[-1] != 55:
        raise ValueError(
            f"canonical state dim must be 55, got {tuple(batch['initial_state'].shape)}"
        )
    _assert_finite(batch)
    return {
        key: _tensor_shapes(batch.get(key))
        for key in (
            "first_frame",
            "video_frames",
            "initial_state",
            "state_mask",
            "action_sequence",
            "action_mask",
            "language_embedding",
            "vlm_inputs",
        )
        if batch.get(key) is not None
    }


def _failure_result(
    target: RoboCOINEpisodeTarget,
    started: float,
    stage: str,
    exc: BaseException,
) -> ValidationResult:
    return ValidationResult(
        task=target.task,
        episode_index=target.episode_index,
        status="bad",
        elapsed_s=time.monotonic() - started,
        condition_frame_idx="last",
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__)
        ),
        measurements={},
    )


def validate_target(
    dataset: LeRobotRoboCOINDataset,
    canonicalize: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    target: RoboCOINEpisodeTarget,
) -> ValidationResult:
    """Validate one target and turn every data exception into a terminal result."""
    started = time.monotonic()
    try:
        raw_sample = dataset.load_episode_sample(
            target.task_idx,
            target.episode_position,
            condition_frame_idx="last",
        )
    except BaseException as exc:
        return _failure_result(target, started, "sample_load", exc)

    try:
        sample = canonicalize(raw_sample)
    except BaseException as exc:
        return _failure_result(target, started, "canonical55", exc)

    try:
        batch = collate_fn([dict(sample)])
        if batch is None:
            raise ValueError("collate_fn returned None for a non-empty sample")
        measurements = _validate_batch(batch)
    except BaseException as exc:
        return _failure_result(target, started, "collate", exc)

    return ValidationResult(
        task=target.task,
        episode_index=target.episode_index,
        status="good",
        elapsed_s=time.monotonic() - started,
        condition_frame_idx="last",
        stage=None,
        error_type=None,
        message=None,
        traceback=None,
        measurements=measurements,
    )


class ValidationDataset(Dataset):
    def __init__(
        self,
        dataset: LeRobotRoboCOINDataset,
        canonicalize: CanonicalizeLoadedSample,
        targets: Sequence[RoboCOINEpisodeTarget],
    ) -> None:
        self.dataset = dataset
        self.canonicalize = canonicalize
        self.targets = tuple(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> ValidationResult:
        return validate_target(self.dataset, self.canonicalize, self.targets[idx])


def _single_result_collate(batch: list[ValidationResult]) -> ValidationResult:
    if len(batch) != 1:
        raise ValueError(f"validation requires batch_size=1, got {len(batch)}")
    return batch[0]


class ReportWriter:
    def __init__(self, output_dir: Path, *, overwrite: bool, resume: bool) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = any((self.output_dir / name).exists() for name in REPORT_FILENAMES)
        if existing and not overwrite and not resume:
            raise ValueError(
                f"report output already exists; use resume or overwrite: {self.output_dir}"
            )
        if overwrite:
            for name in REPORT_FILENAMES:
                path = self.output_dir / name
                if path.exists():
                    path.unlink()

        self.terminal_keys: set[tuple[str, int]] = set()
        self.task_failure_keys: set[str] = set()
        self.summary: dict[str, Any] = {
            "episodes_discovered": 0,
            "episodes_scanned": 0,
            "episodes_skipped_resume": 0,
            "good": 0,
            "bad": 0,
            "task_failures": 0,
            "errors_by_stage": {},
            "errors_by_type": {},
            "tasks": {},
        }
        if resume:
            self._restore()

        self._all = self._open("episode_results.jsonl")
        self._good = self._open("good_episodes.jsonl")
        self._bad = self._open("bad_episodes.jsonl")
        self._task_failures = self._open("task_failures.jsonl")

    def _open(self, name: str):
        return (self.output_dir / name).open("a", encoding="utf-8")

    def _restore(self) -> None:
        path = self.output_dir / "episode_results.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task"]), int(row["episode_index"]))
            if key in self.terminal_keys:
                raise ValueError(f"duplicate terminal episode key in report: {key}")
            self.terminal_keys.add(key)
            self._accumulate(row)
        task_failures_path = self.output_dir / "task_failures.jsonl"
        if task_failures_path.exists():
            for line in task_failures_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                task = str(row["task"])
                if task in self.task_failure_keys:
                    raise ValueError(f"duplicate task failure in report: {task}")
                self.task_failure_keys.add(task)
                self.summary["task_failures"] += 1

    def _accumulate(self, row: Mapping[str, Any]) -> None:
        status = str(row["status"])
        self.summary["episodes_scanned"] += 1
        self.summary[status] += 1
        task = str(row["task"])
        task_summary = self.summary["tasks"].setdefault(
            task, {"scanned": 0, "good": 0, "bad": 0}
        )
        task_summary["scanned"] += 1
        task_summary[status] += 1
        if status == "bad":
            stage = str(row.get("stage") or "unknown")
            error_type = str(row.get("error_type") or "unknown")
            stages = Counter(self.summary["errors_by_stage"])
            types = Counter(self.summary["errors_by_type"])
            stages[stage] += 1
            types[error_type] += 1
            self.summary["errors_by_stage"] = dict(sorted(stages.items()))
            self.summary["errors_by_type"] = dict(sorted(types.items()))

    def set_discovered(self, count: int) -> None:
        self.summary["episodes_discovered"] = int(count)
        self.summary["episodes_skipped_resume"] = len(self.terminal_keys)
        self._write_summary()

    def record(self, result: ValidationResult) -> None:
        row = asdict(result)
        key = (result.task, result.episode_index)
        if key in self.terminal_keys:
            raise ValueError(f"duplicate terminal episode result: {key}")
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        self._all.write(payload)
        (self._good if result.status == "good" else self._bad).write(payload)
        self._all.flush()
        (self._good if result.status == "good" else self._bad).flush()
        self.terminal_keys.add(key)
        self._accumulate(row)
        self._write_summary()

    def record_task_failure(self, task: str, exc: BaseException) -> None:
        if task in self.task_failure_keys:
            return
        row = {
            "task": task,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback_module.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        self._task_failures.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._task_failures.flush()
        self.task_failure_keys.add(task)
        self.summary["task_failures"] += 1
        self._write_summary()

    def _write_summary(self) -> None:
        path = self.output_dir / "summary.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def close(self) -> None:
        for handle in (self._all, self._good, self._bad, self._task_failures):
            handle.close()

    def __enter__(self) -> "ReportWriter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _configure_logging(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "validation.log", mode="a")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def prepare_validation_config(config_path: Path, task: str | None = None):
    config = OmegaConf.load(config_path)
    if str(config.dataset.type) != "lerobot_robocoin":
        raise ValueError("validation config must use dataset.type=lerobot_robocoin")
    config.dataset.image_aug = False
    config.dataset.max_episodes = None
    config.dataset.params.enable_t5_fallback = False
    if task is not None:
        config.dataset.task_mode = "multi"
        config.dataset.task_name = task
        if config.dataset.params.get("task_discovery") is not None:
            config.dataset.params.task_discovery.enabled = False
    return config


def discover_validation_tasks(
    config: Any,
    task_filter: str | None = None,
) -> list[tuple[str, int]]:
    """Discover exactly the task roots accepted by the eager training loader."""
    root = str(config.dataset.params.root)
    discovery = OmegaConf.to_container(
        config.dataset.params.task_discovery,
        resolve=True,
    )
    if task_filter is None:
        names = LeRobotMotusDataset._discover_task_names(root, discovery)
    else:
        task_root = Path(root) / task_filter
        required = discovery.get("required", ["meta/info.json", "data", "videos"])
        if isinstance(required, str):
            required = [required]
        valid = task_root.is_dir() and all(
            (task_root / str(relative)).exists() for relative in required
        )
        if valid and discovery.get("validate_episodes", False):
            valid = LeRobotMotusDataset._has_complete_local_episodes(task_root)
        names = [task_filter] if valid else []
    tasks: list[tuple[str, int]] = []
    for name in names:
        info_path = Path(root) / name / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        tasks.append((name, int(info["total_episodes"])))
    return tasks


def _load_validation_dataset(config_path: Path, task: str | None = None):
    config = prepare_validation_config(config_path, task=task)
    canonical_dataset = create_dataset(config, val=False)
    if not isinstance(canonical_dataset, Canonical55Dataset):
        raise TypeError("RoboCOIN validation requires the canonical55 dataset wrapper")
    raw_dataset = canonical_dataset.dataset
    if not isinstance(raw_dataset, LeRobotRoboCOINDataset):
        raise TypeError("RoboCOIN validation requires the eager LeRobotRoboCOINDataset")
    canonicalize = CanonicalizeLoadedSample(canonical_dataset)
    return raw_dataset, canonicalize


def run_validation(args: argparse.Namespace) -> int:
    if args.workers < 0:
        print("ERROR: workers must be non-negative", file=sys.stderr)
        return 2
    if args.progress_interval < 1:
        print("ERROR: progress-interval must be positive", file=sys.stderr)
        return 2
    if args.resume and args.overwrite:
        print("ERROR: resume and overwrite are mutually exclusive", file=sys.stderr)
        return 2
    if args.episode is not None and not args.task:
        print("ERROR: --episode requires --task", file=sys.stderr)
        return 2

    try:
        with ReportWriter(args.output_dir, overwrite=args.overwrite, resume=args.resume) as writer:
            _configure_logging(args.output_dir)
            base_config = prepare_validation_config(args.config)
            tasks = discover_validation_tasks(base_config, task_filter=args.task)
            if not tasks:
                raise ValueError("no RoboCOIN episodes matched the requested filters")
            discovered = 1 if args.episode is not None else sum(count for _, count in tasks)
            writer.set_discovered(discovered)
            started = time.monotonic()
            completed = 0
            for task, expected_count in tasks:
                if task in writer.task_failure_keys:
                    continue
                expected_indices = (
                    [args.episode]
                    if args.episode is not None
                    else range(expected_count)
                )
                if all((task, int(index)) in writer.terminal_keys for index in expected_indices):
                    continue
                loader = None
                validation_dataset = None
                canonicalize = None
                raw_dataset = None
                try:
                    raw_dataset, canonicalize = _load_validation_dataset(
                        args.config,
                        task=task,
                    )
                    targets = raw_dataset.validation_episode_targets()
                    if args.episode is not None:
                        targets = [
                            target
                            for target in targets
                            if target.episode_index == args.episode
                        ]
                    if not targets:
                        raise ValueError(
                            f"task {task} has no episode matching {args.episode}"
                        )
                    if args.episode is None and len(targets) != expected_count:
                        raise ValueError(
                            f"task {task} target count mismatch: "
                            f"metadata={expected_count}, loader={len(targets)}"
                        )
                    pending = [
                        target
                        for target in targets
                        if (target.task, target.episode_index) not in writer.terminal_keys
                    ]
                    validation_dataset = ValidationDataset(
                        raw_dataset,
                        canonicalize,
                        pending,
                    )
                    loader = DataLoader(
                        validation_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.workers,
                        pin_memory=False,
                        collate_fn=_single_result_collate,
                        persistent_workers=False,
                    )
                    for result in loader:
                        writer.record(result)
                        completed += 1
                        if completed % args.progress_interval == 0:
                            elapsed = max(time.monotonic() - started, 1e-9)
                            LOGGER.info(
                                "progress scanned=%d/%d good=%d bad=%d rate=%.2f ep/s",
                                writer.summary["episodes_scanned"],
                                discovered,
                                writer.summary["good"],
                                writer.summary["bad"],
                                completed / elapsed,
                            )
                except KeyboardInterrupt:
                    LOGGER.warning("Interrupted; completed results have been persisted")
                    return 130
                except Exception as exc:
                    LOGGER.exception("task validation failed before episode completion: %s", task)
                    writer.record_task_failure(task, exc)
                finally:
                    del loader, validation_dataset, canonicalize, raw_dataset
                    gc.collect()
            LOGGER.info(
                "complete discovered=%d scanned=%d good=%d bad=%d task_failures=%d",
                discovered,
                writer.summary["episodes_scanned"],
                writer.summary["good"],
                writer.summary["bad"],
                writer.summary["task_failures"],
            )
            return 1 if writer.summary["bad"] or writer.summary["task_failures"] else 0
    except Exception as exc:
        LOGGER.exception("validation could not start or complete: %s", exc)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load every RoboCOIN episode once through the eager training path."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_validation(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
