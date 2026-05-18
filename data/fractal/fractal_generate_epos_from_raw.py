#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Fractal `epos/*.pt` from original RLDS actions only.

Data source:
  source_root/<version>/...  (TFDS builder directory)

Output target:
  output_root/<split>/<task_slug>/epos/<episode_id>.pt

Action mapping:
  epos[t] = [
      world_vector(3),
      rotation_delta(3),
      gripper_closedness_action(1),
  ]  # [T, 7]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import torch
import yaml

# Reuse Fractal converter helpers so grouping/slugging stays fully consistent.
try:
    from fractal_rlds_to_motus import (  # type: ignore
        _try_import_tf,
        find_builder_dir,
        first_non_empty_instruction,
        is_invalid_instruction,
        normalize_instruction_group_key,
        slugify_task_name,
    )
except Exception:
    this_dir = str(Path(__file__).resolve().parent)
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    from fractal_rlds_to_motus import (  # type: ignore
        _try_import_tf,
        find_builder_dir,
        first_non_empty_instruction,
        is_invalid_instruction,
        normalize_instruction_group_key,
        slugify_task_name,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GenConfig:
    source_root: str
    output_root: str
    splits: Tuple[str, ...]
    tf_disable_gpu: bool = True
    task_grouping: str = "exact_instruction"
    task_slug_max_len: int = 80
    unknown_task_name: str = "unknown_task"
    skip_invalid_instruction: bool = True
    invalid_instruction_patterns: Tuple[str, ...] = (
        "video frame is not showing",
        "frame is not showing",
        "camera is not showing",
        "image is not showing",
        "nan",
        "none",
        "null",
    )
    max_episodes_per_split: int = 0
    epos_dir_name: str = "epos"
    require_existing_qpos: bool = True
    align_with_qpos: bool = True
    length_mode: str = "trim"  # trim|strict
    overwrite: bool = True
    log_every_n: int = 200


@dataclass
class Stats:
    episodes_seen: int = 0
    skipped_invalid_instruction: int = 0
    skipped_missing_qpos: int = 0
    skipped_exists: int = 0
    mismatched_length: int = 0
    written: int = 0
    failed: int = 0


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    default_cfg = str(Path(__file__).with_name("config_fractal_convert.yml"))
    parser = argparse.ArgumentParser(
        description="Generate Fractal epos from original RLDS action.",
    )
    parser.add_argument("--config", type=str, default=default_cfg, help="YAML config path.")
    parser.add_argument("--source_root", type=str, default=None, help="Override source_root in config.")
    parser.add_argument("--output_root", type=str, default=None, help="Override output_root in config.")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to process, e.g. train.")
    parser.add_argument("--max_episodes_per_split", type=int, default=None, help="Override max episodes per split.")
    parser.add_argument("--epos_dir_name", type=str, default="epos", help="Output dir name for epos files.")
    parser.add_argument(
        "--task_grouping",
        type=str,
        choices=["normalized_instruction", "exact_instruction"],
        default=None,
        help="Override task grouping mode.",
    )
    parser.add_argument(
        "--no_require_existing_qpos",
        action="store_true",
        help="Allow generating files even if qpos/<episode_id>.pt does not exist.",
    )
    parser.add_argument("--no_align_with_qpos", action="store_true", help="Disable sequence length alignment to qpos.")
    parser.add_argument("--length_mode", type=str, choices=["trim", "strict"], default="trim")
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Skip existing epos files instead of overwriting them.",
    )
    parser.add_argument("--log_every_n", type=int, default=200, help="Log progress every N episodes.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GenConfig:
    cfg_raw = load_yaml(args.config)

    source_root = str(args.source_root or cfg_raw.get("source_root", "")).strip()
    output_root = str(args.output_root or cfg_raw.get("output_root", "")).strip()
    if not source_root:
        raise ValueError("source_root is required.")
    if not output_root:
        raise ValueError("output_root is required.")

    splits_raw = args.splits or cfg_raw.get("splits", ["train"])
    splits = tuple(str(x).strip() for x in splits_raw if str(x).strip())
    if not splits:
        raise ValueError("At least one split is required.")

    max_eps = cfg_raw.get("max_episodes_per_split", 0)
    if args.max_episodes_per_split is not None:
        max_eps = args.max_episodes_per_split

    task_grouping = cfg_raw.get("task_grouping", "exact_instruction")
    if args.task_grouping is not None:
        task_grouping = args.task_grouping

    return GenConfig(
        source_root=source_root,
        output_root=output_root,
        splits=splits,
        tf_disable_gpu=bool(cfg_raw.get("tf_disable_gpu", True)),
        task_grouping=str(task_grouping),
        task_slug_max_len=int(cfg_raw.get("task_slug_max_len", 80)),
        unknown_task_name=str(cfg_raw.get("unknown_task_name", "unknown_task")),
        skip_invalid_instruction=bool(cfg_raw.get("skip_invalid_instruction", True)),
        invalid_instruction_patterns=tuple(
            str(x).strip().lower()
            for x in cfg_raw.get("invalid_instruction_patterns", [])
            if str(x).strip()
        )
        or GenConfig.invalid_instruction_patterns,
        max_episodes_per_split=int(max_eps),
        epos_dir_name=str(args.epos_dir_name),
        require_existing_qpos=not bool(args.no_require_existing_qpos),
        align_with_qpos=not bool(args.no_align_with_qpos),
        length_mode=str(args.length_mode),
        overwrite=not bool(args.no_overwrite),
        log_every_n=max(1, int(args.log_every_n)),
    )


def _load_qpos_len(qpos_path: Path) -> int:
    data = torch.load(str(qpos_path), map_location="cpu")
    if isinstance(data, torch.Tensor):
        if data.ndim != 2:
            raise ValueError(f"qpos tensor must be [T,D], got {tuple(data.shape)} at {qpos_path}")
        return int(data.shape[0])
    raise TypeError(f"Unsupported qpos format at {qpos_path}: {type(data)}")


def _as_action_field(action: dict[str, Any], key: str) -> np.ndarray:
    if key not in action:
        raise KeyError(f"Action missing '{key}' field.")
    return np.asarray(action[key])


def _extract_action_and_instruction(episode_np: dict[str, Any]) -> tuple[np.ndarray, str]:
    steps = episode_np.get("steps", None)
    if steps is None:
        raise KeyError("Episode missing 'steps'.")

    if isinstance(steps, dict):
        obs = steps.get("observation", None)
        act = steps.get("action", None)
        if not isinstance(obs, dict) or not isinstance(act, dict):
            raise KeyError("Episode steps missing observation/action dict.")

        world_vector = _as_action_field(act, "world_vector")
        rotation_delta = _as_action_field(act, "rotation_delta")
        gripper = _as_action_field(act, "gripper_closedness_action")
        instruction = first_non_empty_instruction(obs.get("natural_language_instruction", None))
    else:
        if not hasattr(steps, "__iter__"):
            raise TypeError(f"Unexpected steps type: {type(steps)}")

        world_vector_list: list[np.ndarray] = []
        rotation_delta_list: list[np.ndarray] = []
        gripper_list: list[np.ndarray] = []
        instr_list: list[Any] = []

        for step in steps:
            if not isinstance(step, dict):
                raise TypeError(f"Unexpected step type: {type(step)}")
            obs = step.get("observation", None)
            act = step.get("action", None)
            if not isinstance(obs, dict) or not isinstance(act, dict):
                raise KeyError("Step missing observation/action dict.")

            world_vector_list.append(_as_action_field(act, "world_vector"))
            rotation_delta_list.append(_as_action_field(act, "rotation_delta"))
            gripper_list.append(_as_action_field(act, "gripper_closedness_action"))
            instr_list.append(obs.get("natural_language_instruction", None))

        if not world_vector_list:
            raise ValueError("Episode has no valid steps.")

        world_vector = np.stack(world_vector_list, axis=0)
        rotation_delta = np.stack(rotation_delta_list, axis=0)
        gripper = np.stack(gripper_list, axis=0)
        instruction = first_non_empty_instruction(instr_list)

    if world_vector.ndim != 2 or world_vector.shape[1] != 3:
        raise ValueError(f"world_vector must be [T,3], got {world_vector.shape}")
    if rotation_delta.ndim != 2 or rotation_delta.shape[1] != 3:
        raise ValueError(f"rotation_delta must be [T,3], got {rotation_delta.shape}")

    t = int(world_vector.shape[0])
    if t <= 0:
        raise ValueError("Empty trajectory.")

    gripper = np.asarray(gripper)
    if gripper.ndim == 1:
        gripper = gripper.reshape(-1, 1)
    elif gripper.ndim > 2:
        gripper = gripper.reshape(t, -1)
    if gripper.ndim != 2 or gripper.shape[1] != 1:
        raise ValueError(f"gripper_closedness_action must be [T,1], got {gripper.shape}")

    if rotation_delta.shape[0] != t or gripper.shape[0] != t:
        raise ValueError(
            "Length mismatch among action fields: "
            f"world={world_vector.shape[0]} rot={rotation_delta.shape[0]} grip={gripper.shape[0]}"
        )

    epos = np.concatenate(
        [
            world_vector.astype(np.float32, copy=False),
            rotation_delta.astype(np.float32, copy=False),
            gripper.astype(np.float32, copy=False),
        ],
        axis=1,
    )
    return epos, instruction


def _resolve_task_slug(cfg: GenConfig, instruction: str) -> str:
    if cfg.task_grouping == "normalized_instruction":
        group_key = normalize_instruction_group_key(instruction)
    else:
        group_key = instruction
    return slugify_task_name(
        group_key,
        max_len=cfg.task_slug_max_len,
        fallback=cfg.unknown_task_name,
    )


def _validate_config(cfg: GenConfig) -> None:
    if cfg.task_grouping not in {"normalized_instruction", "exact_instruction"}:
        raise ValueError("task_grouping must be one of normalized_instruction/exact_instruction")
    if cfg.length_mode not in {"trim", "strict"}:
        raise ValueError("length_mode must be trim or strict")
    if cfg.max_episodes_per_split < 0:
        raise ValueError("max_episodes_per_split must be >= 0")
    if not cfg.epos_dir_name.strip():
        raise ValueError("epos_dir_name must be non-empty")


def _run_split(
    cfg: GenConfig,
    tfds: Any,
    ds_split: Iterable[dict[str, Any]],
    split_name: str,
    stats: Stats,
    failures: list[str],
) -> None:
    for episode_idx, episode in enumerate(tfds.as_numpy(ds_split)):
        if cfg.max_episodes_per_split > 0 and episode_idx >= cfg.max_episodes_per_split:
            break

        stats.episodes_seen += 1
        episode_id = f"{episode_idx:08d}"
        try:
            epos, instruction = _extract_action_and_instruction(episode)

            if cfg.skip_invalid_instruction and is_invalid_instruction(instruction, cfg.invalid_instruction_patterns):
                stats.skipped_invalid_instruction += 1
                continue

            task_slug = _resolve_task_slug(cfg, instruction)
            task_dir = Path(cfg.output_root) / split_name / task_slug
            qpos_path = task_dir / "qpos" / f"{episode_id}.pt"
            epos_path = task_dir / cfg.epos_dir_name / f"{episode_id}.pt"

            if cfg.require_existing_qpos and not qpos_path.exists():
                stats.skipped_missing_qpos += 1
                continue

            if cfg.align_with_qpos and qpos_path.exists():
                qlen = _load_qpos_len(qpos_path)
                if qlen != int(epos.shape[0]):
                    stats.mismatched_length += 1
                    if cfg.length_mode == "strict":
                        raise ValueError(
                            f"length mismatch split={split_name} id={episode_id}: qpos={qlen} epos={epos.shape[0]}"
                        )
                    keep = min(qlen, int(epos.shape[0]))
                    if keep <= 0:
                        raise ValueError(
                            f"non-positive aligned length split={split_name} id={episode_id}: "
                            f"qpos={qlen} epos={epos.shape[0]}"
                        )
                    epos = epos[:keep, :]

            if not cfg.overwrite and epos_path.exists():
                stats.skipped_exists += 1
                continue

            epos_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(epos).float(), str(epos_path))
            stats.written += 1

        except Exception as err:  # noqa: BLE001
            stats.failed += 1
            msg = f"[{split_name}] episode={episode_id} failed: {err}"
            failures.append(msg)
            logger.error(msg)

        if stats.episodes_seen % cfg.log_every_n == 0:
            logger.info(
                "[%s] seen=%d written=%d failed=%d skipped_missing_qpos=%d skipped_exists=%d",
                split_name,
                stats.episodes_seen,
                stats.written,
                stats.failed,
                stats.skipped_missing_qpos,
                stats.skipped_exists,
            )


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = build_config(args)
    _validate_config(cfg)

    logger.info("Fractal epos generation-from-raw config:")
    logger.info("  source_root: %s", cfg.source_root)
    logger.info("  output_root: %s", cfg.output_root)
    logger.info("  splits: %s", list(cfg.splits))
    logger.info("  task_grouping: %s", cfg.task_grouping)
    logger.info("  skip_invalid_instruction: %s", cfg.skip_invalid_instruction)
    logger.info("  require_existing_qpos: %s", cfg.require_existing_qpos)
    logger.info("  align_with_qpos: %s (%s)", cfg.align_with_qpos, cfg.length_mode)
    logger.info("  overwrite: %s", cfg.overwrite)

    tf, tfds = _try_import_tf(disable_tf_gpu=cfg.tf_disable_gpu)
    tf.get_logger().setLevel("ERROR")

    builder_dir = find_builder_dir(cfg.source_root)
    builder = tfds.builder_from_directory(builder_dir=builder_dir)

    stats = Stats()
    failures: list[str] = []
    for split in cfg.splits:
        logger.info("Start split=%s", split)
        ds = builder.as_dataset(split=split, shuffle_files=False)
        _run_split(cfg, tfds, ds, split, stats, failures)

    logger.info("Done.")
    logger.info("  episodes_seen=%d", stats.episodes_seen)
    logger.info("  skipped_invalid_instruction=%d", stats.skipped_invalid_instruction)
    logger.info("  skipped_missing_qpos=%d", stats.skipped_missing_qpos)
    logger.info("  skipped_exists=%d", stats.skipped_exists)
    logger.info("  mismatched_length=%d", stats.mismatched_length)
    logger.info("  written=%d", stats.written)
    logger.info("  failed=%d", stats.failed)

    if failures:
        fail_log = Path(cfg.output_root) / "failed_generate_epos_from_raw.log"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failures) + "\n", encoding="utf-8")
        logger.warning("Failure log written to: %s", fail_log)
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
