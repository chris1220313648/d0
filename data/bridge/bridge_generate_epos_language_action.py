#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Bridge `epos/*.pt` and `language_action/*.txt` from original RLDS actions.

Data source:
  source_root/<version>/...  (TFDS builder directory)

Output target:
  output_root/{train,test}/<task_slug>/epos/<episode_id>.pt
  output_root/{train,test}/<task_slug>/language_action/<episode_id>.txt

Action mapping:
  epos[t] = [world_vector(3), rotation_delta(3), open_gripper(1)]  # [T, 7]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import torch
import yaml

# Reuse Bridge converter helpers to keep task grouping and TFDS behavior consistent.
try:
    from bridge_rlds_to_motus import (  # type: ignore
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
    from bridge_rlds_to_motus import (  # type: ignore
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
    task_grouping: str = "normalized_instruction"
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
    language_action_dir_name: str = "language_action"
    window_size: int = 16
    overwrite_epos: bool = False
    overwrite_language_action: bool = False
    require_existing_qpos: bool = True
    align_with_qpos: bool = True
    length_mode: str = "trim"  # trim|strict
    log_every_n: int = 200


@dataclass
class Stats:
    episodes_seen: int = 0
    skipped_invalid_instruction: int = 0
    skipped_missing_qpos: int = 0
    skipped_epos_exists: int = 0
    skipped_lang_exists: int = 0
    mismatched_length: int = 0
    epos_written: int = 0
    language_written: int = 0
    failed: int = 0


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    default_cfg = str(Path(__file__).with_name("config_bridge_convert.yml"))
    parser = argparse.ArgumentParser(
        description="Generate bridge epos/language_action from RLDS action.",
    )
    parser.add_argument("--config", type=str, default=default_cfg, help="YAML config path.")
    parser.add_argument("--source_root", type=str, default=None, help="Override source_root in config.")
    parser.add_argument("--output_root", type=str, default=None, help="Override output_root in config.")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to process, e.g. train test.")
    parser.add_argument("--max_episodes_per_split", type=int, default=None, help="Override max episodes per split.")
    parser.add_argument("--window_size", type=int, default=16, help="Sliding window size for language_action.")
    parser.add_argument("--epos_dir_name", type=str, default="epos", help="Output dir name for epos files.")
    parser.add_argument(
        "--language_action_dir_name",
        type=str,
        default="language_action",
        help="Output dir name for language_action files.",
    )
    parser.add_argument(
        "--task_grouping",
        type=str,
        choices=["normalized_instruction", "exact_instruction"],
        default=None,
        help="Override task grouping mode.",
    )
    parser.add_argument("--overwrite_epos", action="store_true", help="Overwrite existing epos files.")
    parser.add_argument("--overwrite_language_action", action="store_true", help="Overwrite existing language_action files.")
    parser.add_argument("--overwrite_all", action="store_true", help="Overwrite both epos and language_action.")
    parser.add_argument(
        "--no_require_existing_qpos",
        action="store_true",
        help="Allow generating files even if qpos/<episode_id>.pt does not exist.",
    )
    parser.add_argument("--no_align_with_qpos", action="store_true", help="Disable sequence length alignment to qpos.")
    parser.add_argument("--length_mode", type=str, choices=["trim", "strict"], default="trim")
    parser.add_argument("--log_every_n", type=int, default=200, help="Log progress every N episodes.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GenConfig:
    cfg_raw = load_yaml(args.config)

    source_root = str(args.source_root or cfg_raw.get("source_root", "")).strip()
    output_root = str(args.output_root or cfg_raw.get("output_root", "")).strip()
    if not source_root:
        raise ValueError("source_root is required.")
    if not output_root:
        raise ValueError("output_root is required.")

    splits_raw = args.splits or cfg_raw.get("splits", ["train", "test"])
    splits = tuple(str(x).strip() for x in splits_raw if str(x).strip())
    if not splits:
        raise ValueError("At least one split is required.")

    max_eps = cfg_raw.get("max_episodes_per_split", 0)
    if args.max_episodes_per_split is not None:
        max_eps = args.max_episodes_per_split

    task_grouping = cfg_raw.get("task_grouping", "normalized_instruction")
    if args.task_grouping is not None:
        task_grouping = args.task_grouping

    overwrite_epos = bool(args.overwrite_epos or args.overwrite_all)
    overwrite_lang = bool(args.overwrite_language_action or args.overwrite_all)

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
        language_action_dir_name=str(args.language_action_dir_name),
        window_size=int(args.window_size),
        overwrite_epos=overwrite_epos,
        overwrite_language_action=overwrite_lang,
        require_existing_qpos=not bool(args.no_require_existing_qpos),
        align_with_qpos=not bool(args.no_align_with_qpos),
        length_mode=str(args.length_mode),
        log_every_n=max(1, int(args.log_every_n)),
    )


def _format_numeric(val: float, sum_decimal: str = "0f") -> str:
    match = re.fullmatch(r"(\d+)f", sum_decimal)
    decimals = int(match.group(1)) if match else 0
    return f"{val:.{decimals}f}"


def _round_to_nearest_n(value: float, n: int = 10) -> int:
    return int(round(value / n) * n)


def _summarize_delta_action_window(delta_window: np.ndarray) -> str:
    if delta_window.ndim != 2 or delta_window.shape[1] != 7:
        raise ValueError(f"Expected delta_window [T,7], got {delta_window.shape}")

    dx_m = float(delta_window[:, 0].sum())
    dy_m = float(delta_window[:, 1].sum())
    dz_m = float(delta_window[:, 2].sum())

    droll_rad = float(delta_window[:, 3].sum())
    dpitch_rad = float(delta_window[:, 4].sum())
    dyaw_rad = float(delta_window[:, 5].sum())

    dx = abs(dx_m * 100.0)
    dy = abs(dy_m * 100.0)
    dz = abs(dz_m * 100.0)
    droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), 10)
    dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), 10)
    dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), 10)

    parts: list[str] = []
    if dx_m > 0 and dx > 0:
        parts.append(f"move forward {_format_numeric(dx)} cm")
    elif dx_m < 0 and dx > 0:
        parts.append(f"move back {_format_numeric(dx)} cm")

    if dz_m > 0 and dz > 0:
        parts.append(f"move up {_format_numeric(dz)} cm")
    elif dz_m < 0 and dz > 0:
        parts.append(f"move down {_format_numeric(dz)} cm")

    if dy_m > 0 and dy > 0:
        parts.append(f"move left {_format_numeric(dy)} cm")
    elif dy_m < 0 and dy > 0:
        parts.append(f"move right {_format_numeric(dy)} cm")

    if droll_rad > 0 and droll > 0:
        parts.append(f"tilt left {droll} degrees")
    elif droll_rad < 0 and droll > 0:
        parts.append(f"tilt right {droll} degrees")

    if dpitch_rad > 0 and dpitch > 0:
        parts.append(f"tilt back {dpitch} degrees")
    elif dpitch_rad < 0 and dpitch > 0:
        parts.append(f"tilt forward {dpitch} degrees")

    if dyaw_rad > 0 and dyaw > 0:
        parts.append(f"rotate counterclockwise {dyaw} degrees")
    elif dyaw_rad < 0 and dyaw > 0:
        parts.append(f"rotate clockwise {dyaw} degrees")

    g_last = float(delta_window[-1, 6])
    parts.append("open gripper" if g_last >= 0.5 else "close gripper")

    if not parts:
        return "hold position"
    return ", ".join(parts)


def _build_language_action_lines(epos: np.ndarray, window_size: int) -> list[str]:
    if epos.ndim != 2 or epos.shape[1] != 7:
        raise ValueError(f"epos must be [T,7], got {epos.shape}")
    t = int(epos.shape[0])
    lines: list[str] = []
    for i in range(t):
        j = min(i + window_size, t)
        lines.append(_summarize_delta_action_window(epos[i:j, :]))
    return lines


def _load_qpos_len(qpos_path: Path) -> int:
    data = torch.load(str(qpos_path), map_location="cpu")
    if isinstance(data, torch.Tensor):
        if data.ndim != 2:
            raise ValueError(f"qpos tensor must be [T,D], got {tuple(data.shape)} at {qpos_path}")
        return int(data.shape[0])
    raise TypeError(f"Unsupported qpos format at {qpos_path}: {type(data)}")


def _extract_action_and_instruction(episode_np: dict[str, Any]) -> tuple[np.ndarray, str]:
    steps = episode_np.get("steps", None)
    if steps is None:
        raise KeyError("Episode missing 'steps'.")

    if isinstance(steps, dict):
        obs = steps.get("observation", None)
        act = steps.get("action", None)
        if not isinstance(obs, dict) or not isinstance(act, dict):
            raise KeyError("Episode steps missing observation/action dict.")

        world_vector = np.asarray(act.get("world_vector", None))
        rotation_delta = np.asarray(act.get("rotation_delta", None))
        open_gripper = np.asarray(act.get("open_gripper", None))
        instruction = first_non_empty_instruction(obs.get("natural_language_instruction", None))
    else:
        if not hasattr(steps, "__iter__"):
            raise TypeError(f"Unexpected steps type: {type(steps)}")

        world_vector_list: list[np.ndarray] = []
        rotation_delta_list: list[np.ndarray] = []
        open_gripper_list: list[Any] = []
        instr_list: list[Any] = []

        for step in steps:
            if not isinstance(step, dict):
                raise TypeError(f"Unexpected step type: {type(step)}")
            obs = step.get("observation", None)
            act = step.get("action", None)
            if not isinstance(obs, dict) or not isinstance(act, dict):
                raise KeyError("Step missing observation/action dict.")

            world_vector_list.append(np.asarray(act.get("world_vector", None)))
            rotation_delta_list.append(np.asarray(act.get("rotation_delta", None)))
            open_gripper_list.append(act.get("open_gripper", None))
            instr_list.append(obs.get("natural_language_instruction", None))

        if not world_vector_list:
            raise ValueError("Episode has no valid steps.")

        world_vector = np.stack(world_vector_list, axis=0)
        rotation_delta = np.stack(rotation_delta_list, axis=0)
        open_gripper = np.asarray(open_gripper_list)
        instruction = first_non_empty_instruction(instr_list)

    if world_vector.ndim != 2 or world_vector.shape[1] != 3:
        raise ValueError(f"world_vector must be [T,3], got {world_vector.shape}")
    if rotation_delta.ndim != 2 or rotation_delta.shape[1] != 3:
        raise ValueError(f"rotation_delta must be [T,3], got {rotation_delta.shape}")
    if open_gripper.ndim != 1:
        open_gripper = open_gripper.reshape(-1)

    t = int(world_vector.shape[0])
    if t <= 0:
        raise ValueError("Empty trajectory.")
    if rotation_delta.shape[0] != t or open_gripper.shape[0] != t:
        raise ValueError(
            "Length mismatch among action fields: "
            f"world={world_vector.shape[0]} rot={rotation_delta.shape[0]} grip={open_gripper.shape[0]}"
        )

    g = open_gripper.astype(np.float32, copy=False).reshape(t, 1)
    epos = np.concatenate(
        [
            world_vector.astype(np.float32, copy=False),
            rotation_delta.astype(np.float32, copy=False),
            g,
        ],
        axis=1,
    )
    return epos, instruction


def _episode_outputs(
    cfg: GenConfig,
    split: str,
    task_slug: str,
    episode_id: str,
) -> tuple[Path, Path, Path]:
    task_dir = Path(cfg.output_root) / split / task_slug
    qpos_path = task_dir / "qpos" / f"{episode_id}.pt"
    epos_path = task_dir / cfg.epos_dir_name / f"{episode_id}.pt"
    lang_path = task_dir / cfg.language_action_dir_name / f"{episode_id}.txt"
    return qpos_path, epos_path, lang_path


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
    if cfg.window_size <= 0:
        raise ValueError("window_size must be > 0")
    if cfg.length_mode not in {"trim", "strict"}:
        raise ValueError("length_mode must be trim or strict")


def _run_split(
    cfg: GenConfig,
    tfds: Any,
    ds_split: Iterable[dict[str, Any]],
    split_name: str,
    stats: Stats,
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
            qpos_path, epos_path, lang_path = _episode_outputs(cfg, split_name, task_slug, episode_id)

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
                            f"non-positive aligned length split={split_name} id={episode_id}: qpos={qlen} epos={epos.shape[0]}"
                        )
                    epos = epos[:keep, :]

            need_write_epos = bool(cfg.overwrite_epos or not epos_path.exists())
            need_write_lang = bool(cfg.overwrite_language_action or not lang_path.exists())

            if not need_write_epos:
                stats.skipped_epos_exists += 1
            if not need_write_lang:
                stats.skipped_lang_exists += 1

            if need_write_epos:
                epos_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.from_numpy(epos), str(epos_path))
                stats.epos_written += 1

            if need_write_lang:
                lines = _build_language_action_lines(epos, window_size=cfg.window_size)
                lang_path.parent.mkdir(parents=True, exist_ok=True)
                lang_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                stats.language_written += 1

        except Exception as err:  # noqa: BLE001
            stats.failed += 1
            logger.error("[%s] episode=%s failed: %s", split_name, episode_id, err)

        if stats.episodes_seen % cfg.log_every_n == 0:
            logger.info(
                "[%s] seen=%d epos_written=%d lang_written=%d failed=%d skipped_missing_qpos=%d",
                split_name,
                stats.episodes_seen,
                stats.epos_written,
                stats.language_written,
                stats.failed,
                stats.skipped_missing_qpos,
            )


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = build_config(args)
    _validate_config(cfg)

    logger.info("Bridge epos/language_action generation config:")
    logger.info("  source_root: %s", cfg.source_root)
    logger.info("  output_root: %s", cfg.output_root)
    logger.info("  splits: %s", list(cfg.splits))
    logger.info("  task_grouping: %s", cfg.task_grouping)
    logger.info("  skip_invalid_instruction: %s", cfg.skip_invalid_instruction)
    logger.info("  require_existing_qpos: %s", cfg.require_existing_qpos)
    logger.info("  align_with_qpos: %s (%s)", cfg.align_with_qpos, cfg.length_mode)
    logger.info("  overwrite_epos: %s", cfg.overwrite_epos)
    logger.info("  overwrite_language_action: %s", cfg.overwrite_language_action)

    tf, tfds = _try_import_tf(disable_tf_gpu=cfg.tf_disable_gpu)
    tf.get_logger().setLevel("ERROR")

    builder_dir = find_builder_dir(cfg.source_root)
    builder = tfds.builder_from_directory(builder_dir=builder_dir)

    stats = Stats()
    for split in cfg.splits:
        logger.info("Start split=%s", split)
        ds = builder.as_dataset(split=split, shuffle_files=False)
        _run_split(cfg, tfds, ds, split, stats)

    logger.info("Done.")
    logger.info("  episodes_seen=%d", stats.episodes_seen)
    logger.info("  skipped_invalid_instruction=%d", stats.skipped_invalid_instruction)
    logger.info("  skipped_missing_qpos=%d", stats.skipped_missing_qpos)
    logger.info("  skipped_epos_exists=%d", stats.skipped_epos_exists)
    logger.info("  skipped_lang_exists=%d", stats.skipped_lang_exists)
    logger.info("  mismatched_length=%d", stats.mismatched_length)
    logger.info("  epos_written=%d", stats.epos_written)
    logger.info("  language_written=%d", stats.language_written)
    logger.info("  failed=%d", stats.failed)
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
