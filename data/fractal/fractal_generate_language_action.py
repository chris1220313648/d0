#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Fractal language_action/*.txt from existing epos/*.pt files.

Input:
  output_root/<split>/<task_slug>/epos/<episode_id>.pt

Output:
  output_root/<split>/<task_slug>/language_action/<episode_id>.txt
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GenConfig:
    output_root: str = "/data/user/wsong890/user68/cjy/Motus/data/fractal/fractal_dataset"
    splits: tuple[str, ...] = ("train",)
    epos_dir_name: str = "epos"
    language_action_dir_name: str = "language_action"
    window_size: int = 16
    overwrite_language_action: bool = False
    max_episodes_per_split: int = 0
    log_every_n: int = 200


@dataclass
class Stats:
    episodes_seen: int = 0
    skipped_lang_exists: int = 0
    language_written: int = 0
    failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Fractal language_action from existing epos files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/user/wsong890/user68/cjy/Motus/data/fractal/fractal_dataset",
        help="Dataset output root containing split/task/epos.",
    )
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits to process, e.g. train.")
    parser.add_argument("--epos_dir_name", type=str, default="epos", help="Input epos dir name.")
    parser.add_argument(
        "--language_action_dir_name",
        type=str,
        default="language_action",
        help="Output language_action dir name.",
    )
    parser.add_argument("--window_size", type=int, default=16, help="Sliding window size for language_action.")
    parser.add_argument(
        "--overwrite_language_action",
        action="store_true",
        help="Overwrite existing language_action files.",
    )
    parser.add_argument(
        "--max_episodes_per_split",
        type=int,
        default=0,
        help="0 means full split.",
    )
    parser.add_argument("--log_every_n", type=int, default=200, help="Log progress every N episodes.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GenConfig:
    splits = tuple(str(x).strip() for x in args.splits if str(x).strip())
    if not splits:
        raise ValueError("At least one split is required.")

    cfg = GenConfig(
        output_root=str(args.output_root).strip(),
        splits=splits,
        epos_dir_name=str(args.epos_dir_name).strip(),
        language_action_dir_name=str(args.language_action_dir_name).strip(),
        window_size=int(args.window_size),
        overwrite_language_action=bool(args.overwrite_language_action),
        max_episodes_per_split=int(args.max_episodes_per_split),
        log_every_n=max(1, int(args.log_every_n)),
    )
    if not cfg.output_root:
        raise ValueError("output_root is required.")
    if cfg.window_size <= 0:
        raise ValueError("window_size must be > 0")
    if cfg.max_episodes_per_split < 0:
        raise ValueError("max_episodes_per_split must be >= 0")
    if not cfg.epos_dir_name:
        raise ValueError("epos_dir_name must be non-empty")
    if not cfg.language_action_dir_name:
        raise ValueError("language_action_dir_name must be non-empty")
    return cfg


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


def _load_epos_tensor(epos_path: Path) -> np.ndarray:
    try:
        data = torch.load(str(epos_path), map_location="cpu")
    except Exception as err:  # noqa: BLE001
        raise RuntimeError(f"failed to load epos file {epos_path}: {err}") from err
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Unsupported epos format at {epos_path}: {type(data)}")
    if data.ndim != 2 or int(data.shape[1]) != 7:
        raise ValueError(f"epos tensor must be [T,7], got {tuple(data.shape)} at {epos_path}")
    if int(data.shape[0]) <= 0:
        raise ValueError(f"epos tensor is empty at {epos_path}")
    return data.detach().cpu().numpy().astype(np.float32, copy=False)


def _run_split(cfg: GenConfig, split: str, stats: Stats, failures: list[str]) -> None:
    split_dir = Path(cfg.output_root) / split
    if not split_dir.exists():
        logger.warning("Split directory does not exist, skip: %s", split_dir)
        return

    task_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    seen_in_split = 0
    for task_dir in task_dirs:
        epos_dir = task_dir / cfg.epos_dir_name
        lang_dir = task_dir / cfg.language_action_dir_name
        if not epos_dir.exists():
            continue

        epos_files = sorted(epos_dir.glob("*.pt"))
        for epos_path in epos_files:
            if cfg.max_episodes_per_split > 0 and seen_in_split >= cfg.max_episodes_per_split:
                return

            seen_in_split += 1
            stats.episodes_seen += 1
            episode_id = epos_path.stem
            lang_path = lang_dir / f"{episode_id}.txt"

            if lang_path.exists() and not cfg.overwrite_language_action:
                stats.skipped_lang_exists += 1
                if stats.episodes_seen % cfg.log_every_n == 0:
                    logger.info(
                        "[%s] seen=%d lang_written=%d skipped_lang_exists=%d failed=%d",
                        split,
                        stats.episodes_seen,
                        stats.language_written,
                        stats.skipped_lang_exists,
                        stats.failed,
                    )
                continue

            try:
                epos = _load_epos_tensor(epos_path)
                lines = _build_language_action_lines(epos, window_size=cfg.window_size)
                lang_path.parent.mkdir(parents=True, exist_ok=True)
                lang_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                stats.language_written += 1
            except Exception as err:  # noqa: BLE001
                stats.failed += 1
                msg = f"[{split}] task={task_dir.name} episode={episode_id} failed: {err}"
                logger.error(msg)
                failures.append(msg)

            if stats.episodes_seen % cfg.log_every_n == 0:
                logger.info(
                    "[%s] seen=%d lang_written=%d skipped_lang_exists=%d failed=%d",
                    split,
                    stats.episodes_seen,
                    stats.language_written,
                    stats.skipped_lang_exists,
                    stats.failed,
                )


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = build_config(args)
    logger.info("Fractal language_action generation config:")
    logger.info("  output_root: %s", cfg.output_root)
    logger.info("  splits: %s", list(cfg.splits))
    logger.info("  epos_dir_name: %s", cfg.epos_dir_name)
    logger.info("  language_action_dir_name: %s", cfg.language_action_dir_name)
    logger.info("  window_size: %d", cfg.window_size)
    logger.info("  overwrite_language_action: %s", cfg.overwrite_language_action)
    logger.info("  max_episodes_per_split: %d", cfg.max_episodes_per_split)

    stats = Stats()
    failures: list[str] = []
    for split in cfg.splits:
        logger.info("Start split=%s", split)
        _run_split(cfg, split, stats, failures)

    logger.info("Done.")
    logger.info("  episodes_seen=%d", stats.episodes_seen)
    logger.info("  skipped_lang_exists=%d", stats.skipped_lang_exists)
    logger.info("  language_written=%d", stats.language_written)
    logger.info("  failed=%d", stats.failed)

    if failures:
        fail_log = Path(cfg.output_root) / "failed_language_action.log"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failures) + "\n", encoding="utf-8")
        logger.warning("Failure log written to: %s", fail_log)

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
