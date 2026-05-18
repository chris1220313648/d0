#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Fractal RLDS TFRecord dataset into Motus-style directory layout.

Output layout:
  output_root/
    train/
      <task_slug>/
        videos/<episode_id>.mp4
        qpos/<episode_id>.pt
        epos/<episode_id>.pt
        instructions/<episode_id>.txt
        umt5_wan/<episode_id>.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ConvertConfig:
    source_root: str
    output_root: str
    wan_repo_path: str
    t5_max_length: int = 512
    t5_device: str = "auto"
    generate_epos: bool = True
    epos_dir_name: str = "epos"
    fps: int = 10
    max_episodes_per_split: int = 0
    num_workers: int = 1
    skip_existing: bool = True
    task_slug_max_len: int = 80
    unknown_task_name: str = "unknown_task"
    splits: Tuple[str, ...] = ("train",)
    log_every_n: int = 50
    overwrite: bool = False
    tf_disable_gpu: bool = True
    task_grouping: str = "exact_instruction"
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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    default_cfg = str(Path(__file__).with_name("config_fractal_convert.yml"))
    parser = argparse.ArgumentParser(description="Convert Fractal RLDS to Motus-style dataset.")
    parser.add_argument("--config", type=str, default=default_cfg, help="YAML config path.")
    parser.add_argument("--source_root", type=str, default=None, help="Override source_root in config.")
    parser.add_argument("--output_root", type=str, default=None, help="Override output_root in config.")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to convert, e.g. train.")
    parser.add_argument(
        "--max_episodes_per_split",
        type=int,
        default=None,
        help="Override max episodes per split (0 means all).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Force regenerate existing outputs.")
    parser.add_argument("--log_every_n", type=int, default=None, help="Override log frequency.")
    parser.add_argument("--no_generate_epos", action="store_true", help="Disable epos generation.")
    parser.add_argument("--epos_dir_name", type=str, default=None, help="Override epos dir name.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ConvertConfig:
    cfg_raw = load_yaml(args.config)

    source_root = str(args.source_root or cfg_raw.get("source_root", "")).strip()
    output_root = str(args.output_root or cfg_raw.get("output_root", "")).strip()
    wan_repo_path = str(cfg_raw.get("wan_repo_path", "")).strip()
    splits_raw = args.splits or cfg_raw.get("splits", ["train"])
    splits = tuple(str(x).strip() for x in splits_raw if str(x).strip())

    if not source_root:
        raise ValueError("source_root is required.")
    if not output_root:
        raise ValueError("output_root is required.")
    if not wan_repo_path:
        raise ValueError("wan_repo_path is required.")
    if not splits:
        raise ValueError("At least one split is required.")

    max_eps = cfg_raw.get("max_episodes_per_split", 0)
    if args.max_episodes_per_split is not None:
        max_eps = args.max_episodes_per_split

    log_every_n = cfg_raw.get("log_every_n", 50)
    if args.log_every_n is not None:
        log_every_n = args.log_every_n

    return ConvertConfig(
        source_root=source_root,
        output_root=output_root,
        wan_repo_path=wan_repo_path,
        t5_max_length=int(cfg_raw.get("t5_max_length", 512)),
        t5_device=str(cfg_raw.get("t5_device", "auto")).strip() or "auto",
        generate_epos=not bool(args.no_generate_epos),
        epos_dir_name=str(args.epos_dir_name or cfg_raw.get("epos_dir_name", "epos")).strip() or "epos",
        fps=int(cfg_raw.get("fps", 10)),
        max_episodes_per_split=int(max_eps),
        num_workers=int(cfg_raw.get("num_workers", 1)),
        skip_existing=bool(cfg_raw.get("skip_existing", True)),
        task_slug_max_len=int(cfg_raw.get("task_slug_max_len", 80)),
        unknown_task_name=str(cfg_raw.get("unknown_task_name", "unknown_task")),
        splits=splits,
        log_every_n=max(1, int(log_every_n)),
        overwrite=bool(args.overwrite),
        tf_disable_gpu=bool(cfg_raw.get("tf_disable_gpu", True)),
        task_grouping=str(cfg_raw.get("task_grouping", "exact_instruction")),
        skip_invalid_instruction=bool(cfg_raw.get("skip_invalid_instruction", True)),
        invalid_instruction_patterns=tuple(
            str(x).strip().lower()
            for x in cfg_raw.get("invalid_instruction_patterns", [])
            if str(x).strip()
        ) or ConvertConfig.invalid_instruction_patterns,
    )


def _try_import_tf(disable_tf_gpu: bool = True) -> Tuple[Any, Any]:
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_datasets as tfds  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "tensorflow/tensorflow_datasets is required. "
            "Please install with: pip install tensorflow tensorflow-datasets"
        ) from e

    if disable_tf_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            logger.info("TensorFlow GPU disabled for data reading (tf_disable_gpu=True).")
        except Exception as e:
            logger.warning(f"Failed to disable TensorFlow GPU visibility: {e}")
    else:
        try:
            for dev in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(dev, True)
            logger.info("Enabled TensorFlow GPU memory growth.")
        except Exception as e:
            logger.warning(f"Failed to set TensorFlow memory growth: {e}")
    return tf, tfds


def find_builder_dir(source_root: str) -> str:
    src = Path(source_root)
    if not src.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    if (src / "dataset_info.json").exists():
        return str(src)

    candidates = sorted(
        [p for p in src.iterdir() if p.is_dir() and (p / "dataset_info.json").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        f"Cannot find TFDS builder directory under {source_root}. "
        "Expected dataset_info.json in source root or version subdirectory."
    )


def slugify_task_name(text: str, max_len: int, fallback: str) -> str:
    value = (text or "").strip().lower()
    if not value:
        return fallback
    value = re.sub(r"[\s/\\]+", "_", value)
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return fallback
    return value[:max_len]


_STOPWORDS = {
    "the", "a", "an", "to", "of", "on", "in", "at", "from", "into", "onto", "near",
    "left", "right", "front", "back", "top", "bottom", "middle", "center", "corner",
    "edge", "between", "with", "and", "so", "that", "is", "it", "be", "are", "by",
    "for", "up", "down", "over", "under", "across", "toward", "towards", "around",
    "this", "there", "as", "parallel", "parrellel", "close", "closer",
}
_COLOR_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white",
    "gray", "grey", "brown", "gold", "silver",
}


def normalize_instruction_group_key(text: str, max_tokens: int = 6) -> str:
    s = (text or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    toks = [t for t in s.split() if t]
    filtered = [t for t in toks if t not in _STOPWORDS and t not in _COLOR_WORDS]
    if not filtered:
        filtered = toks
    return " ".join(filtered[:max_tokens])


def is_invalid_instruction(text: str, patterns: Sequence[str]) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return True
    for p in patterns:
        if p and p in s:
            return True
    return False


def decode_bytes_text(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="ignore")
    if isinstance(v, str):
        return v
    return str(v)


def first_non_empty_instruction(instr_values: Any) -> str:
    if instr_values is None:
        return ""
    if isinstance(instr_values, (list, tuple)):
        for x in instr_values:
            s = decode_bytes_text(x).strip()
            if s:
                return s
        return ""
    if isinstance(instr_values, np.ndarray):
        flat = instr_values.reshape(-1)
        for x in flat:
            s = decode_bytes_text(x).strip()
            if s:
                return s
        return ""
    return decode_bytes_text(instr_values).strip()


def ensure_episode_arrays(episode_np: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    steps = episode_np.get("steps", None)
    if steps is None:
        raise KeyError("Episode missing 'steps' field.")

    if isinstance(steps, dict):
        obs = steps.get("observation", None)
        if not isinstance(obs, dict):
            raise KeyError("Episode steps missing 'observation' dict.")

        images = obs.get("image", None)
        qpos = obs.get("base_pose_tool_reached", None)
        instr = obs.get("natural_language_instruction", None)

        if images is None or qpos is None:
            raise KeyError("Episode observation must contain image and base_pose_tool_reached.")

        images_np = np.asarray(images)
        qpos_np = np.asarray(qpos)
        instruction = first_non_empty_instruction(instr)
    else:
        if not hasattr(steps, "__iter__"):
            raise TypeError(f"Unexpected 'steps' type: {type(steps)}")

        image_list: List[np.ndarray] = []
        qpos_list: List[np.ndarray] = []
        instr_list: List[Any] = []

        for step in steps:
            if not isinstance(step, dict):
                raise TypeError(f"Unexpected step item type: {type(step)}")
            obs = step.get("observation", None)
            if not isinstance(obs, dict):
                raise KeyError("Step missing 'observation' dict.")

            img = obs.get("image", None)
            q = obs.get("base_pose_tool_reached", None)
            if img is None or q is None:
                raise KeyError("Step observation must contain image and base_pose_tool_reached.")

            image_list.append(np.asarray(img))
            qpos_list.append(np.asarray(q))
            instr_list.append(obs.get("natural_language_instruction", None))

        if not image_list or not qpos_list:
            raise ValueError("Episode has no valid steps with image/base_pose_tool_reached.")

        images_np = np.stack(image_list, axis=0)
        qpos_np = np.stack(qpos_list, axis=0)
        instruction = first_non_empty_instruction(instr_list)

    if images_np.ndim != 4 or images_np.shape[-1] != 3:
        raise ValueError(f"Invalid image shape: expected [T,H,W,3], got {images_np.shape}")
    if qpos_np.ndim != 2 or qpos_np.shape[1] != 7:
        raise ValueError(f"Invalid qpos shape: expected [T,7], got {qpos_np.shape}")
    if images_np.shape[0] != qpos_np.shape[0]:
        raise ValueError(
            f"Frame/qpos length mismatch: images T={images_np.shape[0]}, qpos T={qpos_np.shape[0]}"
        )
    if images_np.shape[0] < 2:
        raise ValueError(f"Episode too short: T={images_np.shape[0]}")

    return images_np, qpos_np, instruction


def write_video_mp4(images_rgb: np.ndarray, output_path: str, fps: int) -> None:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("opencv-python is required. Please install: pip install opencv-python") from e

    h, w = int(images_rgb.shape[1]), int(images_rgb.shape[2])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    try:
        for frame_rgb in images_rgb:
            if frame_rgb.dtype != np.uint8:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def save_text_embedding_pt(processor: Any, instruction: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tmp:
        tmp.write(instruction.strip() + "\n")
        tmp_path = tmp.name

    try:
        ok = processor.process_meta_file(tmp_path, out_path)
        if not ok:
            raise RuntimeError(f"T5 encoding failed for output: {out_path}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def save_instruction_txt(instruction: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    text = (instruction or "").strip()
    if not text:
        text = "unknown_task"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def compute_epos_from_qpos(qpos_np: np.ndarray) -> np.ndarray:
    """Build per-step pose delta with same length as qpos: epos[0]=0, epos[t]=qpos[t]-qpos[t-1]."""
    if qpos_np.ndim != 2:
        raise ValueError(f"qpos must be [T,D], got {qpos_np.shape}")
    epos = np.zeros_like(qpos_np, dtype=np.float32)
    if qpos_np.shape[0] > 1:
        epos[1:, :] = qpos_np[1:, :].astype(np.float32, copy=False) - qpos_np[:-1, :].astype(np.float32, copy=False)
    return epos


def create_t5_processor(wan_repo_path: str, t5_max_length: int, device: str) -> Any:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from data.robotwin2.robotwin_data_convert.robotwin_converter import T5EmbeddingProcessor
    except Exception:
        try:
            from robotwin2.robotwin_data_convert.robotwin_converter import T5EmbeddingProcessor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Cannot import T5EmbeddingProcessor from robotwin converter. "
                f"repo_root resolved as: {repo_root}. "
                "Please ensure this path exists and dependencies are installed."
            ) from e
    return T5EmbeddingProcessor(
        wan_repo_path=wan_repo_path,
        t5_max_length=t5_max_length,
        device=device,
    )


def convert_split(
    cfg: ConvertConfig,
    tfds: Any,
    ds_split: Iterable[Dict[str, Any]],
    split_name: str,
    t5_processor: Any,
) -> Tuple[int, int, List[str]]:
    output_root = Path(cfg.output_root) / split_name
    output_root.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    skipped_invalid = 0
    failures: List[str] = []

    for episode_idx, episode in enumerate(tfds.as_numpy(ds_split)):
        if cfg.max_episodes_per_split > 0 and episode_idx >= cfg.max_episodes_per_split:
            break

        episode_id = f"{episode_idx:08d}"
        try:
            images_np, qpos_np, instruction = ensure_episode_arrays(episode)
            if cfg.skip_invalid_instruction and is_invalid_instruction(instruction, cfg.invalid_instruction_patterns):
                skipped_invalid += 1
                if (episode_idx + 1) % cfg.log_every_n == 0:
                    logger.info(
                        "[%s] processed=%d success=%d skipped=%d skipped_invalid=%d failed=%d",
                        split_name,
                        episode_idx + 1,
                        success,
                        skipped,
                        skipped_invalid,
                        len(failures),
                    )
                continue

            if cfg.task_grouping == "normalized_instruction":
                group_key = normalize_instruction_group_key(instruction)
            else:
                group_key = instruction
            task_slug = slugify_task_name(
                group_key,
                max_len=cfg.task_slug_max_len,
                fallback=cfg.unknown_task_name,
            )

            task_dir = output_root / task_slug
            video_path = str(task_dir / "videos" / f"{episode_id}.mp4")
            qpos_path = str(task_dir / "qpos" / f"{episode_id}.pt")
            epos_path = str(task_dir / cfg.epos_dir_name / f"{episode_id}.pt")
            instruction_path = str(task_dir / "instructions" / f"{episode_id}.txt")
            t5_path = str(task_dir / "umt5_wan" / f"{episode_id}.pt")

            expected_paths = [video_path, qpos_path, t5_path, instruction_path]
            if cfg.generate_epos:
                expected_paths.append(epos_path)
            exists = all(os.path.exists(p) for p in expected_paths)
            if exists and cfg.skip_existing and not cfg.overwrite:
                skipped += 1
                continue

            write_video_mp4(images_np, video_path, cfg.fps)
            os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
            torch.save(torch.from_numpy(qpos_np).float(), qpos_path)
            if cfg.generate_epos:
                os.makedirs(os.path.dirname(epos_path), exist_ok=True)
                epos_np = compute_epos_from_qpos(qpos_np)
                torch.save(torch.from_numpy(epos_np).float(), epos_path)

            instruction_text = instruction if instruction else cfg.unknown_task_name
            save_instruction_txt(instruction_text, instruction_path)
            save_text_embedding_pt(t5_processor, instruction_text, t5_path)
            success += 1

        except Exception as e:
            err = f"[{split_name}] episode={episode_idx} failed: {e}"
            logger.error(err)
            failures.append(err)

        if (episode_idx + 1) % cfg.log_every_n == 0:
            logger.info(
                "[%s] processed=%d success=%d skipped=%d skipped_invalid=%d failed=%d",
                split_name,
                episode_idx + 1,
                success,
                skipped,
                skipped_invalid,
                len(failures),
            )

    logger.info("[%s] summary: success=%d skipped=%d skipped_invalid=%d failed=%d", split_name, success, skipped, skipped_invalid, len(failures))
    return success, skipped, failures


def validate_config(cfg: ConvertConfig) -> None:
    if cfg.fps <= 0:
        raise ValueError("fps must be > 0")
    if cfg.task_slug_max_len <= 0:
        raise ValueError("task_slug_max_len must be > 0")
    if cfg.max_episodes_per_split < 0:
        raise ValueError("max_episodes_per_split must be >= 0")
    if cfg.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if not cfg.epos_dir_name.strip():
        raise ValueError("epos_dir_name must be non-empty")
    if cfg.task_grouping not in {"normalized_instruction", "exact_instruction"}:
        raise ValueError("task_grouping must be one of: normalized_instruction, exact_instruction")
    if cfg.t5_device not in {"auto", "cpu"} and not cfg.t5_device.startswith("cuda"):
        raise ValueError("t5_device must be one of: auto, cpu, cuda, cuda:<id>")


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config(args)
    validate_config(cfg)

    logger.info("Fractal RLDS conversion config:")
    logger.info("  source_root: %s", cfg.source_root)
    logger.info("  output_root: %s", cfg.output_root)
    logger.info("  splits: %s", list(cfg.splits))
    logger.info("  max_episodes_per_split: %d", cfg.max_episodes_per_split)
    logger.info("  skip_existing: %s", cfg.skip_existing)
    logger.info("  overwrite: %s", cfg.overwrite)
    logger.info("  generate_epos: %s (dir=%s)", cfg.generate_epos, cfg.epos_dir_name)
    logger.info("  num_workers: %d", cfg.num_workers)
    logger.info("  t5_device: %s", cfg.t5_device)
    if cfg.num_workers > 1:
        logger.warning("num_workers is reserved for future parallelization; current implementation runs single-process.")

    tf, tfds = _try_import_tf(disable_tf_gpu=cfg.tf_disable_gpu)
    tf.get_logger().setLevel("ERROR")

    builder_dir = find_builder_dir(cfg.source_root)
    logger.info("Resolved TFDS builder directory: %s", builder_dir)
    builder = tfds.builder_from_directory(builder_dir=builder_dir)

    if cfg.t5_device == "auto":
        t5_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        t5_device = cfg.t5_device

    if t5_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"t5_device is set to {t5_device}, but no CUDA device is available."
        )
    if t5_device == "cpu":
        raise RuntimeError(
            "t5_device resolved to cpu, but current T5EmbeddingProcessor depends on CUDA. "
            "Please run with a CUDA-enabled environment or provide a CUDA device."
        )

    t5_processor = create_t5_processor(
        wan_repo_path=cfg.wan_repo_path,
        t5_max_length=cfg.t5_max_length,
        device=t5_device,
    )

    all_failures: List[str] = []
    total_success = 0
    total_skipped = 0

    for split in cfg.splits:
        logger.info("Starting split: %s", split)
        split_ds = builder.as_dataset(split=split, shuffle_files=False)
        success, skipped, failures = convert_split(cfg, tfds, split_ds, split, t5_processor)
        logger.info(
            "Finished split=%s success=%d skipped=%d failed=%d",
            split,
            success,
            skipped,
            len(failures),
        )
        total_success += success
        total_skipped += skipped
        all_failures.extend(failures)

    logger.info("Conversion complete. success=%d skipped=%d failed=%d", total_success, total_skipped, len(all_failures))
    if all_failures:
        fail_log = Path(cfg.output_root) / "failed_episodes.log"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_log, "w", encoding="utf-8") as f:
            for line in all_failures:
                f.write(line + "\n")
        logger.warning("Failure log written to: %s", fail_log)
        sys.exit(1)


if __name__ == "__main__":
    main()
