#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DROID RLDS TFRecord dataset into Motus-style directory layout.

Output layout:
  output_root/
    train/
      <task_slug>/
        videos/<episode_id>.mp4
        qpos/<episode_id>.pt
        epos/<episode_id>.pt
        instructions/<episode_id>.txt
        umt5_wan/<episode_id>.pt
        language_action/<episode_id>.txt

DROID mapping:
  qpos = steps/action_dict/joint_position                         # [T, 7]
  epos = delta(steps/observation/cartesian_position[:6]) + gripper # [T, 7]

Video frames are T-shape concatenations of three DROID camera views.
"""

from __future__ import annotations

import argparse
import json
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
    camera_keys: Tuple[str, str, str] = (
        "exterior_image_1_left",
        "exterior_image_2_left",
        "wrist_image_left",
    )
    t5_max_length: int = 512
    t5_device: str = "auto"
    generate_t5: bool = True
    generate_language_action: bool = True
    epos_dir_name: str = "epos"
    language_action_dir_name: str = "language_action"
    window_size: int = 16
    fps: int = 10
    max_episodes_per_split: int = 0
    skip_existing: bool = True
    task_slug_max_len: int = 80
    unknown_task_name: str = "unknown_task"
    splits: Tuple[str, ...] = ("train",)
    log_every_n: int = 200
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


@dataclass
class SplitShardInfo:
    split: str
    valid_files: List[Path]
    gstmp_files: List[Path]
    expected_shards: int


@dataclass
class SplitStats:
    seen: int = 0
    written: int = 0
    skipped_existing: int = 0
    skipped_invalid: int = 0
    failed: int = 0


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    default_cfg = str(Path(__file__).with_name("config_droid_convert.yml"))
    parser = argparse.ArgumentParser(description="Convert DROID RLDS to Motus-style dataset.")
    parser.add_argument("--config", type=str, default=default_cfg, help="YAML config path.")
    parser.add_argument("--source_root", type=str, default=None, help="Override source_root in config.")
    parser.add_argument("--output_root", type=str, default=None, help="Override output_root in config.")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to convert, e.g. train.")
    parser.add_argument(
        "--camera_keys",
        nargs=3,
        default=None,
        help="Three observation image keys: top bottom_left bottom_right.",
    )
    parser.add_argument(
        "--max_episodes_per_split",
        type=int,
        default=None,
        help="Override max episodes per split (0 means all available episodes).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Force regenerate existing outputs.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip episodes whose expected outputs already exist.")
    parser.add_argument("--log_every_n", type=int, default=None, help="Override log frequency.")
    parser.add_argument("--window_size", type=int, default=None, help="Language-action sliding window size.")
    parser.add_argument("--no_generate_t5", action="store_true", help="Disable UMT5 embedding generation.")
    parser.add_argument("--no_generate_language_action", action="store_true", help="Disable language_action generation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
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
    if not splits:
        raise ValueError("At least one split is required.")

    camera_keys_raw = args.camera_keys or cfg_raw.get(
        "camera_keys",
        ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"],
    )
    camera_keys = tuple(str(x).strip() for x in camera_keys_raw if str(x).strip())
    if len(camera_keys) != 3:
        raise ValueError("camera_keys must contain exactly three image keys.")

    max_eps = cfg_raw.get("max_episodes_per_split", 0)
    if args.max_episodes_per_split is not None:
        max_eps = args.max_episodes_per_split

    log_every_n = cfg_raw.get("log_every_n", 200)
    if args.log_every_n is not None:
        log_every_n = args.log_every_n

    window_size = cfg_raw.get("window_size", 16)
    if args.window_size is not None:
        window_size = args.window_size

    overwrite = bool(args.overwrite or cfg_raw.get("overwrite", False))
    skip_existing = bool(args.skip_existing or cfg_raw.get("skip_existing", True))

    generate_t5 = bool(cfg_raw.get("generate_t5", True)) and not bool(args.no_generate_t5)
    generate_language_action = bool(cfg_raw.get("generate_language_action", True)) and not bool(args.no_generate_language_action)

    return ConvertConfig(
        source_root=source_root,
        output_root=output_root,
        wan_repo_path=wan_repo_path,
        camera_keys=camera_keys,  # type: ignore[arg-type]
        t5_max_length=int(cfg_raw.get("t5_max_length", 512)),
        t5_device=str(cfg_raw.get("t5_device", "auto")).strip() or "auto",
        generate_t5=generate_t5,
        generate_language_action=generate_language_action,
        epos_dir_name=str(cfg_raw.get("epos_dir_name", "epos")).strip() or "epos",
        language_action_dir_name=str(cfg_raw.get("language_action_dir_name", "language_action")).strip() or "language_action",
        window_size=int(window_size),
        fps=int(cfg_raw.get("fps", 10)),
        max_episodes_per_split=int(max_eps),
        skip_existing=skip_existing,
        task_slug_max_len=int(cfg_raw.get("task_slug_max_len", 80)),
        unknown_task_name=str(cfg_raw.get("unknown_task_name", "unknown_task")),
        splits=splits,
        log_every_n=max(1, int(log_every_n)),
        overwrite=overwrite,
        tf_disable_gpu=bool(cfg_raw.get("tf_disable_gpu", True)),
        task_grouping=str(cfg_raw.get("task_grouping", "exact_instruction")),
        skip_invalid_instruction=bool(cfg_raw.get("skip_invalid_instruction", True)),
        invalid_instruction_patterns=tuple(
            str(x).strip().lower()
            for x in cfg_raw.get("invalid_instruction_patterns", [])
            if str(x).strip()
        )
        or ConvertConfig.invalid_instruction_patterns,
    )


def validate_config(cfg: ConvertConfig) -> None:
    if cfg.fps <= 0:
        raise ValueError("fps must be > 0")
    if cfg.window_size <= 0:
        raise ValueError("window_size must be > 0")
    if cfg.task_slug_max_len <= 0:
        raise ValueError("task_slug_max_len must be > 0")
    if cfg.max_episodes_per_split < 0:
        raise ValueError("max_episodes_per_split must be >= 0")
    if cfg.task_grouping not in {"normalized_instruction", "exact_instruction"}:
        raise ValueError("task_grouping must be one of: normalized_instruction, exact_instruction")
    if cfg.generate_t5 and not cfg.wan_repo_path:
        raise ValueError("wan_repo_path is required when generate_t5=True.")
    if cfg.t5_device not in {"auto", "cpu"} and not cfg.t5_device.startswith("cuda"):
        raise ValueError("t5_device must be one of: auto, cpu, cuda, cuda:<id>")


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
            logger.warning("Failed to disable TensorFlow GPU visibility: %s", e)
    return tf, tfds


def find_builder_dir(source_root: str) -> Path:
    src = Path(source_root)
    if not src.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    if (src / "dataset_info.json").exists():
        return src
    candidates = sorted(
        [p for p in src.iterdir() if p.is_dir() and (p / "dataset_info.json").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot find TFDS builder directory under {source_root}. "
        "Expected dataset_info.json in source root or version subdirectory."
    )


def _expected_shards_from_dataset_info(builder_dir: Path, split: str) -> int:
    info_path = builder_dir / "dataset_info.json"
    if not info_path.exists():
        return 0
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for item in info.get("splits", []):
            if item.get("name") == split:
                return len(item.get("shardLengths", []) or [])
    except Exception as err:  # noqa: BLE001
        logger.warning("Failed to parse dataset_info.json for shard count: %s", err)
    return 0


def _shard_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"tfrecord-(\d+)-of-\d+", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def scan_split_shards(builder_dir: Path, dataset_name: str, split: str) -> SplitShardInfo:
    pattern = f"{dataset_name}-{split}.tfrecord-*"
    all_files = sorted(builder_dir.glob(pattern), key=_shard_sort_key)
    gstmp_files = [p for p in all_files if ".gstmp" in p.name]
    valid_files = [p for p in all_files if ".gstmp" not in p.name and p.is_file()]
    expected_shards = _expected_shards_from_dataset_info(builder_dir, split)

    logger.info(
        "[%s] shard preflight: valid=%d gstmp=%d expected=%d",
        split,
        len(valid_files),
        len(gstmp_files),
        expected_shards,
    )
    if gstmp_files:
        logger.warning("[%s] Skipping %d .gstmp temporary shard files.", split, len(gstmp_files))
    if expected_shards and len(valid_files) < expected_shards:
        logger.warning(
            "[%s] Valid shard count (%d) is lower than dataset_info expected shard count (%d). "
            "Continuing with available completed shards only.",
            split,
            len(valid_files),
            expected_shards,
        )
    if not valid_files:
        raise FileNotFoundError(f"No valid TFRecord files found for split={split} in {builder_dir}")

    return SplitShardInfo(
        split=split,
        valid_files=valid_files,
        gstmp_files=gstmp_files,
        expected_shards=expected_shards,
    )


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


def first_instruction_from_steps(steps: Dict[str, Any]) -> str:
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        instruction = first_non_empty_instruction(steps.get(key, None))
        if instruction:
            return instruction
    return ""


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


def resolve_task_slug(cfg: ConvertConfig, instruction: str) -> str:
    if cfg.task_grouping == "normalized_instruction":
        group_key = normalize_instruction_group_key(instruction)
    else:
        group_key = instruction
    return slugify_task_name(
        group_key,
        max_len=cfg.task_slug_max_len,
        fallback=cfg.unknown_task_name,
    )


def _angle_diff_rad(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return (curr - prev + np.pi) % (2.0 * np.pi) - np.pi


def compute_epos_from_cartesian(cartesian_np: np.ndarray, gripper_np: np.ndarray) -> np.ndarray:
    if cartesian_np.ndim != 2 or cartesian_np.shape[1] != 6:
        raise ValueError(f"cartesian_position must be [T,6], got {cartesian_np.shape}")
    gripper = np.asarray(gripper_np)
    t = int(cartesian_np.shape[0])
    if gripper.ndim == 1:
        gripper = gripper.reshape(-1, 1)
    elif gripper.ndim > 2:
        gripper = gripper.reshape(t, -1)
    if gripper.ndim != 2 or gripper.shape[1] != 1:
        raise ValueError(f"gripper_position must be [T,1], got {gripper.shape}")
    if gripper.shape[0] != t:
        raise ValueError(f"cartesian/gripper length mismatch: cartesian={t} gripper={gripper.shape[0]}")

    epos = np.zeros((t, 7), dtype=np.float32)
    if t > 1:
        epos[1:, :3] = cartesian_np[1:, :3].astype(np.float32) - cartesian_np[:-1, :3].astype(np.float32)
        epos[1:, 3:6] = _angle_diff_rad(cartesian_np[1:, 3:6], cartesian_np[:-1, 3:6]).astype(np.float32)
    epos[:, 6:7] = gripper.astype(np.float32, copy=False)
    return epos


def ensure_episode_arrays(
    episode_np: Dict[str, Any],
    camera_keys: Sequence[str],
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray, str]:
    steps = episode_np.get("steps", None)
    if steps is None:
        raise KeyError("Episode missing 'steps' field.")

    if isinstance(steps, dict):
        obs = steps.get("observation", None)
        action_dict = steps.get("action_dict", None)
        if not isinstance(obs, dict) or not isinstance(action_dict, dict):
            raise KeyError("DROID steps must contain observation and action_dict dicts.")

        missing_cameras = [key for key in camera_keys if key not in obs]
        if missing_cameras:
            raise KeyError(f"Missing camera keys in observation: {missing_cameras}")

        images_np = [np.asarray(obs[key]) for key in camera_keys]
        qpos_np = np.asarray(action_dict.get("joint_position", None))
        cartesian_np = np.asarray(obs.get("cartesian_position", None))
        gripper_np = np.asarray(action_dict.get("gripper_position", None))
        instruction = first_instruction_from_steps(steps)
    else:
        if not hasattr(steps, "__iter__"):
            raise TypeError(f"Unexpected episode['steps'] type: {type(steps)}")

        image_lists: List[List[np.ndarray]] = [[] for _ in camera_keys]
        qpos_list: List[np.ndarray] = []
        cartesian_list: List[np.ndarray] = []
        gripper_list: List[np.ndarray] = []
        instruction_values: List[Any] = []

        for step in steps:
            if not isinstance(step, dict):
                raise TypeError(f"Unexpected step type: {type(step)}")
            obs = step.get("observation", None)
            action_dict = step.get("action_dict", None)
            if not isinstance(obs, dict) or not isinstance(action_dict, dict):
                raise KeyError("DROID step must contain observation and action_dict dicts.")

            for idx, key in enumerate(camera_keys):
                if key not in obs:
                    raise KeyError(f"Missing camera key in observation: {key}")
                image_lists[idx].append(np.asarray(obs[key]))
            qpos_list.append(np.asarray(action_dict.get("joint_position", None)))
            cartesian_list.append(np.asarray(obs.get("cartesian_position", None)))
            gripper_list.append(np.asarray(action_dict.get("gripper_position", None)))
            instruction_values.extend(
                [
                    step.get("language_instruction", None),
                    step.get("language_instruction_2", None),
                    step.get("language_instruction_3", None),
                ]
            )

        if not qpos_list:
            raise ValueError("Episode has no valid steps.")

        images_np = [np.stack(items, axis=0) for items in image_lists]
        qpos_np = np.stack(qpos_list, axis=0)
        cartesian_np = np.stack(cartesian_list, axis=0)
        gripper_np = np.stack(gripper_list, axis=0)
        instruction = first_non_empty_instruction(instruction_values)

    for key, arr in zip(camera_keys, images_np):
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Invalid image shape for {key}: expected [T,H,W,3], got {arr.shape}")
    if qpos_np.ndim != 2 or qpos_np.shape[1] != 7:
        raise ValueError(f"Invalid qpos/action_dict.joint_position shape: expected [T,7], got {qpos_np.shape}")
    if cartesian_np.ndim != 2 or cartesian_np.shape[1] != 6:
        raise ValueError(f"Invalid observation.cartesian_position shape: expected [T,6], got {cartesian_np.shape}")

    t = int(qpos_np.shape[0])
    lengths = [int(arr.shape[0]) for arr in images_np] + [int(cartesian_np.shape[0])]
    if any(length != t for length in lengths):
        raise ValueError(f"Episode length mismatch: qpos={t} other_lengths={lengths}")
    if t < 2:
        raise ValueError(f"Episode too short: T={t}")

    epos_np = compute_epos_from_cartesian(cartesian_np, gripper_np)
    camera_sequences = (images_np[0], images_np[1], images_np[2])
    return camera_sequences, qpos_np.astype(np.float32, copy=False), epos_np, instruction


def concatenate_camera_sequences(top: np.ndarray, bottom_left: np.ndarray, bottom_right: np.ndarray) -> np.ndarray:
    if top.shape[0] != bottom_left.shape[0] or top.shape[0] != bottom_right.shape[0]:
        raise ValueError(
            "Camera sequence length mismatch: "
            f"top={top.shape[0]} bottom_left={bottom_left.shape[0]} bottom_right={bottom_right.shape[0]}"
        )
    return np.stack(
        [concatenate_camera_frame(top[i], bottom_left[i], bottom_right[i]) for i in range(top.shape[0])],
        axis=0,
    )


def concatenate_camera_frame(top: np.ndarray, bottom_left: np.ndarray, bottom_right: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("opencv-python is required. Please install: pip install opencv-python") from e

    top_img = _ensure_uint8_rgb(top)
    left_img = _ensure_uint8_rgb(bottom_left)
    right_img = _ensure_uint8_rgb(bottom_right)

    h, w = top_img.shape[:2]
    half_h, half_w = h // 2, w // 2
    left_resized = cv2.resize(left_img, (half_w, half_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_img, (half_w, half_h), interpolation=cv2.INTER_AREA)
    bottom = np.hstack([left_resized, right_resized])
    return np.vstack([top_img, bottom])


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Image must be [H,W,3], got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


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
            frame_bgr = cv2.cvtColor(_ensure_uint8_rgb(frame_rgb), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def write_tshape_video_mp4(
    top_seq: np.ndarray,
    bottom_left_seq: np.ndarray,
    bottom_right_seq: np.ndarray,
    output_path: str,
    fps: int,
) -> None:
    """Write a T-shape three-camera video without materializing all combined frames."""
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("opencv-python is required. Please install: pip install opencv-python") from e

    if top_seq.shape[0] != bottom_left_seq.shape[0] or top_seq.shape[0] != bottom_right_seq.shape[0]:
        raise ValueError(
            "Camera sequence length mismatch: "
            f"top={top_seq.shape[0]} bottom_left={bottom_left_seq.shape[0]} bottom_right={bottom_right_seq.shape[0]}"
        )
    if int(top_seq.shape[0]) <= 0:
        raise ValueError("Cannot write empty video.")

    first_frame = concatenate_camera_frame(top_seq[0], bottom_left_seq[0], bottom_right_seq[0])
    h, w = int(first_frame.shape[0]), int(first_frame.shape[1])
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
        writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        for i in range(1, int(top_seq.shape[0])):
            frame_rgb = concatenate_camera_frame(top_seq[i], bottom_left_seq[i], bottom_right_seq[i])
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
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


def save_instruction_txt(instruction: str, out_path: str, fallback: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    text = (instruction or "").strip() or fallback
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


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
                f"repo_root resolved as: {repo_root}."
            ) from e
    return T5EmbeddingProcessor(
        wan_repo_path=wan_repo_path,
        t5_max_length=t5_max_length,
        device=device,
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

    parts: List[str] = []
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
    return ", ".join(parts) if parts else "hold position"


def build_language_action_lines(epos: np.ndarray, window_size: int) -> List[str]:
    if epos.ndim != 2 or epos.shape[1] != 7:
        raise ValueError(f"epos must be [T,7], got {epos.shape}")
    lines: List[str] = []
    for i in range(int(epos.shape[0])):
        j = min(i + window_size, int(epos.shape[0]))
        lines.append(_summarize_delta_action_window(epos[i:j, :]))
    return lines


def save_language_action(epos: np.ndarray, out_path: str, window_size: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = build_language_action_lines(epos, window_size=window_size)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_available_shard_dataset(tf: Any, builder: Any, valid_files: Sequence[Path]) -> Any:
    dataset = tf.data.TFRecordDataset(
        [str(p) for p in valid_files],
        buffer_size=8 * 1024 * 1024,
        num_parallel_reads=1,
    )
    return dataset.map(builder.info.features.deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)


def convert_split(
    cfg: ConvertConfig,
    tf: Any,
    tfds: Any,
    builder: Any,
    shard_info: SplitShardInfo,
    t5_processor: Any,
) -> Tuple[SplitStats, List[str]]:
    output_root = Path(cfg.output_root) / shard_info.split
    output_root.mkdir(parents=True, exist_ok=True)

    stats = SplitStats()
    failures: List[str] = []
    ds_split = build_available_shard_dataset(tf, builder, shard_info.valid_files)

    for episode_idx, episode in enumerate(tfds.as_numpy(ds_split)):
        if cfg.max_episodes_per_split > 0 and stats.seen >= cfg.max_episodes_per_split:
            break

        stats.seen += 1
        episode_id = f"{episode_idx:08d}"
        try:
            camera_sequences, qpos_np, epos_np, instruction = ensure_episode_arrays(episode, cfg.camera_keys)
            if cfg.skip_invalid_instruction and is_invalid_instruction(instruction, cfg.invalid_instruction_patterns):
                stats.skipped_invalid += 1
                continue

            task_slug = resolve_task_slug(cfg, instruction)
            task_dir = output_root / task_slug
            video_path = str(task_dir / "videos" / f"{episode_id}.mp4")
            qpos_path = str(task_dir / "qpos" / f"{episode_id}.pt")
            epos_path = str(task_dir / cfg.epos_dir_name / f"{episode_id}.pt")
            instruction_path = str(task_dir / "instructions" / f"{episode_id}.txt")
            t5_path = str(task_dir / "umt5_wan" / f"{episode_id}.pt")
            lang_action_path = str(task_dir / cfg.language_action_dir_name / f"{episode_id}.txt")

            expected_paths = [video_path, qpos_path, epos_path, instruction_path]
            if cfg.generate_t5:
                expected_paths.append(t5_path)
            if cfg.generate_language_action:
                expected_paths.append(lang_action_path)
            if cfg.skip_existing and not cfg.overwrite and all(os.path.exists(p) for p in expected_paths):
                stats.skipped_existing += 1
                continue

            write_tshape_video_mp4(*camera_sequences, video_path, cfg.fps)
            os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
            torch.save(torch.from_numpy(qpos_np).float(), qpos_path)
            os.makedirs(os.path.dirname(epos_path), exist_ok=True)
            torch.save(torch.from_numpy(epos_np).float(), epos_path)

            instruction_text = instruction if instruction else cfg.unknown_task_name
            save_instruction_txt(instruction_text, instruction_path, cfg.unknown_task_name)
            if cfg.generate_t5:
                save_text_embedding_pt(t5_processor, instruction_text, t5_path)
            if cfg.generate_language_action:
                save_language_action(epos_np, lang_action_path, cfg.window_size)

            stats.written += 1

        except Exception as e:  # noqa: BLE001
            stats.failed += 1
            err = f"[{shard_info.split}] episode={episode_id} failed: {e}"
            logger.error(err)
            failures.append(err)

        if stats.seen % cfg.log_every_n == 0:
            logger.info(
                "[%s] seen=%d written=%d skipped_invalid=%d skipped_existing=%d failed=%d skipped_gstmp=%d",
                shard_info.split,
                stats.seen,
                stats.written,
                stats.skipped_invalid,
                stats.skipped_existing,
                stats.failed,
                len(shard_info.gstmp_files),
            )

    logger.info(
        "[%s] summary: seen=%d written=%d skipped_invalid=%d skipped_existing=%d failed=%d skipped_gstmp=%d",
        shard_info.split,
        stats.seen,
        stats.written,
        stats.skipped_invalid,
        stats.skipped_existing,
        stats.failed,
        len(shard_info.gstmp_files),
    )
    return stats, failures


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    cfg = build_config(args)
    validate_config(cfg)

    logger.info("DROID RLDS conversion config:")
    logger.info("  source_root: %s", cfg.source_root)
    logger.info("  output_root: %s", cfg.output_root)
    logger.info("  splits: %s", list(cfg.splits))
    logger.info("  camera_keys: %s", list(cfg.camera_keys))
    logger.info("  max_episodes_per_split: %d", cfg.max_episodes_per_split)
    logger.info("  skip_existing: %s", cfg.skip_existing)
    logger.info("  overwrite: %s", cfg.overwrite)
    logger.info("  generate_t5: %s", cfg.generate_t5)
    logger.info("  generate_language_action: %s", cfg.generate_language_action)
    logger.info("  window_size: %d", cfg.window_size)
    logger.info("  t5_device: %s", cfg.t5_device)

    tf, tfds = _try_import_tf(disable_tf_gpu=cfg.tf_disable_gpu)
    tf.get_logger().setLevel("ERROR")

    builder_dir = find_builder_dir(cfg.source_root)
    logger.info("Resolved TFDS builder directory: %s", builder_dir)
    builder = tfds.builder_from_directory(builder_dir=str(builder_dir))
    dataset_name = str(builder.info.name)

    t5_processor = None
    if cfg.generate_t5:
        if cfg.t5_device == "auto":
            t5_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            t5_device = cfg.t5_device

        if t5_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"t5_device is set to {t5_device}, but no CUDA device is available.")
        if t5_device == "cpu":
            raise RuntimeError(
                "t5_device resolved to cpu, but current T5EmbeddingProcessor depends on CUDA. "
                "Use --no_generate_t5 for smoke tests or run with a CUDA-enabled environment."
            )
        t5_processor = create_t5_processor(cfg.wan_repo_path, cfg.t5_max_length, t5_device)

    all_failures: List[str] = []
    total_written = 0
    total_seen = 0
    for split in cfg.splits:
        logger.info("Starting split: %s", split)
        shard_info = scan_split_shards(builder_dir, dataset_name=dataset_name, split=split)
        stats, failures = convert_split(cfg, tf, tfds, builder, shard_info, t5_processor)
        total_seen += stats.seen
        total_written += stats.written
        all_failures.extend(failures)

    logger.info("Conversion complete. seen=%d written=%d failed=%d", total_seen, total_written, len(all_failures))
    if all_failures:
        fail_log = Path(cfg.output_root) / "failed_episodes.log"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(all_failures) + "\n", encoding="utf-8")
        logger.warning("Failure log written to: %s", fail_log)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
