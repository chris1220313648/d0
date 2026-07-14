"""
RoboCOIN LeRobot Dataset Loader for Motus
-----------------------------------------
RoboCOIN is LeRobot v2.1-compatible, but each robot/task can expose different
state/action dimensions and camera names. This wrapper pads actions/states to a
fixed training dimension and resolves visual streams per task.

Expected RoboCOIN data structure:
  <root>/  # e.g. /root/nasbak/cjy/robot_raw/robocoin/RoboCOIN
    <robot_task>/
      meta/
        info.json
        episodes.jsonl
        episodes_stats.jsonl
        tasks.jsonl
      annotations/
        subtask_annotations.jsonl
        eef_direction_annotation.jsonl
        eef_velocity_annotation.jsonl
        eef_acc_mag_annotation.jsonl
        gripper_mode_annotation.jsonl
        gripper_activity_annotation.jsonl
      data/
        chunk-000/
          episode_000000.parquet
          ...
      videos/
        chunk-000/
          <camera_key>/
            episode_000000.mp4
            ...
      language_action/
        episode_000000.txt
        ...
      t5_embedding/
        episode_000000.pt
        ...

meta/episodes.jsonl should contain language_action_path and t5_embedding_path
when LAP supervision and offline T5 embeddings are enabled. The downloaded
RoboCOIN corpus has heterogeneous action/state dimensions; this loader pads to
target_action_dim/target_state_dim and emits action_mask for valid coordinates.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from lerobot.datasets.video_utils import decode_video_frames

from .lerobot_dataset import LeRobotMotusDataset, preprocess_vlm_messages, tensor_to_pil
from utils.vlm_utils import preprocess_vlm_messages_lap


class LeRobotRoboCOINDataset(LeRobotMotusDataset):
    """RoboCOIN-specific LeRobot wrapper with padded heterogeneous actions."""

    DEFAULT_TARGET_ACTION_DIM = 54
    DEFAULT_TARGET_STATE_DIM = 118
    DEFAULT_VISUAL_PRIORITY: Tuple[Tuple[str, ...], ...] = (
        (
            "observation.images.cam_high_rgb",
            "observation.images.cam_left_wrist_rgb",
            "observation.images.cam_right_wrist_rgb",
        ),
        (
            "observation.images.cam_head_rgb",
            "observation.images.cam_left_wrist_rgb",
            "observation.images.cam_right_wrist_rgb",
        ),
        (
            "observation.images.camera_head_rgb",
            "observation.images.camera_left_wrist_rgb",
            "observation.images.camera_right_wrist_rgb",
        ),
        (
            "observation.images.cam_front_head_rgb",
            "observation.images.cam_left_wrist_rgb",
            "observation.images.cam_right_wrist_rgb",
        ),
        (
            "observation.images.cam_front_chest_rgb",
            "observation.images.cam_left_wrist_rgb",
            "observation.images.cam_right_wrist_rgb",
        ),
        (
            "observation.images.cam_head_left_rgb",
            "observation.images.cam_left_wrist_rgb",
            "observation.images.cam_right_wrist_rgb",
        ),
    )

    def __init__(
        self,
        *args,
        visual_keys: Optional[Sequence[str]] = None,
        language_action_dir_name: str = "language_action",
        normalize_actions: bool = False,
        use_language_action: bool = False,
        target_action_dim: int = DEFAULT_TARGET_ACTION_DIM,
        target_state_dim: int = DEFAULT_TARGET_STATE_DIM,
        **kwargs,
    ):
        kwargs["use_multi_lerobot_dataset"] = False
        if normalize_actions:
            raise ValueError(
                "LeRobotRoboCOINDataset pads heterogeneous raw RoboCOIN actions; "
                "normalization requires per-robot stats and is intentionally disabled."
            )
        super().__init__(*args, **kwargs)
        self.use_language_action = bool(use_language_action)
        self.language_action_dir_name = str(language_action_dir_name).strip() or "language_action"
        self.target_action_dim = int(target_action_dim)
        self.target_state_dim = int(target_state_dim)
        self.visual_keys_override = tuple(visual_keys) if visual_keys is not None else None
        self._language_action_cache: Dict[Tuple[int, int], List[str]] = {}
        self._visual_keys_cache: Dict[str, Tuple[str, ...]] = {}

    def __getitem__(self, idx):
        if not self.lerobot_dataset:
            return None

        episode_idx = int(random.randint(0, self.lerobot_dataset.num_episodes - 1))
        task_idx = 0
        if self.task_mode == "multi":
            task_idx = self.episode_id_to_task_idx[episode_idx]
            if task_idx > 0:
                episode_idx = episode_idx - self.episode_num_accumulated[task_idx - 1]
            from_idx_t = self.lerobot_dataset._datasets[task_idx].episode_data_index["from"][episode_idx]
            to_idx_t = self.lerobot_dataset._datasets[task_idx].episode_data_index["to"][episode_idx]
        else:
            from_idx_t = self.lerobot_dataset.episode_data_index["from"][episode_idx]
            to_idx_t = self.lerobot_dataset.episode_data_index["to"][episode_idx]

        from_idx = int(from_idx_t.item()) if hasattr(from_idx_t, "item") else int(from_idx_t)
        to_idx = int(to_idx_t.item()) if hasattr(to_idx_t, "item") else int(to_idx_t)
        total_frames = int(to_idx - from_idx)
        condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)

        if self.task_mode == "multi" and task_idx > 0:
            from_idx += self.frame_num_accumulated[task_idx - 1]

        global_cond_idx = int(from_idx + condition_frame_idx)
        global_video_indices = [int(from_idx + i) for i in video_indices]
        global_action_indices = [int(from_idx + i) for i in action_indices]

        if self.task_mode == "multi":
            base_offset = int(self.frame_num_accumulated[task_idx - 1]) if task_idx > 0 else 0
            ds_media = self.lerobot_dataset._datasets[task_idx]
            local_cond_idx = int(global_cond_idx - base_offset)
            local_video_indices = [int(g - base_offset) for g in global_video_indices]
            local_action_indices = [int(g - base_offset) for g in global_action_indices]
            hf_dataset = ds_media.hf_dataset
        else:
            ds_media = self.lerobot_dataset
            local_cond_idx = int(global_cond_idx)
            local_video_indices = list(global_video_indices)
            local_action_indices = list(global_action_indices)
            hf_dataset = ds_media.hf_dataset

        item_cond = hf_dataset[local_cond_idx]
        timestamps = self._read_timestamps(hf_dataset, [local_cond_idx] + local_video_indices)
        ep_idx_raw = item_cond.get("episode_index", None)
        if ep_idx_raw is None:
            raise KeyError("episode_index not found in hf_dataset row; cannot resolve video file path")
        ep_for_video = int(ep_idx_raw.item()) if hasattr(ep_idx_raw, "item") else int(ep_idx_raw)

        first_frame, video_frames_sampled = self._load_visual_frames(ds_media, ep_for_video, timestamps)

        if "observation.state" in item_cond:
            initial_state_raw = torch.as_tensor(item_cond["observation.state"]).float()
        elif "action" in item_cond:
            initial_state_raw = torch.as_tensor(item_cond["action"]).float()
        elif "actions" in item_cond:
            initial_state_raw = torch.as_tensor(item_cond["actions"]).float()
        else:
            raise KeyError("No state found in item (expected observation.state/actions/action)")
        initial_state = self._pad_vector(initial_state_raw, self.target_state_dim, "initial_state")

        action_key = "action" if "action" in hf_dataset.column_names else None
        if action_key is None and "actions" in hf_dataset.column_names:
            action_key = "actions"
        if action_key is None:
            raise KeyError("No action column found in hf_dataset (expected 'action' or 'actions')")

        raw_action_sequence = self._to_action_tensor(hf_dataset[local_action_indices][action_key])
        if raw_action_sequence.ndim == 1:
            raw_action_sequence = raw_action_sequence.unsqueeze(0)
        action_sequence = self._pad_vector(raw_action_sequence, self.target_action_dim, "action_sequence")
        action_mask = torch.zeros_like(action_sequence, dtype=torch.bool)
        action_mask[..., : raw_action_sequence.shape[-1]] = True

        language_embedding = self._load_language_embedding(item_cond, task_idx)
        language_action = None
        if self.use_language_action:
            language_action = self._load_language_action(
                item_cond=item_cond,
                episode_index=ep_for_video,
                task_idx=task_idx,
                condition_frame_idx=condition_frame_idx,
            )

        vlm_tokens = None
        if self.vlm_processor:
            text_instr = item_cond.get("language_instruction", None)
            if text_instr is None or (isinstance(text_instr, str) and len(text_instr.strip()) == 0):
                text_instr = item_cond.get("task", "")
            first_frame_pil = tensor_to_pil(first_frame)
            if self.use_language_action:
                vlm_tokens = preprocess_vlm_messages_lap(
                    text_instr,
                    first_frame_pil,
                    self.vlm_processor,
                    language_action,
                    supervise_answer=True,
                )
            else:
                vlm_tokens = preprocess_vlm_messages(text_instr, first_frame_pil, self.vlm_processor)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames_sampled,
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "action_mask": action_mask,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_tokens,
        }

    def _read_timestamps(self, hf_dataset, indices: List[int]) -> List[float]:
        ts_vals = hf_dataset[indices]["timestamp"]
        if isinstance(ts_vals, torch.Tensor):
            return ts_vals.flatten().tolist()
        if isinstance(ts_vals, (list, tuple)) and len(ts_vals) > 0 and isinstance(ts_vals[0], torch.Tensor):
            return torch.stack(ts_vals).flatten().tolist()
        return [float(x) for x in list(ts_vals)]

    def _visual_feature_keys(self, ds_media) -> Tuple[str, ...]:
        cache_key = str(ds_media.root)
        cached = self._visual_keys_cache.get(cache_key)
        if cached is not None:
            return cached

        features = ds_media.features
        if self.visual_keys_override is not None:
            missing = [key for key in self.visual_keys_override if key not in features]
            if missing:
                raise ValueError(f"RoboCOIN visual keys not found in {ds_media.root}: {missing}")
            keys = self.visual_keys_override
        else:
            keys = ()
            for candidate in self.DEFAULT_VISUAL_PRIORITY:
                if all(key in features for key in candidate):
                    keys = candidate
                    break
            if not keys:
                visual_keys = [
                    key for key, feature in features.items()
                    if self._is_visual_feature(feature)
                ]
                if not visual_keys:
                    raise ValueError(f"No RoboCOIN visual features found in {ds_media.root}")
                keys = (self._choose_single_visual_key(visual_keys),)

        self._visual_keys_cache[cache_key] = tuple(keys)
        return tuple(keys)

    @staticmethod
    def _is_visual_feature(feature: Any) -> bool:
        if isinstance(feature, Mapping):
            return feature.get("dtype") in {"video", "image"}
        return feature.__class__.__name__ in {"Image", "VideoFrame"}

    @staticmethod
    def _choose_single_visual_key(keys: Sequence[str]) -> str:
        priorities = ("head", "high", "front", "ego", "cam")
        for token in priorities:
            matches = sorted(key for key in keys if token in key)
            if matches:
                return matches[0]
        return sorted(keys)[0]

    def _decode_key(self, ds_media, episode_index: int, video_key: str, timestamps: List[float]) -> torch.Tensor:
        video_path = Path(ds_media.root) / ds_media.meta.get_video_file_path(episode_index, video_key)
        return decode_video_frames(video_path, timestamps, ds_media.tolerance_s, ds_media.video_backend).squeeze(0)

    def _load_visual_frames(self, ds_media, episode_index: int, timestamps: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self._visual_feature_keys(ds_media)
        decoded = [self._decode_key(ds_media, episode_index, key, timestamps).float() for key in keys]
        frames = []
        for frame_idx in range(decoded[0].shape[0]):
            frame_views = [tensor[frame_idx] for tensor in decoded]
            frames.append(self._stitch_views(frame_views))
        return frames[0], torch.stack(frames[1:], dim=0)

    def _stitch_views(self, views: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(views) == 1:
            return self._resize_frame_chw(views[0], self.video_size)
        if len(views) == 2:
            h = int(max(views[0].shape[1], views[1].shape[1]))
            w0 = int(views[0].shape[2])
            w1 = int(views[1].shape[2])
            left = self._resize_frame_chw(views[0], (h, w0))
            right = self._resize_frame_chw(views[1], (h, w1))
            canvas = torch.zeros((views[0].shape[0], h, w0 + w1), dtype=views[0].dtype)
            canvas[:, :, :w0] = left
            canvas[:, :, w0:] = right
            return self._resize_frame_chw(canvas, self.video_size)

        top = views[0]
        bottom = views[1:3]
        c = int(top.shape[0])
        canvas_w = int(max(top.shape[2], bottom[0].shape[2] + bottom[1].shape[2]))
        top_h = int(top.shape[1])
        bottom_h = int(max(bottom[0].shape[1], bottom[1].shape[1]))
        left_w = canvas_w // 2
        right_w = canvas_w - left_w
        top_r = self._resize_frame_chw(top, (top_h, canvas_w))
        left_r = self._resize_frame_chw(bottom[0], (bottom_h, left_w))
        right_r = self._resize_frame_chw(bottom[1], (bottom_h, right_w))
        canvas = torch.zeros((c, top_h + bottom_h, canvas_w), dtype=top.dtype)
        canvas[:, :top_h, :] = top_r
        canvas[:, top_h:, :left_w] = left_r
        canvas[:, top_h:, left_w:] = right_r
        return self._resize_frame_chw(canvas, self.video_size)

    def _to_action_tensor(self, values: Any) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.float()
        if isinstance(values, (list, tuple)) and len(values) > 0 and isinstance(values[0], torch.Tensor):
            return torch.stack([value.float() for value in values], dim=0)
        if isinstance(values, (list, tuple)) and len(values) > 0 and isinstance(values[0], np.ndarray):
            return torch.from_numpy(np.stack(values, axis=0)).float()
        return torch.tensor(values, dtype=torch.float32)

    def _pad_vector(self, tensor: torch.Tensor, target_dim: int, name: str) -> torch.Tensor:
        current_dim = int(tensor.shape[-1])
        if current_dim == target_dim:
            return tensor.float()
        if current_dim > target_dim:
            raise ValueError(f"{name} dim {current_dim} exceeds RoboCOIN target dim {target_dim}")
        padded_shape = list(tensor.shape)
        padded_shape[-1] = target_dim
        padded = tensor.new_zeros(padded_shape)
        padded[..., :current_dim] = tensor
        return padded.float()

    def _load_language_embedding(self, item_cond: Dict[str, Any], task_idx: int) -> torch.Tensor:
        all_embeddings = item_cond.get("language_embedding", None)
        if all_embeddings is None:
            all_embeddings = item_cond.get("observation.feature.language_embedding", None)

        if all_embeddings is None:
            ep_index_raw = item_cond.get("episode_index", None)
            if ep_index_raw is None:
                raise KeyError("episode_index not found in item; cannot load external embedding")
            ep_index = int(ep_index_raw.item()) if hasattr(ep_index_raw, "item") else int(ep_index_raw)

            cached = self._episode_embedding_cache.get(ep_index, None)
            if cached is None:
                if self.task_mode == "single":
                    ep_meta = self.lerobot_dataset.meta.episodes.get(ep_index, None)
                    dataset_root = Path(self.lerobot_dataset.root)
                else:
                    ep_meta = self.lerobot_dataset._datasets[task_idx].meta.episodes.get(ep_index, None)
                    dataset_root = Path(self.lerobot_dataset._datasets[task_idx].root)
                if ep_meta is None:
                    raise KeyError(f"episode {ep_index} not found in meta.episodes")

                rel_path = ep_meta.get("t5_embedding_path", None)
                if rel_path is None:
                    if not self.enable_t5_fallback:
                        raise KeyError(
                            "language_embedding not found in item and t5_embedding_path not found in meta/episodes.jsonl; "
                            "set enable_t5_fallback=True or pre-generate T5 embeddings."
                        )
                    instr = item_cond.get("language_instruction", None)
                    if instr is None or (isinstance(instr, str) and len(instr.strip()) == 0):
                        instr = item_cond.get("task", "")
                    emb = self._encode_and_cache_t5_embedding(ep_index, str(instr))
                    cached = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                    self._episode_embedding_cache[ep_index] = cached
                else:
                    abs_path = Path(str(rel_path))
                    if not abs_path.is_absolute():
                        abs_path = dataset_root / abs_path
                    emb = torch.load(abs_path, map_location="cpu", weights_only=True)
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    if emb.ndim == 2:
                        emb = emb.unsqueeze(0)
                    self._episode_embedding_cache[ep_index] = emb
                    cached = emb

            all_embeddings = cached

        if not isinstance(all_embeddings, torch.Tensor):
            all_embeddings = torch.tensor(all_embeddings)
        if all_embeddings.ndim == 2:
            all_embeddings = all_embeddings.unsqueeze(0)
        return all_embeddings[0].float()

    def _select_language_action_line(self, lines: List[str], condition_frame_idx: int, source: str) -> str:
        if not lines:
            raise ValueError(f"No language action lines in {source}")
        if condition_frame_idx >= len(lines):
            return lines[-1]
        return lines[condition_frame_idx]

    def _load_language_action_from_file(
        self,
        episode_index: int,
        task_idx: int,
        condition_frame_idx: int,
    ) -> str:
        if self.task_mode == "single":
            ep_meta = self.lerobot_dataset.meta.episodes.get(episode_index, None)
            dataset_root = Path(self.lerobot_dataset.root)
        else:
            ep_meta = self.lerobot_dataset._datasets[task_idx].meta.episodes.get(episode_index, None)
            dataset_root = Path(self.lerobot_dataset._datasets[task_idx].root)
        if ep_meta is None:
            raise KeyError(f"episode {episode_index} not found in meta.episodes")

        rel_path = ep_meta.get("language_action_path", None)
        if rel_path is None:
            rel_path = f"{self.language_action_dir_name}/episode_{episode_index:06d}.txt"

        lang_action_path = Path(str(rel_path))
        if not lang_action_path.is_absolute():
            lang_action_path = dataset_root / lang_action_path
        if not lang_action_path.exists() and ep_meta.get("language_action_path", None) is None:
            raise KeyError(
                "language_action_path not found in meta/episodes.jsonl and default "
                f"{self.language_action_dir_name}/episode_{episode_index:06d}.txt does not exist"
            )

        cache_key = (task_idx, episode_index)
        if cache_key in self._language_action_cache:
            return self._select_language_action_line(
                self._language_action_cache[cache_key],
                condition_frame_idx,
                str(lang_action_path),
            )

        with lang_action_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        self._language_action_cache[cache_key] = lines
        return self._select_language_action_line(lines, condition_frame_idx, str(lang_action_path))

    def _load_language_action(
        self,
        item_cond: Dict[str, Any],
        episode_index: int,
        task_idx: int,
        condition_frame_idx: int,
    ) -> str:
        for key in ("language_action", "observation.language_action"):
            value = item_cond.get(key, None)
            if value is None:
                continue
            if isinstance(value, str):
                if value.strip():
                    return value.strip()
                continue
            if isinstance(value, (list, tuple)):
                lines = [str(v).strip() for v in value if str(v).strip()]
                return self._select_language_action_line(lines, condition_frame_idx, key)

        return self._load_language_action_from_file(
            episode_index=episode_index,
            task_idx=task_idx,
            condition_frame_idx=condition_frame_idx,
        )
