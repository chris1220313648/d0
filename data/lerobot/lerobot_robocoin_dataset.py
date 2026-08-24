"""
RoboCOIN LeRobot Dataset Loader for Motus
-----------------------------------------
RoboCOIN is LeRobot v2.1-compatible, but each robot/task can expose different
state/action dimensions and camera names. This wrapper returns raw
actions/states with feature names; Canonical55Dataset maps them directly to
fixed 55D training tensors before collation.

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
RoboCOIN corpus has heterogeneous action/state dimensions; the feature names
in meta/info.json define how raw coordinates map into canonical55 slots.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from lerobot.datasets.video_utils import decode_video_frames

from .lerobot_dataset import LeRobotMotusDataset, preprocess_vlm_messages, tensor_to_pil
from utils.vlm_utils import preprocess_vlm_messages_lap


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoboCOINEpisodeTarget:
    task: str
    task_idx: int
    episode_position: int
    episode_index: int


class LeRobotRoboCOINDataset(LeRobotMotusDataset):
    """RoboCOIN-specific LeRobot wrapper with raw heterogeneous actions."""
    DEFAULT_BAD_EPISODES_PATH = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "motus-robocoin_lap_v2"
        / "robocoin_full_integrity_scan_20260803"
        / "bad_episodes.jsonl"
    )
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
        bad_episodes_path: Optional[str | Sequence[str]] = None,
        skip_bad_episodes: bool = True,
        sample_retry_attempts: int = 16,
        sample_fallback_attempts: int = 64,
        **kwargs,
    ):
        kwargs["use_multi_lerobot_dataset"] = False
        # Older configs used these raw padding knobs. They are intentionally
        # ignored now because canonical55 mapping happens before collation.
        kwargs.pop("target_action_dim", None)
        kwargs.pop("target_state_dim", None)
        if normalize_actions:
            raise ValueError(
                "LeRobotRoboCOINDataset emits heterogeneous raw RoboCOIN actions; "
                "normalization requires per-robot canonical stats and is intentionally disabled."
            )
        super().__init__(*args, **kwargs)
        self.use_language_action = bool(use_language_action)
        self.language_action_dir_name = str(language_action_dir_name).strip() or "language_action"
        self.visual_keys_override = tuple(visual_keys) if visual_keys is not None else None
        self._language_action_cache: Dict[Tuple[int, int], List[str]] = {}
        self._visual_keys_cache: Dict[str, Tuple[str, ...]] = {}
        self.skip_bad_episodes = bool(skip_bad_episodes)
        self.sample_retry_attempts = max(1, int(sample_retry_attempts))
        self.sample_fallback_attempts = max(0, int(sample_fallback_attempts))
        self.bad_episode_keys = self._load_bad_episode_keys(bad_episodes_path)
        self._sample_episode_targets = self._filtered_episode_targets()
        if not self._sample_episode_targets:
            raise ValueError("RoboCOIN dataset has no eligible episodes after blacklist filtering")
        if self.bad_episode_keys:
            logger.info(
                "Loaded %d RoboCOIN bad episode keys; excluded %d selected episodes from random sampling",
                len(self.bad_episode_keys),
                self._count_selected_bad_episodes(),
            )

    def __getitem__(self, idx):
        if not self.lerobot_dataset:
            return None

        last_error: Optional[BaseException] = None
        candidates = self._sample_episode_targets
        for _ in range(self.sample_retry_attempts):
            target = candidates[random.randint(0, len(candidates) - 1)]
            try:
                return self.load_episode_sample(target.task_idx, target.episode_position)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "RoboCOIN sample load failed for task=%s episode_index=%s; retrying",
                    target.task,
                    target.episode_index,
                    exc_info=True,
                )

        fallback_attempts = min(self.sample_fallback_attempts, len(candidates))
        start = int(idx) % len(candidates)
        for offset in range(fallback_attempts):
            target = candidates[(start + offset) % len(candidates)]
            try:
                return self.load_episode_sample(target.task_idx, target.episode_position)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "RoboCOIN fallback sample load failed for task=%s episode_index=%s",
                    target.task,
                    target.episode_index,
                    exc_info=True,
                )

        raise RuntimeError(
            "RoboCOIN sampler could not load a valid sample after "
            f"{self.sample_retry_attempts} random retries and {fallback_attempts} fallback attempts"
        ) from last_error

    def validation_episode_targets(self) -> List[RoboCOINEpisodeTarget]:
        """Return every selected episode exactly once in loader order."""
        return self._filtered_episode_targets()

    def _filtered_episode_targets(self) -> List[RoboCOINEpisodeTarget]:
        bad_episode_keys = getattr(self, "bad_episode_keys", set())
        if not bad_episode_keys:
            return self._all_episode_targets()
        return [
            target
            for target in self._all_episode_targets()
            if (target.task, target.episode_index) not in bad_episode_keys
        ]

    def _all_episode_targets(self) -> List[RoboCOINEpisodeTarget]:
        if self.task_mode == "multi":
            return [
                RoboCOINEpisodeTarget(
                    task=task,
                    task_idx=task_idx,
                    episode_position=position,
                    episode_index=int(episode_index),
                )
                for task_idx, task in enumerate(self.repo_ids)
                for position, episode_index in enumerate(self.episode_ids[task])
            ]
        return [
            RoboCOINEpisodeTarget(
                task=str(self.repo_id),
                task_idx=0,
                episode_position=position,
                episode_index=int(episode_index),
            )
            for position, episode_index in enumerate(self.episode_ids)
        ]

    def _count_selected_bad_episodes(self) -> int:
        bad_episode_keys = getattr(self, "bad_episode_keys", set())
        return sum(
            1
            for target in self._all_episode_targets()
            if (target.task, target.episode_index) in bad_episode_keys
        )

    def _load_bad_episode_keys(
        self,
        bad_episodes_path: Optional[str | Sequence[str]],
    ) -> set[Tuple[str, int]]:
        if not self.skip_bad_episodes:
            return set()

        explicit = bad_episodes_path is not None
        if bad_episodes_path is None:
            paths: Sequence[str | Path] = (self.DEFAULT_BAD_EPISODES_PATH,)
        elif isinstance(bad_episodes_path, (str, Path)):
            paths = (bad_episodes_path,)
        else:
            paths = tuple(bad_episodes_path)

        keys: set[Tuple[str, int]] = set()
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                if explicit:
                    raise FileNotFoundError(f"RoboCOIN bad_episodes_path does not exist: {path}")
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        keys.add((str(row["task"]), int(row["episode_index"])))
                    except Exception as exc:
                        raise ValueError(f"Invalid bad episode row in {path}:{line_no}") from exc
        return keys

    def load_episode_sample(
        self,
        task_idx: int,
        episode_position: int,
        condition_frame_idx: Optional[int | str] = None,
    ) -> Dict[str, Any]:
        """Load one explicitly selected episode/window through the training path."""
        if not self.lerobot_dataset:
            raise RuntimeError("RoboCOIN dataset is not initialized")

        if self.task_mode == "multi":
            ds_media = self.lerobot_dataset._datasets[int(task_idx)]
        else:
            if int(task_idx) != 0:
                raise IndexError(f"single-task RoboCOIN dataset has no task_idx={task_idx}")
            ds_media = self.lerobot_dataset

        episode_position = int(episode_position)
        from_idx_t = ds_media.episode_data_index["from"][episode_position]
        to_idx_t = ds_media.episode_data_index["to"][episode_position]

        from_idx = int(from_idx_t.item()) if hasattr(from_idx_t, "item") else int(from_idx_t)
        to_idx = int(to_idx_t.item()) if hasattr(to_idx_t, "item") else int(to_idx_t)
        total_frames = int(to_idx - from_idx)
        condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(
            total_frames,
            condition_frame_idx=condition_frame_idx,
        )

        local_cond_idx = int(from_idx + condition_frame_idx)
        local_video_indices = [int(from_idx + i) for i in video_indices]
        local_action_indices = [int(from_idx + i) for i in action_indices]
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
            state_names = self._feature_names(ds_media, "observation.state")
        elif "action" in item_cond:
            initial_state_raw = torch.as_tensor(item_cond["action"]).float()
            state_names = self._feature_names(ds_media, "action")
        elif "actions" in item_cond:
            initial_state_raw = torch.as_tensor(item_cond["actions"]).float()
            state_names = self._feature_names(ds_media, "actions")
        else:
            raise KeyError("No state found in item (expected observation.state/actions/action)")
        initial_state = initial_state_raw.float()

        action_key = "action" if "action" in hf_dataset.column_names else None
        if action_key is None and "actions" in hf_dataset.column_names:
            action_key = "actions"
        if action_key is None:
            raise KeyError("No action column found in hf_dataset (expected 'action' or 'actions')")
        action_names = self._feature_names(ds_media, action_key)

        raw_action_sequence = self._to_action_tensor(hf_dataset[local_action_indices][action_key])
        if raw_action_sequence.ndim == 1:
            raw_action_sequence = raw_action_sequence.unsqueeze(0)
        action_sequence = raw_action_sequence.float()

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
            vlm_tokens = self._preprocess_vlm_inputs(
                text_instr,
                first_frame_pil,
                language_action,
            )

        return {
            "first_frame": first_frame,
            "video_frames": video_frames_sampled,
            "initial_state": initial_state,
            "state_names": state_names,
            "action_sequence": action_sequence,
            "action_names": action_names,
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

    @staticmethod
    def _feature_names(ds_media, key: str) -> Tuple[str, ...]:
        feature = ds_media.features.get(key, None)
        if isinstance(feature, Mapping):
            names = feature.get("names", None)
            if isinstance(names, (list, tuple)):
                return tuple(str(name) for name in names)
        return ()

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

    def _load_language_embedding(self, item_cond: Dict[str, Any], task_idx: int) -> torch.Tensor:
        all_embeddings = item_cond.get("language_embedding", None)
        if all_embeddings is None:
            all_embeddings = item_cond.get("observation.feature.language_embedding", None)

        if all_embeddings is None:
            ep_index_raw = item_cond.get("episode_index", None)
            if ep_index_raw is None:
                raise KeyError("episode_index not found in item; cannot load external embedding")
            ep_index = int(ep_index_raw.item()) if hasattr(ep_index_raw, "item") else int(ep_index_raw)

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
                all_embeddings = self._encode_and_cache_t5_embedding(ep_index, str(instr))
            else:
                abs_path = Path(str(rel_path))
                if not abs_path.is_absolute():
                    abs_path = dataset_root / abs_path
                all_embeddings = torch.load(abs_path, map_location="cpu", weights_only=True)
                if not isinstance(all_embeddings, torch.Tensor):
                    all_embeddings = torch.tensor(all_embeddings)
                if all_embeddings.ndim == 2:
                    all_embeddings = all_embeddings.unsqueeze(0)

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
    ) -> Optional[str]:
        if self.task_mode == "single":
            ep_meta = self.lerobot_dataset.meta.episodes.get(episode_index, None)
            dataset_root = Path(self.lerobot_dataset.root)
        else:
            ep_meta = self.lerobot_dataset._datasets[task_idx].meta.episodes.get(episode_index, None)
            dataset_root = Path(self.lerobot_dataset._datasets[task_idx].root)
        if ep_meta is None:
            raise KeyError(f"episode {episode_index} not found in meta.episodes")

        declared_rel_path = ep_meta.get("language_action_path", None)
        rel_path = declared_rel_path
        if rel_path is None:
            rel_path = f"{self.language_action_dir_name}/episode_{episode_index:06d}.txt"

        lang_action_path = Path(str(rel_path))
        if not lang_action_path.is_absolute():
            lang_action_path = dataset_root / lang_action_path
        if not lang_action_path.exists() and declared_rel_path is None:
            return None

        with lang_action_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return self._select_language_action_line(lines, condition_frame_idx, str(lang_action_path))

    def _load_language_action(
        self,
        item_cond: Dict[str, Any],
        episode_index: int,
        task_idx: int,
        condition_frame_idx: int,
    ) -> Optional[str]:
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

    def _preprocess_vlm_inputs(
        self,
        text_instr: str,
        first_frame_pil,
        language_action: Optional[str],
    ):
        if self.use_language_action and language_action is not None:
            return preprocess_vlm_messages_lap(
                text_instr,
                first_frame_pil,
                self.vlm_processor,
                language_action,
                supervise_answer=True,
            )
        return preprocess_vlm_messages(
            text_instr,
            first_frame_pil,
            self.vlm_processor,
        )
