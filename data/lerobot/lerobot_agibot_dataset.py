"""
LeRobot AgiBot Dataset Loader for Motus
---------------------------------------
Specialized LeRobot wrapper for AgiBotWorld-style datasets.

It keeps the Motus dataset interface used by data/dataset.py::collate_fn, and
maps AgiBot's named raw state/action fields directly into canonical55_v2.

Expected AgiBotWorld2026 data structure:
  <root>/  # e.g. /root/nas/code/d0/data/robot_data/AgiBotWorld2026
    ImitationLearning|RichInteraction/
      .../
        <task_split>/
          meta/
            info.json
            episodes.jsonl
            episodes_stats.jsonl
            tasks.jsonl
          data/
            chunk-000/
              episode_000000.parquet
              episode_000001.parquet
              ...
          videos/
            chunk-000/
              observation.images.top_head/
                episode_000000.mp4
                ...
              observation.images.hand_left/
                episode_000000.mp4
                ...
              observation.images.hand_right/
                episode_000000.mp4
                ...
          t5_embedding/
            episode_000000.pt
            episode_000001.pt
            ...
          language_action/
            episode_000000.txt
            episode_000001.txt
            ...

meta/episodes.jsonl should contain episode_index, length, tasks, plus
t5_embedding_path and language_action_path when offline embeddings and LAP
language-action supervision are enabled.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.video_utils import decode_video_frames

from .lerobot_dataset import LeRobotMotusDataset, preprocess_vlm_messages, tensor_to_pil
from data.canonical55 import (
    ARM_JOINT,
    BASE,
    CANONICAL55_DIM,
    HEAD,
    LEFT_EEF,
    RIGHT_EEF,
    WAIST,
)
from data.utils.image_utils import resize_with_padding
from data.utils.norm import load_normalization_stats, normalize_actions
from utils.vlm_utils import append_setup_control_suffix, preprocess_vlm_messages_lap


class LeRobotAgiBotDataset(LeRobotMotusDataset):
    """AgiBot-specific LeRobot wrapper with canonical55 state/action mapping."""

    ACTION_RAW_MIN_DIM: int = 40
    DEFAULT_VISUAL_KEYS: Tuple[str, str, str] = (
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    )

    def __init__(
        self,
        *args,
        visual_keys: Optional[Sequence[str]] = None,
        language_action_dir_name: str = "language_action",
        normalize_actions: bool = False,
        stats_path: Optional[str] = None,
        stats_key: str = "agibot",
        use_language_action: bool = False,
        output_format: str = "canonical55",
        enable_setup_control_suffix: bool = False,
        setup_text: str = "dual-arm AgiBot robot with grippers",
        action_signal: str = "epos",
        **kwargs,
    ):
        self._feature_fields_cache: Dict[Tuple[str, str], Dict[str, Tuple[int, ...]]] = {}
        kwargs["use_multi_lerobot_dataset"] = False
        self.output_format = str(output_format)
        if self.output_format not in {"canonical55", "dual_eef14_joint14"}:
            raise ValueError(f"Unsupported AgiBot output_format: {self.output_format}")
        if self.output_format == "dual_eef14_joint14" and normalize_actions:
            raise ValueError("AgiBot dual_eef14_joint14 output does not support 55D normalization")
        super().__init__(*args, **kwargs)
        self.normalize_actions = bool(normalize_actions)
        self.stats_key = str(stats_key)
        self.agibot_state_min = None
        self.agibot_state_max = None
        self.agibot_action_min = None
        self.agibot_action_max = None
        if getattr(self, "normalize_actions", False):
            if stats_path is None:
                stats_path = str((Path(__file__).resolve().parent.parent / "utils" / "stat.json"))
            self.agibot_state_min, self.agibot_state_max = load_normalization_stats(
                stats_path,
                self.stats_key,
                "state",
            )
            self.agibot_action_min, self.agibot_action_max = load_normalization_stats(
                stats_path,
                self.stats_key,
                "action",
            )
            for name, stat in (
                ("state min", self.agibot_state_min),
                ("state max", self.agibot_state_max),
                ("action min", self.agibot_action_min),
                ("action max", self.agibot_action_max),
            ):
                if stat is None or len(stat) != CANONICAL55_DIM:
                    raise ValueError(
                        f"AgiBot normalization requires {self.stats_key}/state and "
                        f"{self.stats_key}/action 55D stats in {stats_path}; invalid {name}."
                    )
        self.use_language_action = bool(use_language_action)
        self.enable_setup_control_suffix = bool(enable_setup_control_suffix)
        self.setup_text = str(setup_text)
        self.action_signal = str(action_signal)
        self.language_action_dir_name = str(language_action_dir_name).strip() or "language_action"
        self._language_action_cache: Dict[Tuple[int, int], List[str]] = {}

        keys = tuple(visual_keys) if visual_keys is not None else self.DEFAULT_VISUAL_KEYS
        if len(keys) != 3:
            raise ValueError(f"visual_keys must contain exactly 3 video keys, got {keys}")
        self.visual_keys = keys

        if self.task_mode == "single":
            feature_sets = [(self.repo_id, self.lerobot_dataset.features)]
        else:
            feature_sets = [
                (repo_id, dataset.features)
                for repo_id, dataset in zip(self.repo_ids, self.lerobot_dataset._datasets)
            ]
        missing_by_repo = {
            repo_id: [key for key in self.visual_keys if key not in features]
            for repo_id, features in feature_sets
        }
        missing_by_repo = {repo_id: missing for repo_id, missing in missing_by_repo.items() if missing}
        if missing_by_repo:
            raise ValueError(f"AgiBot visual keys not found in dataset features: {missing_by_repo}")

    def __getitem__(self, idx):
        if not self.lerobot_dataset:
            return None

        episode_idx = self._sample_episode_index()
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
        all_media_indices = [local_cond_idx] + local_video_indices
        timestamps = self._read_timestamps(hf_dataset, all_media_indices)

        ep_idx_raw = item_cond.get("episode_index", None)
        if ep_idx_raw is None:
            raise KeyError("episode_index not found in hf_dataset row; cannot resolve video file path")
        ep_for_video = int(ep_idx_raw.item()) if hasattr(ep_idx_raw, "item") else int(ep_idx_raw)

        first_frame, video_frames_sampled = self._load_stitched_video_frames(ds_media, ep_for_video, timestamps)

        if "observation.state" not in item_cond:
            raise KeyError("observation.state not found in item")
        raw_state = torch.as_tensor(item_cond["observation.state"]).float()

        action_key = "action" if "action" in hf_dataset.column_names else None
        if action_key is None and "actions" in hf_dataset.column_names:
            action_key = "actions"
        if action_key is None:
            raise KeyError("No action column found in hf_dataset (expected 'action' or 'actions')")
        if self.output_format == "dual_eef14_joint14":
            initial_state, state_mask = self._select_state_14d(raw_state, ds_media)
            action_sequence, action_mask = self._select_action_sequence_14d(
                hf_dataset[local_action_indices][action_key],
                item_cond[action_key],
                ds_media,
            )
        else:
            initial_state, state_mask = self._select_state(raw_state, ds_media)
            action_sequence, action_mask = self._select_action_sequence(
                hf_dataset[local_action_indices][action_key],
                item_cond[action_key],
                ds_media,
            )

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
            text_instr = append_setup_control_suffix(
                text_instruction=text_instr,
                enable_setup_control_suffix=self.enable_setup_control_suffix,
                setup_text=self.setup_text,
                action_signal=self.action_signal,
            )
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
            "state_mask": state_mask,
            "action_mask": action_mask,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_tokens,
        }

    def _sample_episode_index(self) -> int:
        return int(np.random.randint(0, self.lerobot_dataset.num_episodes))

    def _read_timestamps(self, hf_dataset, indices: List[int]) -> List[float]:
        ts_vals = hf_dataset[indices]["timestamp"]
        if isinstance(ts_vals, torch.Tensor):
            return ts_vals.flatten().tolist()
        if isinstance(ts_vals, (list, tuple)) and len(ts_vals) > 0 and isinstance(ts_vals[0], torch.Tensor):
            return torch.stack(ts_vals).flatten().tolist()
        return [float(x) for x in list(ts_vals)]

    def _decode_key(self, ds_media, episode_index: int, video_key: str, timestamps: List[float]) -> torch.Tensor:
        video_path = Path(ds_media.root) / ds_media.meta.get_video_file_path(episode_index, video_key)
        frames = decode_video_frames(video_path, timestamps, ds_media.tolerance_s, ds_media.video_backend).squeeze(0)
        return frames

    def _load_stitched_video_frames(self, ds_media, episode_index: int, timestamps: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        decoded = [
            self._decode_key(ds_media, episode_index, video_key, timestamps)
            for video_key in self.visual_keys
        ]

        stitched = []
        for frame_idx in range(decoded[0].shape[0]):
            stitched.append(
                self._stitch_three_views_uint8(
                    decoded[0][frame_idx],
                    decoded[1][frame_idx],
                    decoded[2][frame_idx],
                )
            )
        # Release all three full-resolution decode batches before materializing
        # the model-facing float32 frames.
        del decoded
        # The source frames and the temporary stitch canvas stay uint8. Convert
        # only the final 384x320 outputs consumed by the model.
        output_frames = [
            torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
            for frame in stitched
        ]
        first_frame = output_frames[0]
        video_frames = torch.stack(output_frames[1:], dim=0)
        return first_frame, video_frames

    def _stitch_three_views_uint8(
        self,
        top_frame: torch.Tensor,
        left_frame: torch.Tensor,
        right_frame: torch.Tensor,
    ) -> np.ndarray:
        """Arrange three decoded views without full-resolution float tensors."""
        c = int(top_frame.shape[0])
        canvas_w = int(max(top_frame.shape[2], left_frame.shape[2] + right_frame.shape[2]))
        top_h = int(top_frame.shape[1])
        bottom_h = int(max(left_frame.shape[1], right_frame.shape[1]))
        left_w = canvas_w // 2
        right_w = canvas_w - left_w

        top = self._resize_frame_chw_uint8(top_frame, (top_h, canvas_w))
        left = self._resize_frame_chw_uint8(left_frame, (bottom_h, left_w))
        right = self._resize_frame_chw_uint8(right_frame, (bottom_h, right_w))

        out = np.zeros((top_h + bottom_h, canvas_w, c), dtype=np.uint8)
        out[:top_h, :] = top
        out[top_h:, :left_w] = left
        out[top_h:, left_w:] = right
        return resize_with_padding(out, self.video_size)

    def _feature_fields(self, ds_media, feature_name: str) -> Dict[str, Tuple[int, ...]]:
        dataset_root = Path(ds_media.root)
        cache_key = (str(dataset_root), str(feature_name))
        cached = self._feature_fields_cache.get(cache_key)
        if cached is not None:
            return cached

        feature = getattr(ds_media, "features", {}).get(feature_name, {})
        field_descriptions = feature.get("field_descriptions", {}) if isinstance(feature, dict) else {}
        if not field_descriptions:
            info_path = dataset_root / "meta" / "info.json"
            if not info_path.is_file():
                raise FileNotFoundError(f"AgiBot meta info not found: {info_path}")
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            feature = info.get("features", {}).get(feature_name, {})
            field_descriptions = feature.get("field_descriptions", {}) if isinstance(feature, dict) else {}

        fields: Dict[str, Tuple[int, ...]] = {}
        for field_name, description in field_descriptions.items():
            indices = description.get("indices", []) if isinstance(description, dict) else []
            fields[str(field_name)] = tuple(int(index) for index in indices)
        self._feature_fields_cache[cache_key] = fields
        return fields

    def _empty_canonical(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (*source.shape[:-1], CANONICAL55_DIM)
        return source.new_zeros(shape), torch.zeros(shape, dtype=torch.bool, device=source.device)

    def _empty_14d(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (*source.shape[:-1], 14)
        return source.new_zeros(shape), torch.zeros(shape, dtype=torch.bool, device=source.device)

    def _field_tensor(
        self,
        source: torch.Tensor,
        fields: Dict[str, Tuple[int, ...]],
        field_name: str,
        expected_dim: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        indices = fields.get(field_name)
        if not indices:
            return None
        if expected_dim is not None and len(indices) < expected_dim:
            return None
        if max(indices) >= source.shape[-1]:
            raise ValueError(
                f"AgiBot field {field_name!r} index {max(indices)} exceeds tensor shape {tuple(source.shape)}"
            )
        index = torch.tensor(indices[:expected_dim], dtype=torch.long, device=source.device)
        return source.index_select(-1, index).float()

    def _assign_tensor(self, value: torch.Tensor, mask: torch.Tensor, dst: slice, src: Optional[torch.Tensor]) -> None:
        if src is None:
            return
        value[..., dst] = src
        mask[..., dst] = True

    def _assign_scalar(self, value: torch.Tensor, mask: torch.Tensor, dst: int, src: Optional[torch.Tensor]) -> None:
        if src is None:
            return
        value[..., dst] = src.squeeze(-1)
        mask[..., dst] = True

    def _required_field_tensor(
        self,
        source: torch.Tensor,
        fields: Dict[str, Tuple[int, ...]],
        field_name: str,
        expected_dim: int,
    ) -> torch.Tensor:
        value = self._field_tensor(source, fields, field_name, expected_dim)
        if value is None:
            raise KeyError(f"AgiBot field {field_name!r} with {expected_dim} dims is required")
        return value

    def _quat_to_rotvec(self, quat_values: torch.Tensor) -> torch.Tensor:
        quat_np = quat_values.detach().cpu().numpy()
        return torch.from_numpy(Rotation.from_quat(quat_np).as_rotvec()).to(
            device=quat_values.device,
            dtype=torch.float32,
        )

    def _assign_eef_pose(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        position: Optional[torch.Tensor],
        orientation: Optional[torch.Tensor],
        base_orientation: Optional[torch.Tensor] = None,
    ) -> None:
        if position is not None:
            self._assign_tensor(value, mask, slice(LEFT_EEF.start, LEFT_EEF.start + 3), position[..., 0:3])
            self._assign_tensor(value, mask, slice(RIGHT_EEF.start, RIGHT_EEF.start + 3), position[..., 3:6])

        if orientation is None:
            return
        if base_orientation is None:
            left_rotvec = self._quat_to_rotvec(orientation[..., 0:4])
            right_rotvec = self._quat_to_rotvec(orientation[..., 4:8])
        else:
            left_rotvec = self._relative_rotvec(orientation[..., 0:4], base_orientation[..., 0:4])
            right_rotvec = self._relative_rotvec(orientation[..., 4:8], base_orientation[..., 4:8])
        self._assign_tensor(value, mask, slice(LEFT_EEF.start + 3, LEFT_EEF.stop), left_rotvec)
        self._assign_tensor(value, mask, slice(RIGHT_EEF.start + 3, RIGHT_EEF.stop), right_rotvec)

    def _apply_masked_normalization(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        stat_min: Optional[np.ndarray],
        stat_max: Optional[np.ndarray],
    ) -> torch.Tensor:
        normalized = normalize_actions(value.cpu(), stat_min, stat_max).to(device=value.device, dtype=value.dtype)
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    def _select_state(self, raw_state: torch.Tensor, ds_media) -> Tuple[torch.Tensor, torch.Tensor]:
        fields = self._feature_fields(ds_media, "observation.state")
        value, mask = self._empty_canonical(raw_state)

        self._assign_tensor(value, mask, ARM_JOINT, self._field_tensor(raw_state, fields, "state/joint/position", 14))
        self._assign_eef_pose(
            value,
            mask,
            self._field_tensor(raw_state, fields, "state/end/arm_position", 6),
            self._field_tensor(raw_state, fields, "state/end/arm_orientation", 8),
        )
        self._assign_scalar(
            value,
            mask,
            26,
            self._field_tensor(raw_state, fields, "state/left_effector/position", 1),
        )
        self._assign_scalar(
            value,
            mask,
            27,
            self._field_tensor(raw_state, fields, "state/right_effector/position", 1),
        )
        self._assign_tensor(value, mask, WAIST, self._field_tensor(raw_state, fields, "state/waist/position", 4))
        self._assign_tensor(value, mask, HEAD, self._field_tensor(raw_state, fields, "state/head/position", 2))

        if getattr(self, "normalize_actions", False):
            value = self._apply_masked_normalization(value, mask, self.agibot_state_min, self.agibot_state_max)
        return value.float(), mask

    def _select_state_14d(self, raw_state: torch.Tensor, ds_media) -> Tuple[torch.Tensor, torch.Tensor]:
        fields = self._feature_fields(ds_media, "observation.state")
        joint = self._required_field_tensor(raw_state, fields, "state/joint/position", 14)
        left_gripper = self._required_field_tensor(raw_state, fields, "state/left_effector/position", 1)
        right_gripper = self._required_field_tensor(raw_state, fields, "state/right_effector/position", 1)

        value, mask = self._empty_14d(raw_state)
        value[..., 0:6] = joint[..., 0:6]
        value[..., 6] = left_gripper.squeeze(-1)
        value[..., 7:13] = joint[..., 7:13]
        value[..., 13] = right_gripper.squeeze(-1)
        mask[...] = True
        return value.float(), mask

    def _to_action_tensor(self, values: Any) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.float()
        if isinstance(values, (list, tuple)) and len(values) > 0 and isinstance(values[0], torch.Tensor):
            return torch.stack([v.float() for v in values], dim=0)
        if isinstance(values, (list, tuple)) and len(values) > 0 and isinstance(values[0], np.ndarray):
            return torch.from_numpy(np.stack(values, axis=0)).float()
        return torch.tensor(values, dtype=torch.float32)

    def _relative_rotvec(self, quat_values: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor:
        quat_np = quat_values.detach().cpu().numpy()
        base_quat_np = base_quat.detach().cpu().numpy()
        rel_rot = Rotation.from_quat(quat_np) * Rotation.from_quat(base_quat_np).inv()
        return torch.from_numpy(rel_rot.as_rotvec()).to(device=quat_values.device, dtype=torch.float32)

    def _select_action_sequence(
        self,
        action_values: Any,
        base_action_value: Any,
        ds_media,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_actions = self._to_action_tensor(action_values)
        base_action = self._to_action_tensor(base_action_value).flatten()

        if raw_actions.ndim == 1:
            raw_actions = raw_actions.unsqueeze(0)
        if raw_actions.shape[-1] < self.ACTION_RAW_MIN_DIM:
            raise ValueError(f"AgiBot action must have at least 40 dims, got {tuple(raw_actions.shape)}")
        if base_action.shape[-1] < self.ACTION_RAW_MIN_DIM:
            raise ValueError(f"AgiBot base action must have at least 40 dims, got {tuple(base_action.shape)}")

        fields = self._feature_fields(ds_media, "action")
        value, mask = self._empty_canonical(raw_actions)
        self._assign_tensor(value, mask, ARM_JOINT, self._field_tensor(raw_actions, fields, "action/joint/position", 14))
        self._assign_eef_pose(
            value,
            mask,
            self._field_tensor(raw_actions, fields, "action/end/position", 6),
            self._field_tensor(raw_actions, fields, "action/end/orientation", 8),
            self._field_tensor(base_action, fields, "action/end/orientation", 8),
        )
        self._assign_scalar(
            value,
            mask,
            26,
            self._field_tensor(raw_actions, fields, "action/left_effector/position", 1),
        )
        self._assign_scalar(
            value,
            mask,
            27,
            self._field_tensor(raw_actions, fields, "action/right_effector/position", 1),
        )
        self._assign_tensor(value, mask, WAIST, self._field_tensor(raw_actions, fields, "action/waist/position", 4))
        self._assign_tensor(value, mask, HEAD, self._field_tensor(raw_actions, fields, "action/head/position", 2))
        self._assign_tensor(
            value,
            mask,
            slice(BASE.start, BASE.start + 2),
            self._field_tensor(raw_actions, fields, "action/robot/velocity", 2),
        )

        if getattr(self, "normalize_actions", False):
            value = self._apply_masked_normalization(value, mask, self.agibot_action_min, self.agibot_action_max)
        return value.float(), mask

    def _select_action_sequence_14d(
        self,
        action_values: Any,
        base_action_value: Any,
        ds_media,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_actions = self._to_action_tensor(action_values)
        base_action = self._to_action_tensor(base_action_value).flatten()

        if raw_actions.ndim == 1:
            raw_actions = raw_actions.unsqueeze(0)
        if raw_actions.shape[-1] < self.ACTION_RAW_MIN_DIM:
            raise ValueError(f"AgiBot action must have at least 40 dims, got {tuple(raw_actions.shape)}")
        if base_action.shape[-1] < self.ACTION_RAW_MIN_DIM:
            raise ValueError(f"AgiBot base action must have at least 40 dims, got {tuple(base_action.shape)}")

        fields = self._feature_fields(ds_media, "action")
        position = self._required_field_tensor(raw_actions, fields, "action/end/position", 6)
        orientation = self._required_field_tensor(raw_actions, fields, "action/end/orientation", 8)
        base_orientation = self._required_field_tensor(base_action, fields, "action/end/orientation", 8)
        left_gripper = self._required_field_tensor(raw_actions, fields, "action/left_effector/position", 1)
        right_gripper = self._required_field_tensor(raw_actions, fields, "action/right_effector/position", 1)

        value, mask = self._empty_14d(raw_actions)
        value[..., 0:3] = position[..., 0:3]
        value[..., 7:10] = position[..., 3:6]
        value[..., 3:6] = self._relative_rotvec(orientation[..., 0:4], base_orientation[..., 0:4])
        value[..., 10:13] = self._relative_rotvec(orientation[..., 4:8], base_orientation[..., 4:8])
        value[..., 6] = left_gripper.squeeze(-1)
        value[..., 13] = right_gripper.squeeze(-1)
        mask[...] = True
        return value.float(), mask

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
            else:
                ep_meta = self.lerobot_dataset._datasets[task_idx].meta.episodes.get(ep_index, None)
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
                if not isinstance(instr, str):
                    instr = str(instr)
                all_embeddings = self._encode_and_cache_t5_embedding(ep_index, instr)
            else:
                if self.task_mode == "single":
                    abs_path = Path(self.lerobot_dataset.root) / str(rel_path)
                else:
                    abs_path = Path(self.lerobot_dataset._datasets[task_idx].root) / str(rel_path)
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

        with open(lang_action_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
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
