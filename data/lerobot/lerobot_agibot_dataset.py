"""
LeRobot AgiBot Dataset Loader for Motus
---------------------------------------
Specialized LeRobot wrapper for AgiBotWorld-style datasets.

It keeps the Motus dataset interface used by data/dataset.py::collate_fn, but
selects compact AgiBot action/state vectors from the wide raw columns:

action [24]:
  left arm relative eef [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z, gripper],
  right arm relative eef [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z, gripper],
  head [30:33], waist [33:38], base velocity [38:40]

state [14]:
  joint position resolved from each split's meta/info.json field_descriptions
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.video_utils import decode_video_frames

from .lerobot_dataset import LeRobotMotusDataset, preprocess_vlm_messages, tensor_to_pil
from utils.vlm_utils import preprocess_vlm_messages_lap


class LeRobotAgiBotDataset(LeRobotMotusDataset):
    """AgiBot-specific LeRobot wrapper with compact action/state selection."""

    ACTION_RAW_MIN_DIM: int = 40
    STATE_FIELD_NAME: str = "state/joint/position"
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
        use_language_action: bool = False,
        **kwargs,
    ):
        self._state_indices_cache: Dict[str, Tuple[int, ...]] = {}
        kwargs["use_multi_lerobot_dataset"] = False
        if normalize_actions:
            raise ValueError(
                "LeRobotAgiBotDataset returns selected 24D action / 14D state without normalization. "
                "Generate matching AgiBot stats before enabling normalization."
            )
        super().__init__(*args, **kwargs)
        self.use_language_action = bool(use_language_action)
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
        initial_state = self._select_state(raw_state, ds_media)

        action_key = "action" if "action" in hf_dataset.column_names else None
        if action_key is None and "actions" in hf_dataset.column_names:
            action_key = "actions"
        if action_key is None:
            raise KeyError("No action column found in hf_dataset (expected 'action' or 'actions')")
        action_sequence = self._select_action_sequence(
            hf_dataset[local_action_indices][action_key],
            item_cond[action_key],
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
            self._decode_key(ds_media, episode_index, video_key, timestamps).float()
            for video_key in self.visual_keys
        ]

        stitched = []
        for frame_idx in range(decoded[0].shape[0]):
            stitched.append(
                self._stitch_three_views(
                    decoded[0][frame_idx],
                    decoded[1][frame_idx],
                    decoded[2][frame_idx],
                )
            )
        first_frame = stitched[0]
        video_frames = torch.stack(stitched[1:], dim=0)
        return first_frame, video_frames

    def _stitch_three_views(
        self,
        top_frame: torch.Tensor,
        left_frame: torch.Tensor,
        right_frame: torch.Tensor,
    ) -> torch.Tensor:
        c = int(top_frame.shape[0])
        canvas_w = int(max(top_frame.shape[2], left_frame.shape[2] + right_frame.shape[2]))
        top_h = int(top_frame.shape[1])
        bottom_h = int(max(left_frame.shape[1], right_frame.shape[1]))
        left_w = canvas_w // 2
        right_w = canvas_w - left_w

        top = self._resize_frame_chw(top_frame, (top_h, canvas_w))
        left = self._resize_frame_chw(left_frame, (bottom_h, left_w))
        right = self._resize_frame_chw(right_frame, (bottom_h, right_w))

        out = torch.zeros((c, top_h + bottom_h, canvas_w), dtype=top.dtype)
        out[:, :top_h, :] = top
        out[:, top_h:, :left_w] = left
        out[:, top_h:, left_w:] = right
        return self._resize_frame_chw(out, self.video_size)

    def _state_indices_for_dataset(self, ds_media) -> Tuple[int, ...]:
        dataset_root = Path(ds_media.root)
        cache_key = str(dataset_root)
        cached = self._state_indices_cache.get(cache_key)
        if cached is not None:
            return cached

        info_path = dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"AgiBot meta info not found: {info_path}")

        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)

        state_feature = info.get("features", {}).get("observation.state", {})
        field_descriptions = state_feature.get("field_descriptions", {})
        field = field_descriptions.get(self.STATE_FIELD_NAME)
        if field is None:
            raise KeyError(f"{self.STATE_FIELD_NAME!r} not found in {info_path}")

        indices = tuple(int(i) for i in field.get("indices", []))
        if len(indices) != 14:
            raise ValueError(
                f"Expected 14 indices for {self.STATE_FIELD_NAME!r} in {info_path}, got {len(indices)}: {indices}"
            )

        self._state_indices_cache[cache_key] = indices
        return indices

    def _select_state(self, raw_state: torch.Tensor, ds_media) -> torch.Tensor:
        state_indices = self._state_indices_for_dataset(ds_media)
        if raw_state.shape[-1] <= max(state_indices):
            raise ValueError(
                f"AgiBot state must have dims covering {state_indices}, got {tuple(raw_state.shape)}"
            )
        index = torch.tensor(state_indices, dtype=torch.long, device=raw_state.device)
        return raw_state.index_select(-1, index).float()

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

    def _select_action_sequence(self, action_values: Any, base_action_value: Any) -> torch.Tensor:
        raw_actions = self._to_action_tensor(action_values)
        base_action = self._to_action_tensor(base_action_value).flatten()

        if raw_actions.ndim == 1:
            raw_actions = raw_actions.unsqueeze(0)
        if raw_actions.shape[-1] < self.ACTION_RAW_MIN_DIM:
            raise ValueError(f"AgiBot action must have at least 40 dims, got {tuple(raw_actions.shape)}")
        if base_action.shape[-1] < self.ACTION_RAW_MIN_DIM:
            raise ValueError(f"AgiBot base action must have at least 40 dims, got {tuple(base_action.shape)}")

        left_xyz_delta = raw_actions[:, 2:5] - base_action[2:5]
        right_xyz_delta = raw_actions[:, 5:8] - base_action[5:8]
        left_rotvec_delta = self._relative_rotvec(raw_actions[:, 8:12], base_action[8:12])
        right_rotvec_delta = self._relative_rotvec(raw_actions[:, 12:16], base_action[12:16])
        left_gripper = raw_actions[:, 0:1]
        right_gripper = raw_actions[:, 1:2]
        head_waist_base = raw_actions[:, 30:40]

        return torch.cat(
            [
                left_xyz_delta,
                left_rotvec_delta,
                left_gripper,
                right_xyz_delta,
                right_rotvec_delta,
                right_gripper,
                head_waist_base,
            ],
            dim=-1,
        ).float()

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
                    emb = self._encode_and_cache_t5_embedding(ep_index, instr)
                    self._episode_embedding_cache[ep_index] = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                    cached = self._episode_embedding_cache[ep_index]
                else:
                    if self.task_mode == "single":
                        abs_path = Path(self.lerobot_dataset.root) / str(rel_path)
                    else:
                        abs_path = Path(self.lerobot_dataset._datasets[task_idx].root) / str(rel_path)
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

        with open(lang_action_path, "r", encoding="utf-8") as f:
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
