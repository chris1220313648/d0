"""InternData-A1 LeRobot loader for canonical55 LAP training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from lerobot.datasets.video_utils import decode_video_frames

from data.canonical55 import (
    ARM_JOINT,
    BASE,
    CANONICAL55_DIM,
    GRIPPER,
    HEAD,
    LEFT_ARM_JOINT,
    LEFT_EEF,
    RIGHT_ARM_JOINT,
    RIGHT_EEF,
    WAIST,
    map_robotwin_qpos,
)
from utils.vlm_utils import append_setup_control_suffix, preprocess_vlm_messages_lap

from .lerobot_dataset import LeRobotMotusDataset, preprocess_vlm_messages, tensor_to_pil


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, (list, tuple)) and any(isinstance(item, torch.Tensor) for item in value):
        return torch.stack(
            [
                item.float() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32)
                for item in value
            ]
        )
    return torch.tensor(value, dtype=torch.float32)


def _as_2d_tensor(value: Any) -> torch.Tensor:
    tensor = _to_float_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    return tensor.float()


def _as_feature_tensor(value: Any) -> torch.Tensor:
    tensor = _to_float_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    return tensor.float()


def _first_tensor(columns: Mapping[str, Any]) -> torch.Tensor:
    for value in columns.values():
        if value is None:
            continue
        try:
            return _as_feature_tensor(value)
        except (TypeError, ValueError):
            continue
    return torch.zeros((1, 1), dtype=torch.float32)


def _candidate_numeric_keys(prefix: str) -> List[str]:
    return [
        "action" if prefix == "actions" else "observation.state",
        f"{prefix}.joint.position",
        f"{prefix}.tcp_to_robot_pose",
        f"{prefix}.ee_to_robot_pose",
        f"{prefix}.tcp_to_armbase_pose",
        f"{prefix}.ee_to_armbase_pose",
        f"{prefix}.gripper.pose",
        f"{prefix}.gripper.position",
        f"{prefix}.gripper.openness",
        f"{prefix}.effector.position",
        f"{prefix}.waist.position",
        f"{prefix}.head.position",
    ]


def _empty_like_columns(columns: Mapping[str, Any], prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
    refs = {key: columns[key] for key in _candidate_numeric_keys(prefix) if key in columns}
    ref = _first_tensor(refs if refs else columns)
    shape = (int(ref.shape[0]), CANONICAL55_DIM)
    return ref.new_zeros(shape), torch.zeros(shape, dtype=torch.bool, device=ref.device)


def _empty_14_like_columns(columns: Mapping[str, Any], keys: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    refs = {key: columns[key] for key in keys if key in columns}
    ref = _first_tensor(refs if refs else columns)
    shape = (int(ref.shape[0]), 14)
    return ref.new_zeros(shape), torch.zeros(shape, dtype=torch.bool, device=ref.device)


def _assign(value: torch.Tensor, mask: torch.Tensor, dst: slice, src: torch.Tensor) -> None:
    width = min(dst.stop - dst.start, src.shape[-1])
    if width <= 0:
        return
    value[..., dst.start : dst.start + width] = src[..., :width]
    mask[..., dst.start : dst.start + width] = True


def _assign_scalar(value: torch.Tensor, mask: torch.Tensor, dst: int, src: torch.Tensor) -> None:
    scalar = _as_2d_tensor(src)[..., 0]
    value[..., dst] = scalar
    mask[..., dst] = True


def _quat_wxyz_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    q = quat.to(dtype=torch.float64)
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / norm
    qw, qx, qy, qz = q.unbind(dim=-1)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1).to(dtype=quat.dtype)


def _pose_to_xyzrpy(pose: torch.Tensor) -> torch.Tensor:
    pose = _as_feature_tensor(pose)
    if pose.shape[-1] >= 7:
        return torch.cat([pose[..., :3], _quat_wxyz_to_rpy(pose[..., 3:7])], dim=-1)
    if pose.shape[-1] >= 6:
        return pose[..., :6]
    raise ValueError(f"InternData pose needs at least 6 dims, got {tuple(pose.shape)}")


def _require_column(columns: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in columns:
        raise KeyError(f"InternData column {key!r} is required")
    return _as_feature_tensor(columns[key])


def _first_available_pose(columns: Mapping[str, Any], names: Tuple[str, ...]) -> torch.Tensor:
    for name in names:
        if name in columns:
            return _pose_to_xyzrpy(columns[name])
    raise KeyError(f"InternData requires one of {names}")


def _first_available_tensor(columns: Mapping[str, Any], names: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for name in names:
        if name in columns:
            return _as_feature_tensor(columns[name])
    return None


def _first_available_2d_tensor(columns: Mapping[str, Any], names: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for name in names:
        if name in columns:
            return _as_2d_tensor(columns[name])
    return None


def _assign_dual_joint14(
    value: torch.Tensor,
    mask: torch.Tensor,
    left_joint: torch.Tensor,
    right_joint: torch.Tensor,
    left_gripper: torch.Tensor,
    right_gripper: torch.Tensor,
) -> None:
    if left_joint.shape[-1] < 6 or right_joint.shape[-1] < 6:
        raise ValueError(
            f"InternData joint action requires left/right 6D joints, got {left_joint.shape[-1]} and {right_joint.shape[-1]}"
        )
    value[..., 0:6] = left_joint[..., 0:6]
    value[..., 6] = _as_2d_tensor(left_gripper)[..., 0]
    value[..., 7:13] = right_joint[..., 0:6]
    value[..., 13] = _as_2d_tensor(right_gripper)[..., 0]
    mask[...] = True


def _assign_single_joint14(
    value: torch.Tensor,
    mask: torch.Tensor,
    joint: torch.Tensor,
) -> None:
    if joint.shape[-1] < 7:
        raise ValueError(f"InternData single-arm joint state requires at least 7D joints, got {joint.shape[-1]}")
    value[..., 0:7] = joint[..., 0:7]
    mask[..., 0:7] = True


def _assign_single_eef14(
    value: torch.Tensor,
    mask: torch.Tensor,
    pose: torch.Tensor,
    gripper: torch.Tensor,
) -> None:
    value[..., 0:6] = _pose_to_xyzrpy(pose)[..., 0:6]
    value[..., 6] = _as_2d_tensor(gripper)[..., 0]
    mask[..., 0:7] = True


def infer_interndata_action_signal(columns: Mapping[str, Any], default_signal: str = "epos") -> str:
    left_eef_names = (
        "actions.left_tcp_to_robot_pose",
        "actions.left_ee_to_robot_pose",
        "actions.left_tcp_to_left_armbase_pose",
        "actions.left_ee_to_left_armbase_pose",
    )
    right_eef_names = (
        "actions.right_tcp_to_robot_pose",
        "actions.right_ee_to_robot_pose",
        "actions.right_tcp_to_right_armbase_pose",
        "actions.right_ee_to_right_armbase_pose",
    )
    if any(name in columns for name in left_eef_names) and any(name in columns for name in right_eef_names):
        return "epos"
    if any(
        name in columns
        for name in (
            "actions.tcp_to_robot_pose",
            "actions.ee_to_robot_pose",
            "actions.tcp_to_armbase_pose",
            "actions.ee_to_armbase_pose",
        )
    ):
        return "epos"
    if any(
        name in columns
        for name in (
            "action",
            "actions.joint.position",
            "actions.left_joint.position",
            "actions.right_joint.position",
        )
    ):
        return "qpos"
    return str(default_signal)


def _assign_single_eef_from_candidates(
    value: torch.Tensor,
    mask: torch.Tensor,
    columns: Mapping[str, Any],
    prefix: str,
) -> None:
    for name in (
        f"{prefix}.tcp_to_robot_pose",
        f"{prefix}.ee_to_robot_pose",
        f"{prefix}.tcp_to_armbase_pose",
        f"{prefix}.ee_to_armbase_pose",
        f"{prefix}.gripper.pose",
    ):
        if name in columns:
            _assign(value, mask, LEFT_EEF, _pose_to_xyzrpy(columns[name]))
            return


def _assign_dual_eef_from_candidates(
    value: torch.Tensor,
    mask: torch.Tensor,
    columns: Mapping[str, Any],
    prefix: str,
) -> None:
    for side, dst in (("left", LEFT_EEF), ("right", RIGHT_EEF)):
        for name in (
            f"{prefix}.{side}_tcp_to_robot_pose",
            f"{prefix}.{side}_ee_to_robot_pose",
            f"{prefix}.{side}_tcp_to_{side}_armbase_pose",
            f"{prefix}.{side}_ee_to_{side}_armbase_pose",
        ):
            if name in columns:
                _assign(value, mask, dst, _pose_to_xyzrpy(columns[name]))
                break


def build_interndata_canonical55(columns: Mapping[str, Any], prefix: str = "actions") -> Tuple[torch.Tensor, torch.Tensor]:
    """Map InternData-A1 action/state columns into canonical55 slots."""

    value, mask = _empty_like_columns(columns, prefix)

    vector_key = "action" if prefix == "actions" else "observation.state"
    if vector_key in columns:
        vector = _as_feature_tensor(columns[vector_key])
        if vector.shape[-1] == 14:
            mapped, mapped_mask = map_robotwin_qpos(vector)
            return mapped, mapped_mask
        if vector.shape[-1] == 7:
            _assign(value, mask, LEFT_ARM_JOINT, vector)
        else:
            _assign(value, mask, ARM_JOINT, vector)

    joint_key = f"{prefix}.joint.position"
    if joint_key in columns:
        joint = _as_feature_tensor(columns[joint_key])
        if joint.shape[-1] <= 7:
            _assign(value, mask, LEFT_ARM_JOINT, joint)
        else:
            _assign(value, mask, ARM_JOINT, joint)

    _assign_single_eef_from_candidates(value, mask, columns, prefix)
    _assign_dual_eef_from_candidates(value, mask, columns, prefix)

    for key, dst in (
        (f"{prefix}.gripper.position", 26),
        (f"{prefix}.gripper.openness", 26),
        (f"{prefix}.left_gripper.position", 26),
        (f"{prefix}.left_gripper.openness", 26),
        (f"{prefix}.right_gripper.position", 27),
        (f"{prefix}.right_gripper.openness", 27),
    ):
        if key in columns:
            _assign_scalar(value, mask, dst, columns[key])

    effector_key = f"{prefix}.effector.position"
    if effector_key in columns:
        effector = _as_feature_tensor(columns[effector_key])
        _assign(value, mask, GRIPPER, effector)

    for key, dst in ((f"{prefix}.waist.position", WAIST), (f"{prefix}.head.position", HEAD)):
        if key in columns:
            _assign(value, mask, dst, _as_feature_tensor(columns[key]))

    for key in (f"{prefix}.robot.velocity", f"{prefix}.base.velocity"):
        if key in columns:
            _assign(value, mask, BASE, _as_feature_tensor(columns[key]))

    return value, mask


def build_interndata_dual_eef14_joint14(
    columns: Mapping[str, Any],
    prefix: str = "actions",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map InternData columns into 14D [dual EEF+gripper] action or [dual joint+gripper] state."""

    if prefix == "actions":
        keys = [
            "action",
            "actions.joint.position",
            "actions.effector.position",
            "actions.left_joint.position",
            "actions.right_joint.position",
            "actions.left_tcp_to_robot_pose",
            "actions.left_ee_to_robot_pose",
            "actions.left_tcp_to_left_armbase_pose",
            "actions.left_ee_to_left_armbase_pose",
            "actions.right_tcp_to_robot_pose",
            "actions.right_ee_to_robot_pose",
            "actions.right_tcp_to_right_armbase_pose",
            "actions.right_ee_to_right_armbase_pose",
            "actions.left_gripper.position",
            "actions.left_gripper.openness",
            "actions.right_gripper.position",
            "actions.right_gripper.openness",
            "actions.tcp_to_robot_pose",
            "actions.ee_to_robot_pose",
            "actions.tcp_to_armbase_pose",
            "actions.ee_to_armbase_pose",
            "actions.gripper.position",
            "actions.gripper.openness",
        ]
        value, mask = _empty_14_like_columns(columns, keys)
        left_pose = _first_available_tensor(
            columns,
            (
                "actions.left_tcp_to_robot_pose",
                "actions.left_ee_to_robot_pose",
                "actions.left_tcp_to_left_armbase_pose",
                "actions.left_ee_to_left_armbase_pose",
            ),
        )
        right_pose = _first_available_tensor(
            columns,
            (
                "actions.right_tcp_to_robot_pose",
                "actions.right_ee_to_robot_pose",
                "actions.right_tcp_to_right_armbase_pose",
                "actions.right_ee_to_right_armbase_pose",
            ),
        )
        if left_pose is not None and right_pose is not None:
            value[..., 0:6] = _pose_to_xyzrpy(left_pose)[..., 0:6]
            value[..., 7:13] = _pose_to_xyzrpy(right_pose)[..., 0:6]
        else:
            vector = _first_available_tensor(columns, ("action", "actions.joint.position"))
            effector = _first_available_tensor(columns, ("actions.effector.position",))
            if vector is not None and vector.shape[-1] >= 14:
                if effector is None:
                    value[...] = vector[..., 0:14]
                else:
                    value[..., 0:6] = vector[..., 0:6]
                    value[..., 6] = effector[..., 0]
                    value[..., 7:13] = vector[..., 7:13]
                    value[..., 13] = effector[..., 1]
                mask[...] = True
                return value.float(), mask

            single_pose = _first_available_tensor(
                columns,
                (
                    "actions.tcp_to_robot_pose",
                    "actions.ee_to_robot_pose",
                    "actions.tcp_to_armbase_pose",
                    "actions.ee_to_armbase_pose",
                ),
            )
            single_gripper = _first_available_2d_tensor(
                columns,
                ("actions.gripper.position", "actions.gripper.openness"),
            )
            if single_pose is not None and single_gripper is not None:
                _assign_single_eef14(value, mask, single_pose, single_gripper)
                return value.float(), mask

            left_joint = _first_available_tensor(columns, ("actions.left_joint.position",))
            right_joint = _first_available_tensor(columns, ("actions.right_joint.position",))
            left_gripper = _first_available_2d_tensor(
                columns,
                ("actions.left_gripper.position", "actions.left_gripper.openness"),
            )
            right_gripper = _first_available_2d_tensor(
                columns,
                ("actions.right_gripper.position", "actions.right_gripper.openness"),
            )
            single_joint = _first_available_tensor(columns, ("actions.joint.position",))
            if single_joint is not None and single_joint.shape[-1] < 14:
                _assign_single_joint14(value, mask, single_joint)
                return value.float(), mask
            if left_joint is None or right_joint is None or left_gripper is None or right_gripper is None:
                raise KeyError("InternData action requires EEF pose fields or joint/gripper fallback fields")
            _assign_dual_joint14(value, mask, left_joint, right_joint, left_gripper, right_gripper)
            return value.float(), mask
        for dst, candidates in (
            (6, ("actions.left_gripper.position", "actions.left_gripper.openness")),
            (13, ("actions.right_gripper.position", "actions.right_gripper.openness")),
        ):
            for name in candidates:
                if name in columns:
                    value[..., dst] = _as_2d_tensor(columns[name])[..., 0]
                    break
            else:
                raise KeyError(f"InternData requires one of {candidates}")
        mask[...] = True
        return value.float(), mask

    if prefix == "states":
        keys = [
            "observation.state",
            "states.joint.position",
            "states.effector.position",
            "states.left_joint.position",
            "states.right_joint.position",
            "states.left_gripper.position",
            "states.left_gripper.openness",
            "states.right_gripper.position",
            "states.right_gripper.openness",
        ]
        value, mask = _empty_14_like_columns(columns, keys)
        left_joint = _first_available_tensor(columns, ("states.left_joint.position",))
        right_joint = _first_available_tensor(columns, ("states.right_joint.position",))
        left_gripper = _first_available_2d_tensor(
            columns,
            ("states.left_gripper.position", "states.left_gripper.openness"),
        )
        right_gripper = _first_available_2d_tensor(
            columns,
            ("states.right_gripper.position", "states.right_gripper.openness"),
        )
        if left_joint is not None and right_joint is not None and left_gripper is not None and right_gripper is not None:
            _assign_dual_joint14(value, mask, left_joint, right_joint, left_gripper, right_gripper)
            return value.float(), mask

        single_joint = _first_available_tensor(columns, ("states.joint.position",))
        if single_joint is not None and single_joint.shape[-1] < 14:
            _assign_single_joint14(value, mask, single_joint)
            return value.float(), mask

        vector = _first_available_tensor(columns, ("observation.state", "states.joint.position"))
        effector = _first_available_tensor(columns, ("states.effector.position",))
        if vector is None or vector.shape[-1] < 14:
            raise KeyError(
                "InternData state requires split joint/gripper fields or a 14D observation.state/states.joint.position"
            )

        if effector is None:
            value[...] = vector[..., 0:14]
        else:
            value[..., 0:6] = vector[..., 0:6]
            value[..., 6] = effector[..., 0]
            value[..., 7:13] = vector[..., 7:13]
            value[..., 13] = effector[..., 1]
        mask[...] = True
        return value.float(), mask

    raise ValueError(f"Unsupported InternData 14D prefix: {prefix}")


class LeRobotInternDataDataset(LeRobotMotusDataset):
    """InternData-A1 wrapper that emits canonical55 tensors and LAP prompts."""

    def __init__(
        self,
        *args,
        language_action_dir_name: str = "language_action",
        use_language_action: bool = True,
        normalize_actions: bool = False,
        output_format: str = "canonical55",
        enable_setup_control_suffix: bool = False,
        setup_text: str = "InternData robot with gripper",
        action_signal: str = "epos",
        **kwargs,
    ):
        kwargs["use_multi_lerobot_dataset"] = False
        self.output_format = str(output_format)
        if self.output_format not in {"canonical55", "dual_eef14_joint14"}:
            raise ValueError(f"Unsupported InternData output_format: {self.output_format}")
        if normalize_actions:
            raise ValueError("LeRobotInternDataDataset emits canonical55 tensors; external normalization is not supported.")
        super().__init__(*args, **kwargs)
        self.use_language_action = bool(use_language_action)
        self.enable_setup_control_suffix = bool(enable_setup_control_suffix)
        self.setup_text = str(setup_text)
        self.action_signal = str(action_signal)
        self.language_action_dir_name = str(language_action_dir_name).strip() or "language_action"
        self._language_action_cache: Dict[Tuple[int, int], List[str]] = {}

    def __getitem__(self, idx):
        if not self.lerobot_dataset:
            return None

        episode_idx = random.randint(0, self.lerobot_dataset.num_episodes - 1)
        task_idx = 0
        if self.task_mode == "multi":
            task_idx = self.episode_id_to_task_idx[episode_idx]
            if task_idx > 0:
                episode_idx -= self.episode_num_accumulated[task_idx - 1]
            from_idx_t = self.lerobot_dataset._datasets[task_idx].episode_data_index["from"][episode_idx]
            to_idx_t = self.lerobot_dataset._datasets[task_idx].episode_data_index["to"][episode_idx]
        else:
            from_idx_t = self.lerobot_dataset.episode_data_index["from"][episode_idx]
            to_idx_t = self.lerobot_dataset.episode_data_index["to"][episode_idx]

        from_idx = int(from_idx_t.item()) if hasattr(from_idx_t, "item") else int(from_idx_t)
        to_idx = int(to_idx_t.item()) if hasattr(to_idx_t, "item") else int(to_idx_t)
        condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(int(to_idx - from_idx))

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
        first_frame, video_frames_sampled = self._load_visual_frames(
            ds_media,
            hf_dataset,
            item_cond,
            [local_cond_idx] + local_video_indices,
        )

        state_columns = {key: item_cond[key] for key in item_cond.keys()}
        action_batch = hf_dataset[local_action_indices]
        if self.output_format == "dual_eef14_joint14":
            initial_state, state_mask = build_interndata_dual_eef14_joint14(state_columns, prefix="states")
            initial_state = initial_state[0]
            state_mask = state_mask[0]
            action_sequence, action_mask = build_interndata_dual_eef14_joint14(action_batch, prefix="actions")
        else:
            initial_state, state_mask = build_interndata_canonical55(state_columns, prefix="states")
            if not state_mask.any():
                initial_state, state_mask = build_interndata_canonical55(state_columns, prefix="actions")
            initial_state = initial_state[0]
            state_mask = state_mask[0]

            action_sequence, action_mask = build_interndata_canonical55(action_batch, prefix="actions")
            if not action_mask.any():
                raise KeyError("No usable InternData action columns found")

        language_embedding = self._load_language_embedding(item_cond, task_idx)
        language_action = None
        if self.use_language_action:
            ep_raw = item_cond.get("episode_index", None)
            if ep_raw is None:
                raise KeyError("episode_index not found in InternData item")
            ep_index = int(ep_raw.item()) if hasattr(ep_raw, "item") else int(ep_raw)
            language_action = self._load_language_action_from_file(ep_index, task_idx, condition_frame_idx)

        vlm_tokens = None
        if self.vlm_processor:
            text_instr = self._instruction_from_item(item_cond)
            text_instr = append_setup_control_suffix(
                text_instruction=text_instr,
                enable_setup_control_suffix=self.enable_setup_control_suffix,
                setup_text=self.setup_text,
                action_signal=infer_interndata_action_signal(action_batch, self.action_signal),
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
            "state_mask": state_mask,
            "action_sequence": action_sequence,
            "action_mask": action_mask,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_tokens,
        }

    def _load_visual_frames(self, ds_media, hf_dataset, item_cond: Mapping[str, Any], indices: List[int]):
        ts_vals = hf_dataset[indices]["timestamp"]
        if isinstance(ts_vals, torch.Tensor):
            timestamps = ts_vals.flatten().tolist()
        elif isinstance(ts_vals, (list, tuple)) and len(ts_vals) > 0 and isinstance(ts_vals[0], torch.Tensor):
            timestamps = torch.stack(ts_vals).flatten().tolist()
        else:
            timestamps = [float(x) for x in list(ts_vals)]

        ep_raw = item_cond.get("episode_index", None)
        if ep_raw is None:
            raise KeyError("episode_index not found in InternData item")
        episode_index = int(ep_raw.item()) if hasattr(ep_raw, "item") else int(ep_raw)

        visual_key = self._select_visual_key(ds_media, hf_dataset)
        video_path = Path(ds_media.root) / ds_media.meta.get_video_file_path(episode_index, visual_key)
        frames = decode_video_frames(video_path, timestamps, ds_media.tolerance_s, ds_media.video_backend).squeeze(0)
        first_frame = self._resize_frame_chw(frames[0].float(), self.video_size)
        video_frames = torch.stack(
            [self._resize_frame_chw(frames[i].float(), self.video_size) for i in range(1, frames.shape[0])],
            dim=0,
        )
        return first_frame, video_frames

    def _select_visual_key(self, ds_media, hf_dataset) -> str:
        for key in (
            "observation.images.cam_concatenated",
            "images.rgb.head",
            "images.rgb.hand",
            "observation.images.cam_high",
            "observation.images.main",
            "observation.image",
            "image",
        ):
            if key in getattr(ds_media.meta, "video_keys", []) or key in hf_dataset.column_names:
                return key
        video_keys = sorted(getattr(ds_media.meta, "video_keys", []))
        if video_keys:
            return video_keys[0]
        raise ValueError(f"No InternData video key found for {ds_media.root}")

    def _load_language_embedding(self, item_cond: Mapping[str, Any], task_idx: int) -> torch.Tensor:
        all_embeddings = item_cond.get("language_embedding", None)
        if all_embeddings is None:
            all_embeddings = item_cond.get("observation.feature.language_embedding", None)

        if all_embeddings is None:
            ep_raw = item_cond.get("episode_index", None)
            if ep_raw is None:
                raise KeyError("episode_index not found in InternData item")
            ep_index = int(ep_raw.item()) if hasattr(ep_raw, "item") else int(ep_raw)

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
                raise KeyError("t5_embedding_path not found in InternData meta/episodes.jsonl")
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

    def _load_language_action_from_file(self, episode_index: int, task_idx: int, condition_frame_idx: int) -> str:
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
        path = Path(str(rel_path))
        if not path.is_absolute():
            path = dataset_root / path
        if not path.exists():
            raise FileNotFoundError(f"InternData language_action file not found: {path}")

        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"No language action lines in {path}")
        return lines[min(condition_frame_idx, len(lines) - 1)]

    @staticmethod
    def _instruction_from_item(item: Mapping[str, Any]) -> str:
        for key in ("language_instruction", "task"):
            value = item.get(key, None)
            if isinstance(value, str) and value.strip():
                return value
        tasks = item.get("tasks", None)
        if isinstance(tasks, (list, tuple)) and tasks:
            return str(tasks[0])
        return ""
