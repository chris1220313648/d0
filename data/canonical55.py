from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


CANONICAL55_FORMAT = "canonical55_v2"
CANONICAL55_DIM = 55
DATASET_TYPE_ALIASES = {
    "lerobot_agibot_lazy": "lerobot_agibot",
    "lerobot_robocoin_lazy": "lerobot_robocoin",
    "lerobot_interndata_lazy": "lerobot_interndata",
}

ARM_JOINT = slice(0, 14)
LEFT_ARM_JOINT = slice(0, 7)
RIGHT_ARM_JOINT = slice(7, 14)
LEFT_EEF = slice(14, 20)
RIGHT_EEF = slice(20, 26)
GRIPPER = slice(26, 28)
HAND_JOINT = slice(28, 40)
WAIST = slice(40, 44)
HEAD = slice(44, 46)
BASE = slice(46, 49)
RESERVED = slice(49, 55)


def _canonical_empty(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (*source.shape[:-1], CANONICAL55_DIM)
    return source.new_zeros(shape), torch.zeros(shape, dtype=torch.bool, device=source.device)


def _require_last_dim(source: torch.Tensor, expected: int, name: str) -> None:
    if source.shape[-1] != expected:
        raise ValueError(f"{name} expects last dim {expected}, got {tuple(source.shape)}")


def _assign(value: torch.Tensor, mask: torch.Tensor, dst: slice, src: torch.Tensor) -> None:
    value[..., dst] = src
    mask[..., dst] = True


def _assign_scalar(value: torch.Tensor, mask: torch.Tensor, dst: int, src: torch.Tensor) -> None:
    value[..., dst] = src
    mask[..., dst] = True


def map_robotwin_qpos(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map RobotWin/Aloha-style [left 6 + grip, right 6 + grip] qpos to 55D."""
    _require_last_dim(source, 14, "map_robotwin_qpos")
    value, mask = _canonical_empty(source)
    _assign(value, mask, slice(0, 6), source[..., 0:6])
    _assign(value, mask, slice(7, 13), source[..., 7:13])
    _assign_scalar(value, mask, 26, source[..., 6])
    _assign_scalar(value, mask, 27, source[..., 13])
    return value, mask


def map_dual_joint14(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map a 14D dual-arm joint vector without grippers to 55D arm slots."""
    _require_last_dim(source, 14, "map_dual_joint14")
    value, mask = _canonical_empty(source)
    _assign(value, mask, ARM_JOINT, source)
    return value, mask


def map_primary_joint7(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map a 7D primary/single-arm joint vector to the left/primary arm slot."""
    _require_last_dim(source, 7, "map_primary_joint7")
    value, mask = _canonical_empty(source)
    _assign(value, mask, LEFT_ARM_JOINT, source)
    return value, mask


def map_primary_eef_7d(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map [xyz, rpy, gripper] to primary EEF and gripper slots."""
    _require_last_dim(source, 7, "map_primary_eef_7d")
    value, mask = _canonical_empty(source)
    _assign(value, mask, LEFT_EEF, source[..., 0:6])
    _assign_scalar(value, mask, 26, source[..., 6])
    return value, mask


def map_dual_eef_14d(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map [left xyz/rpy/grip, right xyz/rpy/grip] to dual EEF and gripper slots."""
    _require_last_dim(source, 14, "map_dual_eef_14d")
    value, mask = _canonical_empty(source)
    _assign(value, mask, LEFT_EEF, source[..., 0:6])
    _assign_scalar(value, mask, 26, source[..., 6])
    _assign(value, mask, RIGHT_EEF, source[..., 7:13])
    _assign_scalar(value, mask, 27, source[..., 13])
    return value, mask


def map_agibot_24d(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map the compact AgiBot action [dual EEF, grips, head, waist, base] to 55D."""
    _require_last_dim(source, 24, "map_agibot_24d")
    value, mask = _canonical_empty(source)
    _assign(value, mask, LEFT_EEF, source[..., 0:6])
    _assign_scalar(value, mask, 26, source[..., 6])
    _assign(value, mask, RIGHT_EEF, source[..., 7:13])
    _assign_scalar(value, mask, 27, source[..., 13])
    _assign(value, mask, HEAD, source[..., 14:16])
    _assign(value, mask, WAIST, source[..., 17:21])
    _assign(value, mask, slice(46, 48), source[..., 22:24])
    return value, mask


def map_passthrough_55(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _require_last_dim(source, CANONICAL55_DIM, "map_passthrough_55")
    return source.clone(), torch.ones_like(source, dtype=torch.bool)


def map_zero_55(source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    value = source.new_zeros((*source.shape[:-1], CANONICAL55_DIM))
    return value, torch.zeros_like(value, dtype=torch.bool)


def _name_to_index(names: Sequence[str]) -> Dict[str, int]:
    return {str(name): idx for idx, name in enumerate(names)}


def _assign_named_scalar(
    value: torch.Tensor,
    mask: torch.Tensor,
    dst: int,
    source: torch.Tensor,
    name_to_idx: Dict[str, int],
    name: str,
) -> bool:
    idx = name_to_idx.get(name)
    if idx is None or idx >= source.shape[-1]:
        return False
    _assign_scalar(value, mask, dst, source[..., idx])
    return True


def _assign_named_vector(
    value: torch.Tensor,
    mask: torch.Tensor,
    dst: slice,
    source: torch.Tensor,
    name_to_idx: Dict[str, int],
    names: Sequence[str],
) -> bool:
    indices = [name_to_idx.get(name) for name in names]
    if any(idx is None or idx >= source.shape[-1] for idx in indices):
        return False
    gathered = torch.stack([source[..., int(idx)] for idx in indices], dim=-1)
    _assign(value, mask, dst, gathered)
    return True


def _quat_xyzw_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    q = quat.to(dtype=torch.float64)
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / norm
    qx, qy, qz, qw = q.unbind(dim=-1)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1).to(dtype=quat.dtype)


def _assign_robocoin_arm_joints(
    value: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    name_to_idx: Dict[str, int],
    side: str,
) -> None:
    base = 0 if side == "left" else 7
    for joint_idx in range(1, 8):
        name = f"{side}_arm_joint_{joint_idx}_rad"
        _assign_named_scalar(value, mask, base + joint_idx - 1, source, name_to_idx, name)


def _assign_robocoin_hand_joints(
    value: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    name_to_idx: Dict[str, int],
    side: str,
) -> None:
    base = 28 if side == "left" else 34
    for joint_idx in range(1, 7):
        name = f"{side}_hand_joint_{joint_idx}_rad"
        _assign_named_scalar(value, mask, base + joint_idx - 1, source, name_to_idx, name)


def _assign_robocoin_eef(
    value: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    name_to_idx: Dict[str, int],
    side: str,
) -> None:
    dst = LEFT_EEF if side == "left" else RIGHT_EEF
    dst_start = dst.start or 0
    for prefix in (f"{side}_eef", f"{side}_end"):
        pos_names = [f"{prefix}_pos_{axis}_m" for axis in ("x", "y", "z")]
        _assign_named_vector(value, mask, slice(dst_start, dst_start + 3), source, name_to_idx, pos_names)

        euler_names = [f"{prefix}_rot_euler_{axis}_rad" for axis in ("x", "y", "z")]
        if _assign_named_vector(value, mask, slice(dst_start + 3, dst_start + 6), source, name_to_idx, euler_names):
            return

        quat_names = [f"{prefix}_quat_{axis}" for axis in ("x", "y", "z", "w")]
        quat_indices = [name_to_idx.get(name) for name in quat_names]
        if any(idx is None or idx >= source.shape[-1] for idx in quat_indices):
            continue
        quat = torch.stack([source[..., int(idx)] for idx in quat_indices], dim=-1)
        _assign(value, mask, slice(dst_start + 3, dst_start + 6), _quat_xyzw_to_rpy(quat))


def _assign_robocoin_head_waist_base(
    value: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    name_to_idx: Dict[str, int],
) -> None:
    for dst, candidates in {
        44: ("head_joint_1_rad", "head_yaw_rad", "neck_joint_1_rad"),
        45: ("head_joint_2_rad", "head_pitch_rad", "neck_joint_2_rad"),
        40: ("waist_yaw_rad", "waist_joint_1_rad", "torso_joint_1_rad"),
        41: ("waist_pitch_rad", "waist_joint_2_rad", "torso_joint_2_rad"),
        42: ("waist_roll_rad", "waist_joint_3_rad", "torso_joint_3_rad"),
        43: ("waist_joint_4_rad", "torso_joint_4_rad"),
        46: ("base_vel_x_m_s", "base_linear_x_m_s", "base_x_vel_m_s"),
        47: ("base_vel_y_m_s", "base_linear_y_m_s", "base_y_vel_m_s"),
        48: ("base_angular_vel_rad_s", "base_yaw_vel_rad_s"),
    }.items():
        for name in candidates:
            if _assign_named_scalar(value, mask, dst, source, name_to_idx, name):
                break


def map_robocoin_named_vector(source: torch.Tensor, names: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map RoboCOIN vectors to canonical55 using per-dataset feature names."""
    if source.shape[-1] < len(names):
        raise ValueError(f"RoboCOIN names length {len(names)} exceeds source shape {tuple(source.shape)}")

    value, mask = _canonical_empty(source)
    name_to_idx = _name_to_index(names)

    for side, gripper_dst in (("left", 26), ("right", 27)):
        _assign_robocoin_arm_joints(value, mask, source, name_to_idx, side)
        _assign_robocoin_hand_joints(value, mask, source, name_to_idx, side)
        _assign_robocoin_eef(value, mask, source, name_to_idx, side)
        _assign_named_scalar(value, mask, gripper_dst, source, name_to_idx, f"{side}_gripper_open")

    _assign_robocoin_head_waist_base(value, mask, source, name_to_idx)
    return value, mask


def _canonicalize_state(
    source: torch.Tensor,
    dataset_type: str,
    names: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_type = DATASET_TYPE_ALIASES.get(dataset_type, dataset_type)
    if dataset_type in {"image_qa", "egoverse_trimodal", "latent_action"}:
        return map_zero_55(source)
    if source.shape[-1] == CANONICAL55_DIM:
        return map_passthrough_55(source)
    if dataset_type == "lerobot_robocoin" and names is not None:
        return map_robocoin_named_vector(source, names)
    if dataset_type in {"robotwin", "ac_one", "aloha_agilex_2"}:
        return map_robotwin_qpos(source)
    if dataset_type == "lerobot_agibot":
        return map_dual_joint14(source)
    if dataset_type in {"bridge", "fractal_bridge", "droid", "droid_bridge"}:
        if source.shape[-1] == 7:
            return map_primary_joint7(source)
        if source.shape[-1] == 14:
            return map_dual_joint14(source)
    raise ValueError(f"No canonical55 state mapper for dataset_type={dataset_type!r}, shape={tuple(source.shape)}")


def _canonicalize_action(
    source: torch.Tensor,
    dataset_type: str,
    names: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_type = DATASET_TYPE_ALIASES.get(dataset_type, dataset_type)
    if dataset_type in {"image_qa", "egoverse_trimodal", "latent_action"}:
        return map_zero_55(source)
    if source.shape[-1] == CANONICAL55_DIM:
        return map_passthrough_55(source)
    if dataset_type == "lerobot_robocoin" and names is not None:
        return map_robocoin_named_vector(source, names)
    if dataset_type in {"robotwin", "ac_one", "aloha_agilex_2"}:
        return map_robotwin_qpos(source)
    if dataset_type == "lerobot_agibot":
        return map_agibot_24d(source)
    if dataset_type in {"bridge", "fractal_bridge", "droid", "droid_bridge"}:
        if source.shape[-1] == 7:
            return map_primary_eef_7d(source)
        if source.shape[-1] == 14:
            return map_dual_eef_14d(source)
    raise ValueError(f"No canonical55 action mapper for dataset_type={dataset_type!r}, shape={tuple(source.shape)}")


class Canonical55Dataset(Dataset):
    """Dataset wrapper that maps sample state/action tensors into canonical55_v2."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_type: str,
        normalization: Optional[Mapping[str, Any]] = None,
    ):
        self.dataset = dataset
        self.dataset_type = str(dataset_type)
        normalization = dict(normalization or {})
        self.normalize_state = bool(normalization.get("enabled", False) and normalization.get("normalize_state", False))
        self.normalize_action = bool(normalization.get("enabled", False) and normalization.get("normalize_action", False))
        self._state_stats = None
        self._action_stats = None
        if self.normalize_state or self.normalize_action:
            from data.utils.norm import load_masked_normalization_stats

            stats_path = normalization.get("stats_path")
            stats_key = normalization.get("stats_key")
            if not stats_path or not stats_key:
                raise ValueError("canonical normalization requires stats_path and stats_key")
            if self.normalize_state:
                self._state_stats = load_masked_normalization_stats(
                    str(stats_path), str(stats_key), "state", expected_dim=CANONICAL55_DIM
                )
            if self.normalize_action:
                self._action_stats = load_masked_normalization_stats(
                    str(stats_path), str(stats_key), "action", expected_dim=CANONICAL55_DIM
                )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.dataset[idx]
        if sample is None:
            return None

        sample = dict(sample)
        sample["canonical_format"] = CANONICAL55_FORMAT

        if sample.get("initial_state") is not None:
            state, state_mask = _canonicalize_state(
                sample["initial_state"].float(),
                self.dataset_type,
                names=sample.get("state_names"),
            )
            existing_state_mask = sample.get("state_mask")
            if existing_state_mask is not None and existing_state_mask.shape == state_mask.shape:
                state_mask = state_mask & existing_state_mask.to(device=state_mask.device, dtype=torch.bool)
            if self._state_stats is not None:
                from data.utils.norm import normalize_masked

                state = normalize_masked(state, state_mask, *self._state_stats)
            sample["initial_state"] = state
            sample["state_mask"] = state_mask

        if sample.get("action_sequence") is not None:
            action, action_mask = _canonicalize_action(
                sample["action_sequence"].float(),
                self.dataset_type,
                names=sample.get("action_names"),
            )
            existing_mask = sample.get("action_mask")
            if existing_mask is not None and existing_mask.shape == action_mask.shape:
                action_mask = action_mask & existing_mask.to(device=action_mask.device, dtype=torch.bool)
            if self._action_stats is not None:
                from data.utils.norm import normalize_masked

                action = normalize_masked(action, action_mask, *self._action_stats)
            sample["action_sequence"] = action
            sample["action_mask"] = action_mask

        if sample.get("history_action_sequence") is not None:
            history, history_mask = _canonicalize_state(
                sample["history_action_sequence"].float(),
                self.dataset_type,
                names=sample.get("action_names"),
            )
            sample["history_action_sequence"] = history
            sample["history_action_mask"] = history_mask

        return sample


def should_use_canonical55(canonical_format: Optional[str]) -> bool:
    return canonical_format == CANONICAL55_FORMAT
