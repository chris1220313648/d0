"""OLA LeRobot v3 dataset loader for Motus LAP post-training.

This loader intentionally does not depend on ``LeRobotDataset`` because the
cluster's installed LeRobot release predates the aggregated v3 data/video
layout used by OLA. It reads v3 metadata and parquet shards directly while
using LeRobot's video decoder.

Model contract
--------------
* state: measured dual-arm joints, 14D
  [left_joint_1..6, left_gripper, right_joint_1..6, right_gripper]
* raw current EEF/action: absolute dual-arm EEF pose, 16D
  [left_xyz, left_quat_xyzw, left_gripper,
   right_xyz, right_quat_xyzw, right_gripper]
* model action: base-frame relative EEF, 14D
  [left_dxyz, left_drpy_xyz, left_gripper,
   right_dxyz, right_drpy_xyz, right_gripper]
"""

from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lerobot.datasets.video_utils import decode_video_frames

from utils.vlm_utils import preprocess_vlm_messages_lap

from .lerobot_dataset import preprocess_vlm_messages, tensor_to_pil

try:
    from transformers import AutoProcessor
except ImportError:  # pragma: no cover - training environment provides transformers
    AutoProcessor = None


logger = logging.getLogger(__name__)


DEFAULT_CAMERA_KEYS = (
    "observation.images.right_front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _normalize_quaternion_xyzw(quat: torch.Tensor) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must end in 4 values, got {tuple(quat.shape)}")
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError("Zero-norm quaternion found in OLA EEF pose")
    return quat / norm


def _quat_inverse_xyzw(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quaternion_xyzw(quat)
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def _quat_multiply_xyzw(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product for xyzw quaternions."""
    lx, ly, lz, lw = lhs.unbind(dim=-1)
    rx, ry, rz, rw = rhs.unbind(dim=-1)
    return torch.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dim=-1,
    )


def quat_xyzw_to_rpy_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternion to xyz Euler angles in radians."""
    q = _normalize_quaternion_xyzw(quat).to(dtype=torch.float64)
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


def rpy_xyz_to_quat_xyzw(rpy: torch.Tensor) -> torch.Tensor:
    """Convert xyz Euler angles in radians to normalized xyzw quaternion."""
    angles = _as_float_tensor(rpy)
    if angles.shape[-1] != 3:
        raise ValueError(f"RPY tensor must end in 3 values, got {tuple(angles.shape)}")
    roll, pitch, yaw = (angles / 2.0).unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    quat = torch.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dim=-1,
    )
    return _normalize_quaternion_xyzw(quat)


def absolute_eef_to_relative_rpy(
    measured_eef: torch.Tensor,
    commanded_eef: torch.Tensor,
    *,
    gripper_closed_value: float = 0.0,
    gripper_open_value: float = 100.0,
) -> torch.Tensor:
    """Convert absolute dual-arm EEF commands into 14D base-frame deltas.

    Rotation follows ``R_delta = R_command * inverse(R_measured)`` and is then
    represented as xyz RPY radians. Gripper remains an absolute command and is
    linearly normalized to [0, 1].
    """
    measured = _as_float_tensor(measured_eef)
    commanded = _as_float_tensor(commanded_eef)
    if measured.shape != commanded.shape or measured.shape[-1] != 16:
        raise ValueError(
            "OLA measured/action EEF tensors must have identical [...,16] shape, "
            f"got measured={tuple(measured.shape)}, action={tuple(commanded.shape)}"
        )
    grip_range = float(gripper_open_value) - float(gripper_closed_value)
    if abs(grip_range) < 1e-8:
        raise ValueError("gripper_open_value and gripper_closed_value must differ")

    arms: List[torch.Tensor] = []
    for start in (0, 8):
        measured_xyz = measured[..., start : start + 3]
        command_xyz = commanded[..., start : start + 3]
        measured_quat = _normalize_quaternion_xyzw(measured[..., start + 3 : start + 7])
        command_quat = _normalize_quaternion_xyzw(commanded[..., start + 3 : start + 7])
        relative_quat = _quat_multiply_xyzw(command_quat, _quat_inverse_xyzw(measured_quat))
        relative_rpy = quat_xyzw_to_rpy_xyz(relative_quat)
        gripper = (
            (commanded[..., start + 7 : start + 8] - float(gripper_closed_value)) / grip_range
        ).clamp(0.0, 1.0)
        arms.append(torch.cat([command_xyz - measured_xyz, relative_rpy, gripper], dim=-1))
    return torch.cat(arms, dim=-1).float()


def relative_rpy_to_absolute_eef(
    measured_eef: torch.Tensor,
    relative_action: torch.Tensor,
    *,
    gripper_closed_value: float = 0.0,
    gripper_open_value: float = 100.0,
) -> torch.Tensor:
    """Inverse of :func:`absolute_eef_to_relative_rpy` for deployment.

    ``relative_action`` must already be denormalized from model [0,1] space
    into physical delta meters/radians plus normalized gripper [0,1].
    """
    measured = _as_float_tensor(measured_eef)
    relative = _as_float_tensor(relative_action)
    if measured.shape[-1] != 16 or relative.shape[-1] != 14:
        raise ValueError(
            f"Expected measured [...,16] and relative [...,14], got {measured.shape}, {relative.shape}"
        )
    if measured.shape[:-1] != relative.shape[:-1]:
        raise ValueError("Measured EEF and relative action leading dimensions must match")
    grip_range = float(gripper_open_value) - float(gripper_closed_value)
    if abs(grip_range) < 1e-8:
        raise ValueError("gripper_open_value and gripper_closed_value must differ")

    arms: List[torch.Tensor] = []
    for pose_start, action_start in ((0, 0), (8, 7)):
        measured_xyz = measured[..., pose_start : pose_start + 3]
        measured_quat = _normalize_quaternion_xyzw(measured[..., pose_start + 3 : pose_start + 7])
        delta_xyz = relative[..., action_start : action_start + 3]
        delta_quat = rpy_xyz_to_quat_xyzw(relative[..., action_start + 3 : action_start + 6])
        target_quat = _normalize_quaternion_xyzw(_quat_multiply_xyzw(delta_quat, measured_quat))
        gripper = (
            relative[..., action_start + 6 : action_start + 7].clamp(0.0, 1.0) * grip_range
            + float(gripper_closed_value)
        )
        arms.append(torch.cat([measured_xyz + delta_xyz, target_quat, gripper], dim=-1))
    return torch.cat(arms, dim=-1).float()


def normalize_joint_grippers(
    state: torch.Tensor,
    *,
    gripper_closed_value: float,
    gripper_open_value: float,
) -> torch.Tensor:
    state = _as_float_tensor(state).clone()
    if state.shape[-1] != 14:
        raise ValueError(f"OLA measured joint state must be 14D, got {tuple(state.shape)}")
    grip_range = float(gripper_open_value) - float(gripper_closed_value)
    if abs(grip_range) < 1e-8:
        raise ValueError("gripper_open_value and gripper_closed_value must differ")
    for index in (6, 13):
        state[..., index] = ((state[..., index] - float(gripper_closed_value)) / grip_range).clamp(0.0, 1.0)
    return state


def _linear_normalize(values: torch.Tensor, lower: np.ndarray, upper: np.ndarray) -> torch.Tensor:
    lower_t = torch.as_tensor(lower, dtype=values.dtype, device=values.device)
    upper_t = torch.as_tensor(upper, dtype=values.dtype, device=values.device)
    width = torch.where((upper_t - lower_t).abs() < 1e-8, torch.ones_like(upper_t), upper_t - lower_t)
    return ((values - lower_t) / width).clamp(0.0, 1.0)


@dataclass(frozen=True)
class OLAEpisode:
    episode_index: int
    tasks: Tuple[str, ...]
    length: int
    data_chunk_index: int
    data_file_index: int
    video_locations: Dict[str, Tuple[int, int, float, float]]


class _ShardData:
    def __init__(self, table: Any, columns: Sequence[str]):
        self.arrays: Dict[str, np.ndarray] = {}
        for key in columns:
            if key not in table.column_names:
                continue
            self.arrays[key] = np.asarray(table[key].to_pylist())

        if "episode_index" not in self.arrays or "frame_index" not in self.arrays:
            raise KeyError("OLA data parquet must contain episode_index and frame_index")
        self.episode_rows: Dict[int, np.ndarray] = {}
        episode_ids = self.arrays["episode_index"].astype(np.int64).reshape(-1)
        frame_ids = self.arrays["frame_index"].astype(np.int64).reshape(-1)
        for episode_index in np.unique(episode_ids):
            rows = np.flatnonzero(episode_ids == episode_index)
            rows = rows[np.argsort(frame_ids[rows], kind="stable")]
            self.episode_rows[int(episode_index)] = rows

    def episode(self, episode_index: int) -> Dict[str, np.ndarray]:
        rows = self.episode_rows.get(int(episode_index))
        if rows is None:
            raise KeyError(f"episode {episode_index} not found in OLA data shard")
        return {key: values[rows] for key, values in self.arrays.items()}


class LeRobotOLAV3Dataset(Dataset):
    """Direct reader for OLA's aggregated LeRobot v3 layout."""

    def __init__(
        self,
        dataset_dir: str,
        global_downsample_rate: int = 1,
        video_action_freq_ratio: int = 2,
        num_video_frames: int = 8,
        video_size: Tuple[int, int] = (384, 320),
        max_episodes: Optional[int] = None,
        image_aug: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
        use_language_action: bool = True,
        val: bool = False,
        state_key: str = "observation.state",
        eef_state_key: str = "observation.ee_pose",
        absolute_action_key: str = "action.ee_pose",
        camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
        gripper_closed_value: float = 0.0,
        gripper_open_value: float = 100.0,
        normalize_state: bool = True,
        normalize_actions: bool = True,
        normalization_mode: str = "minmax",
        stats_path: Optional[str] = None,
        motus_dir_name: str = "motus",
        video_backend: str = "pyav",
        shard_cache_size: int = 2,
        excluded_episode_indices: Optional[Sequence[int]] = None,
        task_text_filter: Optional[str] = None,
        **_: Any,
    ):
        super().__init__()
        self.root = Path(dataset_dir).expanduser().resolve()
        self.info = self._read_json(self.root / "meta" / "info.json")
        if str(self.info.get("codebase_version", "")) != "v3.0":
            raise ValueError(
                f"LeRobotOLAV3Dataset requires codebase_version=v3.0, got {self.info.get('codebase_version')}"
            )

        self.global_downsample_rate = int(global_downsample_rate)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.num_video_frames = int(num_video_frames)
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.image_aug = bool(image_aug)
        self.val = bool(val)
        self.state_key = str(state_key)
        self.eef_state_key = str(eef_state_key)
        self.absolute_action_key = str(absolute_action_key)
        self.camera_keys = tuple(str(key) for key in camera_keys)
        if len(self.camera_keys) != 3:
            raise ValueError("OLA camera_keys must contain [right_front, left_wrist, right_wrist]")
        self.gripper_closed_value = float(gripper_closed_value)
        self.gripper_open_value = float(gripper_open_value)
        self.use_language_action = bool(use_language_action)
        self.motus_root = self.root / str(motus_dir_name)
        self.video_backend = str(video_backend)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self.excluded_episode_indices = {
            int(index) for index in (excluded_episode_indices or [])
        }
        self.task_text_filter = str(task_text_filter) if task_text_filter else None
        self._shard_cache: OrderedDict[Tuple[int, int], _ShardData] = OrderedDict()
        self._language_action_cache: Dict[int, Dict[int, str]] = {}
        self._t5_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        self._validate_features()
        episodes = self._load_episodes()
        if self.excluded_episode_indices:
            episodes = [
                episode
                for episode in episodes
                if episode.episode_index not in self.excluded_episode_indices
            ]
        if self.task_text_filter:
            episodes = [
                episode
                for episode in episodes
                if self.task_text_filter in episode.tasks
            ]
        if max_episodes is not None and int(max_episodes) > 0:
            episodes = episodes[: int(max_episodes)]
        physical_span = self.action_chunk_size * self.global_downsample_rate
        self.episodes = [episode for episode in episodes if episode.length > physical_span]
        if not self.episodes:
            raise ValueError("No OLA episodes are long enough for the configured action chunk")
        self.sample_counts = [max(1, episode.length - physical_span) for episode in self.episodes]
        self._cumulative_counts = np.cumsum(self.sample_counts)

        self.normalize_state = bool(normalize_state)
        self.normalize_actions = bool(normalize_actions)
        self.normalization_mode = str(normalization_mode).lower()
        self.state_lower = self.state_upper = None
        self.action_lower = self.action_upper = None
        if self.normalize_state or self.normalize_actions:
            resolved_stats = Path(stats_path) if stats_path else self.motus_root / "stats.json"
            self._load_stats(resolved_stats)

        self.vlm_processor = None
        if vlm_checkpoint_path:
            if AutoProcessor is None:
                raise ImportError("transformers is required to load the Qwen3-VL processor")
            self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)

        logger.info(
            "OLA v3 dataset initialized: root=%s episodes=%d samples=%d state=%s eef=%s action=%s "
            "excluded=%s task_filter=%s",
            self.root,
            len(self.episodes),
            len(self),
            self.state_key,
            self.eef_state_key,
            self.absolute_action_key,
            sorted(self.excluded_episode_indices),
            self.task_text_filter,
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _validate_features(self) -> None:
        features = self.info.get("features", {})
        expected = {
            self.state_key: 14,
            self.eef_state_key: 16,
            self.absolute_action_key: 16,
        }
        errors = []
        for key, dim in expected.items():
            feature = features.get(key)
            if feature is None:
                errors.append(f"missing {key!r}")
                continue
            shape = feature.get("shape", [])
            if not shape or int(shape[-1]) != dim:
                errors.append(f"{key!r} expected {dim}D, got shape={shape}")
        for key in self.camera_keys:
            feature = features.get(key)
            if feature is None or feature.get("dtype") != "video":
                errors.append(f"missing video feature {key!r}")
        if errors:
            raise ValueError("OLA v3 schema mismatch: " + "; ".join(errors))

    def _load_episodes(self) -> List[OLAEpisode]:
        import pyarrow.parquet as pq

        episodes: List[OLAEpisode] = []
        paths = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No v3 episode metadata parquet under {self.root / 'meta' / 'episodes'}")
        for path in paths:
            for row in pq.read_table(path).to_pylist():
                video_locations: Dict[str, Tuple[int, int, float, float]] = {}
                for key in self.camera_keys:
                    prefix = f"videos/{key}"
                    video_locations[key] = (
                        int(row[f"{prefix}/chunk_index"]),
                        int(row[f"{prefix}/file_index"]),
                        float(row[f"{prefix}/from_timestamp"]),
                        float(row[f"{prefix}/to_timestamp"]),
                    )
                tasks = row.get("tasks") or []
                episodes.append(
                    OLAEpisode(
                        episode_index=int(row["episode_index"]),
                        tasks=tuple(str(task) for task in tasks),
                        length=int(row["length"]),
                        data_chunk_index=int(row["data/chunk_index"]),
                        data_file_index=int(row["data/file_index"]),
                        video_locations=video_locations,
                    )
                )
        return sorted(episodes, key=lambda episode: episode.episode_index)

    def _load_stats(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"OLA stats file not found: {path}. Run data/lerobot/prepare_ola_v3_motus.py first."
            )
        stats = self._read_json(path)
        if self.normalization_mode == "minmax":
            low_key, high_key = "min", "max"
        elif self.normalization_mode in {"q01_q99", "quantile"}:
            low_key, high_key = "q01", "q99"
        else:
            raise ValueError(f"Unsupported OLA normalization_mode={self.normalization_mode!r}")
        self.state_lower = np.asarray(stats["state"][low_key], dtype=np.float32)
        self.state_upper = np.asarray(stats["state"][high_key], dtype=np.float32)
        self.action_lower = np.asarray(stats["action"][low_key], dtype=np.float32)
        self.action_upper = np.asarray(stats["action"][high_key], dtype=np.float32)
        if self.state_lower.shape != (14,) or self.action_lower.shape != (14,):
            raise ValueError(f"OLA normalization stats must be 14D: {path}")

    def __len__(self) -> int:
        return int(self._cumulative_counts[-1])

    def _sample_episode_and_condition(self, idx: int) -> Tuple[OLAEpisode, int]:
        if self.val:
            flat_index = int(idx) % len(self)
            ep_slot = int(np.searchsorted(self._cumulative_counts, flat_index, side="right"))
            previous = int(self._cumulative_counts[ep_slot - 1]) if ep_slot > 0 else 0
            return self.episodes[ep_slot], flat_index - previous
        episode = random.choice(self.episodes)
        max_condition = episode.length - self.action_chunk_size * self.global_downsample_rate - 1
        return episode, random.randint(0, max_condition)

    def _data_file_path(self, episode: OLAEpisode) -> Path:
        template = str(self.info["data_path"])
        rel = template.format(
            chunk_index=episode.data_chunk_index,
            file_index=episode.data_file_index,
            episode_index=episode.episode_index,
        )
        return self.root / rel

    def _get_shard(self, episode: OLAEpisode) -> _ShardData:
        cache_key = (episode.data_chunk_index, episode.data_file_index)
        cached = self._shard_cache.pop(cache_key, None)
        if cached is not None:
            self._shard_cache[cache_key] = cached
            return cached

        import pyarrow.parquet as pq

        columns = [
            self.state_key,
            self.eef_state_key,
            self.absolute_action_key,
            "timestamp",
            "frame_index",
            "episode_index",
            "task_index",
        ]
        table = pq.read_table(self._data_file_path(episode), columns=columns, memory_map=True)
        shard = _ShardData(table, columns)
        self._shard_cache[cache_key] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    def _sampling_indices(self, condition: int) -> Tuple[List[int], List[int]]:
        action_indices = [
            condition + (step + 1) * self.global_downsample_rate
            for step in range(self.action_chunk_size)
        ]
        video_indices = [
            action_indices[(frame + 1) * self.video_action_freq_ratio - 1]
            for frame in range(self.num_video_frames)
        ]
        return video_indices, action_indices

    def _video_path(self, episode: OLAEpisode, key: str) -> Tuple[Path, float]:
        chunk_index, file_index, from_timestamp, _ = episode.video_locations[key]
        rel = str(self.info["video_path"]).format(
            video_key=key,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_index=episode.episode_index,
        )
        return self.root / rel, from_timestamp

    def _decode_views(
        self,
        episode: OLAEpisode,
        local_timestamps: Sequence[float],
    ) -> List[torch.Tensor]:
        decoded = []
        fps = float(self.info.get("fps", 30))
        tolerance_s = 0.5 / fps + 1e-4
        for key in self.camera_keys:
            path, offset = self._video_path(episode, key)
            timestamps = [offset + float(timestamp) for timestamp in local_timestamps]
            frames = decode_video_frames(path, timestamps, tolerance_s, self.video_backend).squeeze(0).float()
            decoded.append(frames)
        return decoded

    def _stitch_three_views(self, front: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Match RobotWin's T mosaic before the final model resize.

        RobotWin keeps the front camera at its source resolution, resizes each
        wrist view to half the front width/height, concatenates the wrists
        below the front view, and finally resizes the whole 3:2-height mosaic
        to ``video_size``.  For 640x480 inputs this yields a 640x720 mosaic,
        so the front view occupies the top 2/3 of the model image.
        """
        if any(frame.ndim != 3 for frame in (front, left, right)):
            raise ValueError(
                "Expected CHW camera frames, got "
                f"front={tuple(front.shape)} left={tuple(left.shape)} right={tuple(right.shape)}"
            )
        source_h, source_w = int(front.shape[-2]), int(front.shape[-1])
        wrist_h = max(1, source_h // 2)
        left_w = source_w // 2
        right_w = source_w - left_w
        left_resized = F.interpolate(
            left.unsqueeze(0), size=(wrist_h, left_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        right_resized = F.interpolate(
            right.unsqueeze(0), size=(wrist_h, right_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        bottom = torch.cat([left_resized, right_resized], dim=-1)
        mosaic = torch.cat([front, bottom], dim=-2)
        return F.interpolate(
            mosaic.unsqueeze(0), size=self.video_size, mode="bilinear", align_corners=False
        ).squeeze(0)

    def _load_visuals(
        self,
        episode: OLAEpisode,
        timestamps: Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        front, left, right = self._decode_views(episode, timestamps)
        mosaics = [
            self._stitch_three_views(front[index], left[index], right[index])
            for index in range(front.shape[0])
        ]
        return mosaics[0], torch.stack(mosaics[1:], dim=0)

    def _load_t5_embedding(self, task_index: int, episode_index: int) -> torch.Tensor:
        cache_key = (int(task_index), int(episode_index))
        cached = self._t5_cache.get(cache_key)
        if cached is not None:
            return cached
        candidates = (
            self.motus_root / "t5_embeddings" / f"task_{task_index:06d}.pt",
            self.motus_root / "t5_embeddings" / f"episode_{episode_index:06d}.pt",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"OLA T5 embedding not found; checked: {candidates}")
        try:
            embedding = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch
            embedding = torch.load(path, map_location="cpu")
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        if embedding.ndim != 2:
            raise ValueError(f"OLA T5 embedding must be [S,D], got {tuple(embedding.shape)} at {path}")
        embedding = embedding.float()
        self._t5_cache[cache_key] = embedding
        return embedding

    def _load_language_action(self, episode_index: int, frame_index: int) -> str:
        cached = self._language_action_cache.get(int(episode_index))
        if cached is None:
            jsonl_path = self.motus_root / "language_action" / f"episode_{episode_index:06d}.jsonl"
            text_path = self.motus_root / "language_action" / f"episode_{episode_index:06d}.txt"
            cached = {}
            if jsonl_path.exists():
                for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cached[int(row["frame_index"])] = str(row["text"])
            elif text_path.exists():
                cached = {
                    index: line.strip()
                    for index, line in enumerate(text_path.read_text(encoding="utf-8").splitlines())
                    if line.strip()
                }
            else:
                raise FileNotFoundError(
                    f"OLA language_action missing for episode {episode_index}: {jsonl_path}"
                )
            self._language_action_cache[int(episode_index)] = cached
        if int(frame_index) not in cached:
            raise KeyError(f"No OLA language action for episode={episode_index}, frame={frame_index}")
        return cached[int(frame_index)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode, condition = self._sample_episode_and_condition(idx)
        rows = self._get_shard(episode).episode(episode.episode_index)
        if len(rows["frame_index"]) != episode.length:
            raise ValueError(
                f"OLA episode length mismatch for {episode.episode_index}: "
                f"meta={episode.length}, parquet={len(rows['frame_index'])}"
            )
        video_indices, action_indices = self._sampling_indices(condition)
        visual_indices = [condition] + video_indices
        timestamps = np.asarray(rows["timestamp"], dtype=np.float64).reshape(-1)[visual_indices].tolist()
        first_frame, video_frames = self._load_visuals(episode, timestamps)

        initial_state = normalize_joint_grippers(
            _as_float_tensor(rows[self.state_key][condition]),
            gripper_closed_value=self.gripper_closed_value,
            gripper_open_value=self.gripper_open_value,
        )
        measured_eef = _as_float_tensor(rows[self.eef_state_key][action_indices])
        commanded_eef = _as_float_tensor(rows[self.absolute_action_key][action_indices])
        action_sequence = absolute_eef_to_relative_rpy(
            measured_eef,
            commanded_eef,
            gripper_closed_value=self.gripper_closed_value,
            gripper_open_value=self.gripper_open_value,
        )
        if self.normalize_state:
            initial_state = _linear_normalize(initial_state, self.state_lower, self.state_upper)
        if self.normalize_actions:
            action_sequence = _linear_normalize(action_sequence, self.action_lower, self.action_upper)

        task_index = int(np.asarray(rows["task_index"]).reshape(-1)[condition])
        instruction = episode.tasks[0] if episode.tasks else ""
        language_embedding = self._load_t5_embedding(task_index, episode.episode_index)
        language_action = None
        if self.use_language_action:
            frame_index = int(np.asarray(rows["frame_index"]).reshape(-1)[condition])
            language_action = self._load_language_action(episode.episode_index, frame_index)

        vlm_inputs = None
        if self.vlm_processor is not None:
            first_frame_pil = tensor_to_pil(first_frame)
            if self.use_language_action:
                final_frame_pil = tensor_to_pil(video_frames[-1])
                vlm_inputs = preprocess_vlm_messages_lap(
                    instruction,
                    first_frame_pil,
                    self.vlm_processor,
                    language_action,
                    supervise_answer=True,
                    final_frame_pil=final_frame_pil,
                )
            else:
                vlm_inputs = preprocess_vlm_messages(instruction, first_frame_pil, self.vlm_processor)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames,
            "initial_state": initial_state.float(),
            "action_sequence": action_sequence.float(),
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_inputs,
        }
