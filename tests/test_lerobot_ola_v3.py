import math
import inspect
from types import SimpleNamespace

import numpy as np
import torch

from data.lerobot.lerobot_ola_v3_dataset import (
    LeRobotOLAV3Dataset,
    absolute_eef_to_relative_rpy,
    normalize_joint_grippers,
    relative_rpy_to_absolute_eef,
)
from data.lerobot.prepare_ola_v3_motus import _language_action_rows, prepare


def _dual_pose(left_xyz, left_quat, left_grip, right_xyz=None, right_quat=None, right_grip=0.0):
    right_xyz = left_xyz if right_xyz is None else right_xyz
    right_quat = left_quat if right_quat is None else right_quat
    return torch.tensor(
        list(left_xyz)
        + list(left_quat)
        + [left_grip]
        + list(right_xyz)
        + list(right_quat)
        + [right_grip],
        dtype=torch.float32,
    )


def test_absolute_eef_to_relative_rpy_identity_and_gripper():
    identity = [0.0, 0.0, 0.0, 1.0]
    measured = _dual_pose([0.1, 0.2, 0.3], identity, 0.0)
    commanded = _dual_pose([0.2, 0.1, 0.5], identity, 50.0)
    result = absolute_eef_to_relative_rpy(measured, commanded)
    assert result.shape == (14,)
    torch.testing.assert_close(result[:3], torch.tensor([0.1, -0.1, 0.2]))
    torch.testing.assert_close(result[3:6], torch.zeros(3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(result[6], torch.tensor(0.5))


def test_absolute_eef_to_relative_rpy_wraps_yaw_boundary():
    def yaw_quat(degrees):
        radians = math.radians(degrees)
        return [0.0, 0.0, math.sin(radians / 2.0), math.cos(radians / 2.0)]

    measured = _dual_pose([0.0, 0.0, 0.0], yaw_quat(179.0), 0.0)
    commanded = _dual_pose([0.0, 0.0, 0.0], yaw_quat(-179.0), 0.0)
    result = absolute_eef_to_relative_rpy(measured, commanded)
    assert abs(float(result[5]) - math.radians(2.0)) < 1e-5


def test_joint_gripper_normalization_only_changes_grippers():
    state = torch.arange(14, dtype=torch.float32)
    state[6] = 25.0
    state[13] = 75.0
    normalized = normalize_joint_grippers(
        state,
        gripper_closed_value=0.0,
        gripper_open_value=100.0,
    )
    torch.testing.assert_close(normalized[6], torch.tensor(0.25))
    torch.testing.assert_close(normalized[13], torch.tensor(0.75))
    torch.testing.assert_close(normalized[:6], state[:6])


def test_relative_absolute_round_trip():
    measured = _dual_pose(
        [0.2, -0.1, 0.4],
        [0.1, -0.2, 0.3, 0.9],
        10.0,
        [-0.2, 0.3, 0.1],
        [-0.1, 0.1, 0.2, 0.95],
        20.0,
    )
    commanded = _dual_pose(
        [0.25, -0.08, 0.35],
        [-0.05, 0.15, 0.25, 0.95],
        80.0,
        [-0.18, 0.25, 0.14],
        [0.2, -0.1, 0.05, 0.97],
        40.0,
    )
    relative = absolute_eef_to_relative_rpy(measured, commanded)
    reconstructed = relative_rpy_to_absolute_eef(measured, relative)
    torch.testing.assert_close(reconstructed[[0, 1, 2, 7, 8, 9, 10, 15]], commanded[[0, 1, 2, 7, 8, 9, 10, 15]], atol=1e-5, rtol=1e-5)
    for start in (3, 11):
        expected = commanded[start : start + 4] / torch.linalg.vector_norm(commanded[start : start + 4])
        actual = reconstructed[start : start + 4]
        assert abs(float(torch.dot(expected, actual))) > 1.0 - 1e-5


def test_three_camera_mosaic_shape_and_range():
    dataset = object.__new__(LeRobotOLAV3Dataset)
    dataset.video_size = (384, 320)
    front = torch.ones(3, 480, 640)
    left = torch.full((3, 480, 640), 0.25)
    right = torch.full((3, 480, 640), 0.75)
    mosaic = dataset._stitch_three_views(front, left, right)
    assert mosaic.shape == (3, 384, 320)
    assert float(mosaic.min()) >= 0.0
    assert float(mosaic.max()) <= 1.0
    # RobotWin geometry: front occupies top 2/3, wrists bottom 1/3.
    torch.testing.assert_close(mosaic[:, 100, 160], torch.ones(3))
    torch.testing.assert_close(mosaic[:, 330, 80], torch.full((3,), 0.25))
    torch.testing.assert_close(mosaic[:, 330, 240], torch.full((3,), 0.75))


def test_new_ola_schema_keys_are_loader_defaults():
    signature = inspect.signature(LeRobotOLAV3Dataset.__init__)
    assert signature.parameters["state_key"].default == "observation.state"
    assert signature.parameters["eef_state_key"].default == "observation.ee_pose"
    assert signature.parameters["absolute_action_key"].default == "action.ee_pose"


def test_history_indices_are_causal_fixed_length_and_episode_local():
    dataset = object.__new__(LeRobotOLAV3Dataset)
    dataset.history_action_length = 4
    dataset.global_downsample_rate = 1
    assert dataset._history_indices(0) == [0, 0, 0, 0]
    assert dataset._history_indices(2) == [0, 0, 1, 2]
    assert dataset._history_indices(5) == [2, 3, 4, 5]

    dataset.global_downsample_rate = 3
    assert dataset._history_indices(10) == [1, 4, 7, 10]


def test_lap_uses_same_future_delta_chunk_as_action_loss():
    relative = np.zeros((20, 14), dtype=np.float32)
    relative[0, 0] = 1.0  # Must not appear in the annotation conditioned at frame 0.
    relative[1:17, 0] = 0.001
    rows = _language_action_rows(relative, np.arange(20), window_size=16)
    first = rows[0]
    assert first["start_frame_index"] == 1
    assert first["end_frame_index"] == 16
    assert "move forward 1.6 cm" in first["text"]


def test_lap_supports_48_action_chunk_for_eight_video_frames():
    relative = np.zeros((60, 14), dtype=np.float32)
    relative[1:49, 0] = 0.001
    rows = _language_action_rows(relative, np.arange(60), window_size=48)
    first = rows[0]
    assert first["start_frame_index"] == 1
    assert first["end_frame_index"] == 48
    assert "move forward 4.8 cm" in first["text"]


def test_multi_root_stats_requires_explicit_shared_path_before_io():
    args = SimpleNamespace(
        roots=["dataset_a", "dataset_b"],
        write_stats=True,
        shared_stats_path=None,
    )
    try:
        prepare(args)
    except ValueError as error:
        assert "--shared-stats-path" in str(error)
    else:
        raise AssertionError("Expected multi-root preparation to require shared stats")
