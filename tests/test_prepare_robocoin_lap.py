import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_euler_eef_action_generates_bimanual_lap_text():
    from data.lerobot.prepare_robocoin_lap import build_language_action_from_eef_source

    names = [
        "left_eef_pos_x_m",
        "left_eef_pos_y_m",
        "left_eef_pos_z_m",
        "left_eef_rot_euler_x_rad",
        "left_eef_rot_euler_y_rad",
        "left_eef_rot_euler_z_rad",
        "left_gripper_open",
        "right_eef_pos_x_m",
        "right_eef_pos_y_m",
        "right_eef_pos_z_m",
        "right_eef_rot_euler_x_rad",
        "right_eef_rot_euler_y_rad",
        "right_eef_rot_euler_z_rad",
        "right_gripper_open",
    ]
    action = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0.01, 0, 0.02, 0, 0, math.radians(10), 1, 0, -0.03, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    lines = build_language_action_from_eef_source(action, names, window_size=2, quat_order="xyzw")

    assert len(lines) == 2
    assert "Left arm: move forward 1.0 cm" in lines[0]
    assert "move up 2.0 cm" in lines[0]
    assert "rotate counterclockwise 10 degrees" in lines[0]
    assert "open gripper" in lines[0]
    assert "Right arm: move right 3.0 cm" in lines[0]
    assert "close gripper" in lines[0]


def test_quaternion_end_pose_generates_lap_text():
    from data.lerobot.prepare_robocoin_lap import build_language_action_from_eef_source

    names = [
        "left_end_pos_x_m",
        "left_end_pos_y_m",
        "left_end_pos_z_m",
        "left_end_quat_x",
        "left_end_quat_y",
        "left_end_quat_z",
        "left_end_quat_w",
        "left_gripper_open",
        "right_end_pos_x_m",
        "right_end_pos_y_m",
        "right_end_pos_z_m",
        "right_end_quat_x",
        "right_end_quat_y",
        "right_end_quat_z",
        "right_end_quat_w",
        "right_gripper_open",
    ]
    action = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0.02, 0, 0, 0, 0, 0, 1, 1, -0.01, 0, 0, 0, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )

    lines = build_language_action_from_eef_source(action, names, window_size=2, quat_order="xyzw")

    assert "Left arm: move forward 2.0 cm" in lines[0]
    assert "Right arm: move back 1.0 cm" in lines[0]


def test_episode_language_action_falls_back_from_action_to_state_eef():
    from data.lerobot.prepare_robocoin_lap import build_language_action_from_episode_arrays

    action = np.zeros((2, 2), dtype=np.float32)
    action_names = ["left_arm_joint_1_rad", "right_arm_joint_1_rad"]
    state_names = [
        "left_eef_pos_x_m",
        "left_eef_pos_y_m",
        "left_eef_pos_z_m",
        "left_eef_rot_euler_x_rad",
        "left_eef_rot_euler_y_rad",
        "left_eef_rot_euler_z_rad",
        "right_eef_pos_x_m",
        "right_eef_pos_y_m",
        "right_eef_pos_z_m",
        "right_eef_rot_euler_x_rad",
        "right_eef_rot_euler_y_rad",
        "right_eef_rot_euler_z_rad",
    ]
    state = np.zeros((2, len(state_names)), dtype=np.float32)
    state[1, 0] = 0.04

    result = build_language_action_from_episode_arrays(
        action,
        action_names,
        state,
        state_names,
        window_size=2,
        quat_order="xyzw",
    )

    assert result is not None
    lines, source = result
    assert source == "observation.state"
    assert "Left arm: move forward 4.0 cm" in lines[0]


def test_episode_language_action_returns_none_without_eef_fields():
    from data.lerobot.prepare_robocoin_lap import build_language_action_from_episode_arrays

    result = build_language_action_from_episode_arrays(
        np.zeros((2, 2), dtype=np.float32),
        ["left_arm_joint_1_rad", "right_arm_joint_1_rad"],
        np.zeros((2, 2), dtype=np.float32),
        ["left_arm_joint_1_rad", "right_arm_joint_1_rad"],
        window_size=2,
        quat_order="xyzw",
    )

    assert result is None


def test_ensure_language_action_respects_max_episodes(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from data.lerobot.prepare_robocoin_lap import ensure_language_action

    root = tmp_path / "robocoin_task"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    names = [
        "left_eef_pos_x_m",
        "left_eef_pos_y_m",
        "left_eef_pos_z_m",
        "left_eef_rot_euler_x_rad",
        "left_eef_rot_euler_y_rad",
        "left_eef_rot_euler_z_rad",
        "right_eef_pos_x_m",
        "right_eef_pos_y_m",
        "right_eef_pos_z_m",
        "right_eef_rot_euler_x_rad",
        "right_eef_rot_euler_y_rad",
        "right_eef_rot_euler_z_rad",
    ]
    (root / "meta" / "info.json").write_text(
        json.dumps({"features": {"action": {"names": names}, "observation.state": {"names": names}}}),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        '{"episode_index": 0, "tasks": ["task zero"], "length": 2}\n'
        '{"episode_index": 1, "tasks": ["task one"], "length": 2}\n',
        encoding="utf-8",
    )
    for ep_idx in (0, 1):
        action = [[0.0] * len(names), [0.01] + [0.0] * (len(names) - 1)]
        table = pa.table({"action": pa.array(action), "observation.state": pa.array(action)})
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

    stats = ensure_language_action(
        root,
        folder_name="language_action",
        window_size=2,
        quat_order="xyzw",
        overwrite=False,
        max_episodes=1,
    )

    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert stats.updated == 1
    assert (root / "language_action" / "episode_000000.txt").exists()
    assert not (root / "language_action" / "episode_000001.txt").exists()
    assert episodes[0]["language_action_path"] == "language_action/episode_000000.txt"
    assert "language_action_path" not in episodes[1]
