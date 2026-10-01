from types import SimpleNamespace

import numpy as np
import torch

from data.lerobot.lerobot_ola_v3_dataset import LeRobotOLAV3Dataset


def test_joint_action_is_absolute_and_gripper_remains_continuous():
    dataset = object.__new__(LeRobotOLAV3Dataset)
    dataset.action_representation = "absolute_joint"
    dataset.joint_action_key = "action"
    dataset.state_key = "observation.state"
    dataset.global_downsample_rate = 1
    dataset.video_action_freq_ratio = 6
    dataset.num_video_frames = 8
    dataset.action_chunk_size = 48
    dataset.include_history_actions = True
    dataset.history_action_length = 48
    dataset.gripper_closed_value = 0.0
    dataset.gripper_open_value = 100.0
    dataset.normalize_state = False
    dataset.normalize_actions = False
    dataset.use_language_action = False
    dataset.vlm_processor = None

    actions = np.zeros((60, 14), dtype=np.float32)
    actions[:, 0] = np.arange(60)
    actions[:, 6] = 37.0
    actions[:, 13] = 62.0
    rows = {
        "action": actions,
        "observation.state": np.zeros((60, 14), dtype=np.float32),
        "timestamp": np.arange(60, dtype=np.float64) / 30,
        "frame_index": np.arange(60),
        "task_index": np.zeros(60, dtype=np.int64),
    }
    dataset._sample_episode_and_condition = lambda _: (SimpleNamespace(episode_index=0, length=60, tasks=("task",)), 0)
    dataset._get_shard = lambda _: SimpleNamespace(episode=lambda __: rows)
    dataset._load_visuals = lambda *_: (torch.zeros(3, 4, 4), torch.zeros(8, 3, 4, 4))
    dataset._load_t5_embedding = lambda *_: torch.zeros(1, 4)

    sample = dataset[0]
    assert sample["action_sequence"].shape == (48, 14)
    assert sample["action_sequence"][0, 0] == 1  # D0 still predicts t+1.
    assert sample["action_sequence"][0, 6] == 0.37
    assert sample["action_sequence"][0, 13] == 0.62
    assert sample["history_action_sequence"][-1, 6] == 0.37
