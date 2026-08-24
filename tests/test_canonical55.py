import sys
import json
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_robotwin_qpos_maps_six_joint_arms_and_separate_grippers():
    from data.canonical55 import CANONICAL55_DIM, map_robotwin_qpos

    qpos = torch.arange(14, dtype=torch.float32)
    value, mask = map_robotwin_qpos(qpos)

    assert value.shape == (CANONICAL55_DIM,)
    assert mask.shape == (CANONICAL55_DIM,)
    assert value[:6].tolist() == [0, 1, 2, 3, 4, 5]
    assert value[6].item() == 0
    assert value[7:13].tolist() == [7, 8, 9, 10, 11, 12]
    assert value[13].item() == 0
    assert value[26:28].tolist() == [6, 13]
    assert mask[:6].all()
    assert not mask[6]
    assert mask[7:13].all()
    assert not mask[13]
    assert mask[26:28].all()


def test_primary_eef_action_maps_pose_and_gripper_without_joint_mask():
    from data.canonical55 import map_primary_eef_7d

    eef = torch.tensor([0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 0.5])
    value, mask = map_primary_eef_7d(eef)

    assert value[14:20].tolist() == eef[:6].tolist()
    assert value[26].item() == eef[6].item()
    assert not mask[:14].any()
    assert mask[14:20].all()
    assert mask[26]
    assert not mask[27]


def test_agibot_24d_maps_dual_eef_gripper_head_waist_and_base_slots():
    from data.canonical55 import map_agibot_24d

    action = torch.arange(24, dtype=torch.float32)
    value, mask = map_agibot_24d(action)

    assert value[14:20].tolist() == [0, 1, 2, 3, 4, 5]
    assert value[20:26].tolist() == [7, 8, 9, 10, 11, 12]
    assert value[26:28].tolist() == [6, 13]
    assert value[44:46].tolist() == [14, 15]
    assert value[40:44].tolist() == [17, 18, 19, 20]
    assert value[46:48].tolist() == [22, 23]
    assert mask[14:28].all()
    assert mask[40:48].all()


def test_canonical_wrapper_converts_sample_and_preserves_existing_fields():
    from torch.utils.data import Dataset

    from data.canonical55 import Canonical55Dataset

    class OneSampleDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "initial_state": torch.arange(14, dtype=torch.float32),
                "action_sequence": torch.arange(28, dtype=torch.float32).reshape(2, 14),
                "first_frame": torch.zeros(3, 4, 4),
            }

    wrapped = Canonical55Dataset(OneSampleDataset(), dataset_type="robotwin")
    sample = wrapped[0]

    assert sample["initial_state"].shape == (55,)
    assert sample["action_sequence"].shape == (2, 55)
    assert sample["state_mask"].shape == (55,)
    assert sample["action_mask"].shape == (2, 55)
    assert sample["canonical_format"] == "canonical55_v2"
    assert torch.equal(sample["first_frame"], torch.zeros(3, 4, 4))


def test_zero_action_dataset_keeps_55d_masks_invalid():
    from torch.utils.data import Dataset

    from data.canonical55 import Canonical55Dataset

    class ZeroSampleDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "initial_state": torch.zeros(55),
                "action_sequence": torch.zeros(2, 55),
            }

    sample = Canonical55Dataset(ZeroSampleDataset(), dataset_type="egoverse_trimodal")[0]

    assert sample["initial_state"].shape == (55,)
    assert sample["action_sequence"].shape == (2, 55)
    assert not sample["state_mask"].any()
    assert not sample["action_mask"].any()


def test_passthrough_55d_preserves_existing_state_and_action_masks():
    from torch.utils.data import Dataset

    from data.canonical55 import Canonical55Dataset

    state_mask = torch.zeros(55, dtype=torch.bool)
    state_mask[:14] = True
    action_mask = torch.zeros(2, 55, dtype=torch.bool)
    action_mask[:, 14:28] = True

    class Masked55Dataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "initial_state": torch.ones(55),
                "state_mask": state_mask,
                "action_sequence": torch.ones(2, 55),
                "action_mask": action_mask,
            }

    sample = Canonical55Dataset(Masked55Dataset(), dataset_type="lerobot_agibot")[0]

    assert torch.equal(sample["state_mask"], state_mask)
    assert torch.equal(sample["action_mask"], action_mask)


def test_robocoin_named_vector_maps_eef_joint_gripper_and_ignores_velocity():
    from data.canonical55 import map_robocoin_named_vector

    names = [
        "left_arm_joint_1_rad",
        "left_arm_joint_2_vel_rad_s",
        "left_eef_pos_x_m",
        "left_eef_pos_y_m",
        "left_eef_pos_z_m",
        "left_eef_rot_euler_x_rad",
        "left_eef_rot_euler_y_rad",
        "left_eef_rot_euler_z_rad",
        "left_gripper_open",
        "right_arm_joint_1_rad",
        "right_eef_pos_x_m",
        "right_eef_pos_y_m",
        "right_eef_pos_z_m",
        "right_eef_rot_euler_x_rad",
        "right_eef_rot_euler_y_rad",
        "right_eef_rot_euler_z_rad",
        "right_gripper_open",
        "waist_yaw_rad",
        "head_pitch_rad",
        "left_hand_joint_1_rad",
        "right_hand_joint_1_rad",
    ]
    source = torch.arange(len(names), dtype=torch.float32)

    value, mask = map_robocoin_named_vector(source, names)

    assert value[0].item() == 0
    assert mask[0]
    assert value[14:20].tolist() == [2, 3, 4, 5, 6, 7]
    assert value[20:26].tolist() == [10, 11, 12, 13, 14, 15]
    assert value[26:28].tolist() == [8, 16]
    assert value[40].item() == 17
    assert value[45].item() == 18
    assert value[28].item() == 19
    assert value[34].item() == 20
    assert not mask[1]


def test_canonical_wrapper_uses_robocoin_field_names():
    from torch.utils.data import Dataset

    from data.canonical55 import Canonical55Dataset

    class RoboCoinSampleDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "initial_state": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                "state_names": [
                    "left_eef_pos_x_m",
                    "left_eef_pos_y_m",
                    "left_eef_pos_z_m",
                ],
                "action_sequence": torch.tensor([[1.0, 2.0, 3.0, 0.8]], dtype=torch.float32),
                "action_names": [
                    "right_eef_pos_x_m",
                    "right_eef_pos_y_m",
                    "right_eef_pos_z_m",
                    "right_gripper_open",
                ],
            }

    sample = Canonical55Dataset(RoboCoinSampleDataset(), dataset_type="lerobot_robocoin")[0]

    assert torch.allclose(sample["initial_state"][14:17], torch.tensor([0.1, 0.2, 0.3]))
    assert sample["state_mask"][14:17].all()
    assert torch.allclose(sample["action_sequence"][0, 20:23], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(sample["action_sequence"][0, 27], torch.tensor(0.8))
    assert sample["action_mask"][0, 20:23].all()
    assert sample["action_mask"][0, 27]


def test_canonical_wrapper_treats_lazy_lerobot_types_like_base_types():
    from torch.utils.data import Dataset

    from data.canonical55 import Canonical55Dataset

    class RoboCoinLazySampleDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "initial_state": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                "state_names": [
                    "left_eef_pos_x_m",
                    "left_eef_pos_y_m",
                    "left_eef_pos_z_m",
                ],
                "action_sequence": torch.tensor([[1.0, 2.0, 3.0, 0.8]], dtype=torch.float32),
                "action_names": [
                    "right_eef_pos_x_m",
                    "right_eef_pos_y_m",
                    "right_eef_pos_z_m",
                    "right_gripper_open",
                ],
            }

    sample = Canonical55Dataset(RoboCoinLazySampleDataset(), dataset_type="lerobot_robocoin_lazy")[0]

    assert torch.allclose(sample["initial_state"][14:17], torch.tensor([0.1, 0.2, 0.3]))
    assert sample["state_mask"][14:17].all()
    assert torch.allclose(sample["action_sequence"][0, 20:23], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(sample["action_sequence"][0, 27], torch.tensor(0.8))
    assert sample["action_mask"][0, 20:23].all()
    assert sample["action_mask"][0, 27]


def test_canonical_wrapper_normalizes_state_and_action_after_mapping(tmp_path):
    from torch.utils.data import Dataset

    from data.canonical55 import Canonical55Dataset

    active_mask = [False] * 55
    active_mask[14:17] = [True, True, True]
    active_mask[27] = True
    state_min = [0.0] * 55
    state_max = [0.0] * 55
    action_min = [0.0] * 55
    action_max = [0.0] * 55
    state_min[14], state_max[14] = 0.0, 2.0
    state_min[15], state_max[15] = 0.0, 1.0
    state_min[16], state_max[16] = 0.0, 1.0
    action_min[27], action_max[27] = 900.0, 1100.0
    stats_path = tmp_path / "stat.json"
    stats_path.write_text(
        json.dumps(
            {
                "robocoin": {
                    "state": {
                        "min": state_min,
                        "max": state_max,
                        "active_mask": active_mask,
                    },
                    "action": {
                        "min": action_min,
                        "max": action_max,
                        "active_mask": active_mask,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    class RawRoboCoinDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "initial_state": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
                "state_names": [
                    "left_eef_pos_x_m",
                    "left_eef_pos_y_m",
                    "left_eef_pos_z_m",
                ],
                "action_sequence": torch.tensor([[1000.0]], dtype=torch.float32),
                "action_names": ["right_gripper_open"],
            }

    wrapped = Canonical55Dataset(
        RawRoboCoinDataset(),
        dataset_type="lerobot_robocoin_lazy",
        normalization={
            "enabled": True,
            "stats_path": str(stats_path),
            "stats_key": "robocoin",
            "normalize_state": True,
            "normalize_action": True,
        },
    )
    sample = wrapped[0]

    assert sample["initial_state"][14].item() == 0.5
    assert sample["action_sequence"][0, 27].item() == 0.5
    assert sample["state_mask"][14]
    assert sample["action_mask"][0, 27]
