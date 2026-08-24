import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_lerobot_video_stub():
    lerobot_module = sys.modules.setdefault("lerobot", ModuleType("lerobot"))
    datasets_module = sys.modules.setdefault("lerobot.datasets", ModuleType("lerobot.datasets"))
    video_utils_module = sys.modules.setdefault("lerobot.datasets.video_utils", ModuleType("lerobot.datasets.video_utils"))
    lerobot_dataset_module = sys.modules.setdefault(
        "lerobot.datasets.lerobot_dataset", ModuleType("lerobot.datasets.lerobot_dataset")
    )
    setattr(lerobot_module, "datasets", datasets_module)
    setattr(datasets_module, "video_utils", video_utils_module)
    setattr(video_utils_module, "decode_video_frames", lambda *args, **kwargs: None)
    setattr(datasets_module, "lerobot_dataset", lerobot_dataset_module)
    setattr(lerobot_dataset_module, "LeRobotDataset", object)
    setattr(lerobot_dataset_module, "LeRobotDatasetMetadata", object)
    setattr(lerobot_dataset_module, "MultiLeRobotDataset", object)


def _fake_ds_media():
    return SimpleNamespace(
        root="/tmp/agibot",
        features={
            "observation.state": {
                "field_descriptions": {
                    "state/left_effector/position": {"indices": [0]},
                    "state/right_effector/position": {"indices": [1]},
                    "state/end/arm_orientation": {"indices": [26, 27, 28, 29, 30, 31, 32, 33]},
                    "state/end/arm_position": {"indices": [34, 35, 36, 37, 38, 39]},
                    "state/joint/position": {
                        "indices": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
                    },
                    "state/head/position": {"indices": [82, 83, 84]},
                    "state/waist/position": {"indices": [85, 86, 87, 88, 89]},
                }
            },
            "action": {
                "field_descriptions": {
                    "action/left_effector/position": {"indices": [0]},
                    "action/right_effector/position": {"indices": [1]},
                    "action/end/position": {"indices": [2, 3, 4, 5, 6, 7]},
                    "action/end/orientation": {"indices": [8, 9, 10, 11, 12, 13, 14, 15]},
                    "action/joint/position": {
                        "indices": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
                    },
                    "action/head/position": {"indices": [30, 31, 32]},
                    "action/waist/position": {"indices": [33, 34, 35, 36, 37]},
                    "action/robot/velocity": {"indices": [38, 39]},
                }
            },
        },
    )


def test_agibot_select_state_builds_canonical55_from_named_fields():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_agibot_dataset import LeRobotAgiBotDataset

    dataset = object.__new__(LeRobotAgiBotDataset)
    dataset._feature_fields_cache = {}

    raw_state = torch.zeros(169, dtype=torch.float32)
    raw_state[0] = 0.1
    raw_state[1] = 0.2
    raw_state[26:34] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    raw_state[34:40] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    raw_state[40:54] = torch.arange(10.0, 24.0)
    raw_state[82:85] = torch.tensor([30.0, 31.0, 32.0])
    raw_state[85:90] = torch.tensor([40.0, 41.0, 42.0, 43.0, 44.0])

    value, mask = dataset._select_state(raw_state, _fake_ds_media())

    assert value.shape == (55,)
    assert mask.shape == (55,)
    assert value[:14].tolist() == torch.arange(10.0, 24.0).tolist()
    assert value[14:17].tolist() == [1.0, 2.0, 3.0]
    assert value[20:23].tolist() == [4.0, 5.0, 6.0]
    assert torch.allclose(value[26:28], torch.tensor([0.1, 0.2]))
    assert value[40:44].tolist() == [40.0, 41.0, 42.0, 43.0]
    assert value[44:46].tolist() == [30.0, 31.0]
    assert mask[:28].all()
    assert not mask[28:40].any()
    assert mask[40:46].all()
    assert not mask[46:].any()


def test_agibot_select_action_sequence_builds_canonical55_from_named_fields():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_agibot_dataset import LeRobotAgiBotDataset

    dataset = object.__new__(LeRobotAgiBotDataset)
    dataset._feature_fields_cache = {}

    raw_actions = torch.zeros(2, 40, dtype=torch.float32)
    raw_actions[:, 0] = torch.tensor([0.1, 0.3])
    raw_actions[:, 1] = torch.tensor([0.2, 0.4])
    raw_actions[:, 2:8] = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    raw_actions[:, 8:16] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    raw_actions[:, 16:30] = torch.arange(28.0, dtype=torch.float32).reshape(2, 14)
    raw_actions[:, 30:33] = torch.tensor([[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]])
    raw_actions[:, 33:38] = torch.tensor([[40.0, 41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0, 49.0]])
    raw_actions[:, 38:40] = torch.tensor([[50.0, 51.0], [52.0, 53.0]])

    value, mask = dataset._select_action_sequence(raw_actions, raw_actions[0], _fake_ds_media())

    assert value.shape == (2, 55)
    assert mask.shape == (2, 55)
    assert value[:, :14].tolist() == raw_actions[:, 16:30].tolist()
    assert value[:, 14:17].tolist() == raw_actions[:, 2:5].tolist()
    assert value[:, 20:23].tolist() == raw_actions[:, 5:8].tolist()
    assert value[:, 26:28].tolist() == raw_actions[:, 0:2].tolist()
    assert value[:, 40:44].tolist() == raw_actions[:, 33:37].tolist()
    assert value[:, 44:46].tolist() == raw_actions[:, 30:32].tolist()
    assert value[:, 46:48].tolist() == raw_actions[:, 38:40].tolist()
    assert mask[:, :28].all()
    assert not mask[:, 28:40].any()
    assert mask[:, 40:48].all()
    assert not mask[:, 48:].any()


def test_agibot_dual_eef14_action_and_joint14_state_output():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_agibot_dataset import LeRobotAgiBotDataset

    dataset = object.__new__(LeRobotAgiBotDataset)
    dataset._feature_fields_cache = {}

    raw_state = torch.zeros(169, dtype=torch.float32)
    raw_state[0] = 0.1
    raw_state[1] = 0.2
    raw_state[40:54] = torch.arange(10.0, 24.0)

    state, state_mask = dataset._select_state_14d(raw_state, _fake_ds_media())

    assert state.shape == (14,)
    assert state_mask.shape == (14,)
    torch.testing.assert_close(
        state,
        torch.tensor(
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.1, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 0.2],
            dtype=torch.float32,
        ),
    )
    assert state_mask.all()

    raw_actions = torch.zeros(2, 40, dtype=torch.float32)
    raw_actions[:, 0] = torch.tensor([0.3, 0.5])
    raw_actions[:, 1] = torch.tensor([0.4, 0.6])
    raw_actions[:, 2:8] = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    )
    raw_actions[:, 8:16] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    raw_actions[:, 16:30] = torch.arange(28.0, dtype=torch.float32).reshape(2, 14)

    action, action_mask = dataset._select_action_sequence_14d(raw_actions, raw_actions[0], _fake_ds_media())

    assert action.shape == (2, 14)
    assert action_mask.shape == (2, 14)
    torch.testing.assert_close(
        action,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.3, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.4],
                [7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.5, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.6],
            ],
            dtype=torch.float32,
        ),
    )
    assert action_mask.all()


def test_dataset_factory_passes_agibot_normalization_options(monkeypatch):
    _install_lerobot_video_stub()
    from omegaconf import OmegaConf

    import data.lerobot.lerobot_agibot_dataset as agibot_module
    from data.dataset import _create_single_dataset

    captured = {}

    class FakeAgiBotDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agibot_module, "LeRobotAgiBotDataset", FakeAgiBotDataset)

    config = OmegaConf.create(
        {
            "common": {
                "global_downsample_rate": 1,
                "video_action_freq_ratio": 2,
                "num_video_frames": 8,
                "video_height": 384,
                "video_width": 320,
            },
            "dataset": {
                "type": "lerobot_agibot",
                "task_mode": "multi",
                "task_name": None,
                "max_episodes": None,
                "image_aug": False,
                "use_language_action": True,
                "normalize_actions": True,
                "stats_path": "/tmp/agibot-stat.json",
                "stats_key": "agibot",
                "enable_setup_control_suffix": True,
                "setup_text": "dual-arm AgiBot robot with grippers",
                "action_stats_signal": "epos",
                "params": {
                    "root": "/tmp/agibot",
                    "repo_id": "AgiBotWorld2026",
                    "embodiment_type": "aloha_agilex_2",
                },
            },
            "model": {"vlm": {"checkpoint_path": None}},
        }
    )

    _create_single_dataset(config, val=False)

    assert captured["normalize_actions"] is True
    assert captured["stats_path"] == "/tmp/agibot-stat.json"
    assert captured["stats_key"] == "agibot"
    assert captured["enable_setup_control_suffix"] is True
    assert captured["setup_text"] == "dual-arm AgiBot robot with grippers"
    assert captured["action_signal"] == "epos"


def test_agibot_stitch_keeps_source_frames_uint8_until_final_resize():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_agibot_dataset import LeRobotAgiBotDataset

    dataset = object.__new__(LeRobotAgiBotDataset)
    dataset.video_size = (8, 8)
    top = torch.full((3, 4, 6), 10, dtype=torch.uint8)
    left = torch.full((3, 4, 3), 20, dtype=torch.uint8)
    right = torch.full((3, 4, 3), 30, dtype=torch.uint8)

    stitched = dataset._stitch_three_views_uint8(top, left, right)
    final = dataset._resize_frame_chw(top, (8, 8))

    assert stitched.dtype.name == "uint8"
    assert stitched.shape == (8, 8, 3)
    assert final.shape == (3, 8, 8)
    assert final.dtype == torch.float32
    assert torch.all(top == 10)
