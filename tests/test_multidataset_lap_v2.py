from pathlib import Path
import json
import sys
from types import ModuleType

import torch
import yaml
from omegaconf import OmegaConf

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


def test_multidataset_lap_v2_robot_weights_match_requested_values():
    config_path = Path("configs/multidataset_lap_v2.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    datasets = config["dataset"]["datasets"]
    weights = {item["name"]: item["weight"] for item in datasets}
    types = {item["name"]: item["type"] for item in datasets}

    assert weights == {
        "agibot_lerobot_split": 0.2187,
        "bridge_pt_v5": 0.0067,
        "droid_pt_v5": 0.0895,
        "fractal_pt_v5": 0.0251,
        "robocoin_lerobot": 0.2171,
        "interndata_a1": 0.3428,
        "robotwin": 0.1000,
    }
    assert types == {
        "agibot_lerobot_split": "lerobot_agibot",
        "bridge_pt_v5": "bridge",
        "droid_pt_v5": "droid",
        "fractal_pt_v5": "fractal_bridge",
        "robocoin_lerobot": "lerobot_robocoin",
        "interndata_a1": "lerobot_interndata",
        "robotwin": "robotwin",
    }
    assert "egoverse_trimodal" not in types.values()
    assert "image_qa" not in types.values()


def test_multidataset_lap_v2_lazy_config_uses_lazy_lerobot_types():
    config_path = Path("configs/multidataset_lap_v2_lazy.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    datasets = config["dataset"]["datasets"]
    types = {item["name"]: item["type"] for item in datasets}

    assert types["agibot_lerobot_split"] == "lerobot_agibot_lazy"
    assert types["robocoin_lerobot"] == "lerobot_robocoin_lazy"
    assert types["interndata_a1"] == "lerobot_interndata_lazy"
    assert types["robotwin"] == "robotwin"


def test_v2_configs_enable_shared_canonical_normalization_for_robocoin_and_interndata():
    for config_name in ("multidataset_lap_v2.yaml", "multidataset_lap_v2_lazy.yaml"):
        config = yaml.safe_load((Path("configs") / config_name).read_text(encoding="utf-8"))
        by_name = {item["name"]: item for item in config["dataset"]["datasets"]}

        for dataset_name, stats_key in (
            ("robocoin_lerobot", "robocoin"),
            ("interndata_a1", "interndata"),
        ):
            normalization = by_name[dataset_name]["canonical_normalization"]
            assert normalization == {
                "enabled": True,
                "stats_path": "data/utils/stat.json",
                "stats_key": stats_key,
                "normalize_state": True,
                "normalize_action": True,
            }


def test_dataset_factory_passes_canonical_normalization_to_wrapper(tmp_path):
    from torch.utils.data import Dataset

    from data.canonical55 import Canonical55Dataset
    from data.dataset import _maybe_wrap_canonical_dataset

    stats_path = tmp_path / "stat.json"
    active = [True] * 55
    stats_path.write_text(
        json.dumps(
            {
                "interndata": {
                    "state": {"min": [0.0] * 55, "max": [2.0] * 55, "active_mask": active},
                    "action": {"min": [0.0] * 55, "max": [4.0] * 55, "active_mask": active},
                }
            }
        ),
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "dataset": {
                "canonical_format": "canonical55_v2",
                "canonical_normalization": {
                    "enabled": True,
                    "stats_path": str(stats_path),
                    "stats_key": "interndata",
                    "normalize_state": True,
                    "normalize_action": True,
                },
            }
        }
    )

    class CanonicalSample(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "initial_state": torch.ones(55),
                "state_mask": torch.ones(55, dtype=torch.bool),
                "action_sequence": torch.full((1, 55), 2.0),
                "action_mask": torch.ones(1, 55, dtype=torch.bool),
            }

    wrapped = _maybe_wrap_canonical_dataset(
        CanonicalSample(), config, dataset_type="lerobot_interndata_lazy"
    )
    sample = wrapped[0]

    assert isinstance(wrapped, Canonical55Dataset)
    assert torch.allclose(sample["initial_state"], torch.full((55,), 0.5))
    assert torch.allclose(sample["action_sequence"], torch.full((1, 55), 0.5))


def test_multidataset_lap_v2_robocoin_maps_directly_to_canonical55():
    config_path = Path("configs/multidataset_lap_v2.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    robocoin = next(item for item in config["dataset"]["datasets"] if item["name"] == "robocoin_lerobot")
    params = robocoin["params"]

    assert config["dataset"]["target_action_dim"] == 55
    assert config["dataset"]["target_state_dim"] == 55
    assert "target_action_dim" not in params
    assert "target_state_dim" not in params


def test_interndata_columns_map_joint_eef_and_gripper_to_canonical55():
    from data.lerobot.lerobot_interndata_dataset import build_interndata_canonical55

    columns = {
        "actions.joint.position": torch.arange(7, dtype=torch.float32).reshape(1, 7),
        "actions.tcp_to_robot_pose": torch.tensor(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "actions.gripper.position": torch.tensor([0.7], dtype=torch.float32),
    }

    value, mask = build_interndata_canonical55(columns, prefix="actions")

    assert value.shape == (1, 55)
    assert mask.shape == (1, 55)
    assert value[0, 0:7].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert mask[0, 0:7].all()
    assert value[0, 14:20].tolist() == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    assert mask[0, 14:20].all()
    assert value[0, 26].item() == torch.tensor(0.7, dtype=torch.float32).item()
    assert mask[0, 26]


def test_interndata_single_frame_vectors_are_feature_dimensions():
    from data.lerobot.lerobot_interndata_dataset import build_interndata_canonical55

    columns = {
        "states.joint.position": torch.arange(7, dtype=torch.float32),
        "states.tcp_to_robot_pose": torch.tensor(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        ),
    }

    value, mask = build_interndata_canonical55(columns, prefix="states")

    assert value.shape == (1, 55)
    assert value[0, 0:7].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert mask[0, 0:7].all()
    assert value[0, 14:20].tolist() == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    assert mask[0, 14:20].all()


def test_interndata_action_chunk_accepts_list_of_tensors():
    from data.lerobot.lerobot_interndata_dataset import build_interndata_canonical55

    columns = {
        "actions.joint.position": [
            torch.arange(7, dtype=torch.float32),
            torch.arange(7, dtype=torch.float32) + 10.0,
        ],
        "actions.gripper.position": [torch.tensor(0.2), torch.tensor(0.8)],
    }

    value, mask = build_interndata_canonical55(columns, prefix="actions")

    assert value.shape == (2, 55)
    assert value[:, 0:7].tolist() == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    ]
    assert torch.allclose(value[:, 26], torch.tensor([0.2, 0.8]))
    assert mask[:, 0:7].all()
    assert mask[:, 26].all()


def test_interndata_dual_eef14_action_and_joint14_state_mapping():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_interndata_dataset import build_interndata_dual_eef14_joint14

    columns = {
        "actions.left_tcp_to_robot_pose": torch.tensor(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], [11.0, 12.0, 13.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "actions.right_tcp_to_robot_pose": torch.tensor(
            [[4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0], [14.0, 15.0, 16.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "actions.left_gripper.position": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "actions.right_gripper.position": torch.tensor([0.3, 0.4], dtype=torch.float32),
        "states.left_joint.position": torch.tensor(
            [[20.0, 21.0, 22.0, 23.0, 24.0, 25.0], [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]],
            dtype=torch.float32,
        ),
        "states.right_joint.position": torch.tensor(
            [[26.0, 27.0, 28.0, 29.0, 30.0, 31.0], [36.0, 37.0, 38.0, 39.0, 40.0, 41.0]],
            dtype=torch.float32,
        ),
        "states.left_gripper.position": torch.tensor([0.5, 0.6], dtype=torch.float32),
        "states.right_gripper.position": torch.tensor([0.7, 0.8], dtype=torch.float32),
    }

    action, action_mask = build_interndata_dual_eef14_joint14(columns, prefix="actions")
    state, state_mask = build_interndata_dual_eef14_joint14(columns, prefix="states")

    assert action.shape == (2, 14)
    torch.testing.assert_close(
        action,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.1, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.3],
                [11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 0.2, 14.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.4],
            ],
            dtype=torch.float32,
        ),
    )
    assert action_mask.all()
    assert state.shape == (2, 14)
    torch.testing.assert_close(
        state,
        torch.tensor(
            [
                [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 0.5, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 0.7],
                [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 0.6, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 0.8],
            ],
            dtype=torch.float32,
        ),
    )
    assert state_mask.all()


def test_interndata_dual14_action_falls_back_to_joint_when_eef_pose_missing():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_interndata_dataset import build_interndata_dual_eef14_joint14

    columns = {
        "actions.left_joint.position": torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]],
            dtype=torch.float32,
        ),
        "actions.right_joint.position": torch.tensor(
            [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [17.0, 18.0, 19.0, 20.0, 21.0, 22.0]],
            dtype=torch.float32,
        ),
        "actions.left_gripper.position": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "actions.right_gripper.position": torch.tensor([0.3, 0.4], dtype=torch.float32),
    }

    action, action_mask = build_interndata_dual_eef14_joint14(columns, prefix="actions")

    assert action.shape == (2, 14)
    torch.testing.assert_close(
        action,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.3],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.2, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 0.4],
            ],
            dtype=torch.float32,
        ),
    )
    assert action_mask.all()


def test_interndata_dual14_state_falls_back_to_joint_vector_when_split_joints_missing():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_interndata_dataset import build_interndata_dual_eef14_joint14

    columns = {
        "states.joint.position": torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 98.0]],
            dtype=torch.float32,
        ),
        "states.effector.position": torch.tensor([[0.1, 0.3]], dtype=torch.float32),
    }

    state, state_mask = build_interndata_dual_eef14_joint14(columns, prefix="states")

    assert state.shape == (1, 14)
    torch.testing.assert_close(
        state,
        torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.3]],
            dtype=torch.float32,
        ),
    )
    assert state_mask.all()


def test_interndata_dual14_maps_single_arm_franka_eef_action_and_joint_state_to_left_slots():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_interndata_dataset import build_interndata_dual_eef14_joint14

    columns = {
        "actions.tcp_to_robot_pose": torch.tensor(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        ),
        "actions.gripper.position": torch.tensor([0.2], dtype=torch.float32),
        "states.joint.position": torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]], dtype=torch.float32),
    }

    action, action_mask = build_interndata_dual_eef14_joint14(columns, prefix="actions")
    state, state_mask = build_interndata_dual_eef14_joint14(columns, prefix="states")

    torch.testing.assert_close(
        action,
        torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        state,
        torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert action_mask[0, :7].all() and not action_mask[0, 7:].any()
    assert state_mask[0, :7].all() and not state_mask[0, 7:].any()


def test_interndata_action_signal_uses_qpos_for_joint_fallback_and_epos_for_eef():
    _install_lerobot_video_stub()
    from data.lerobot.lerobot_interndata_dataset import infer_interndata_action_signal

    assert (
        infer_interndata_action_signal(
            {
                "actions.left_tcp_to_robot_pose": torch.zeros(1, 7),
                "actions.right_tcp_to_robot_pose": torch.zeros(1, 7),
            },
            default_signal="epos",
        )
        == "epos"
    )
    assert (
        infer_interndata_action_signal(
            {
                "actions.left_joint.position": torch.zeros(1, 6),
                "actions.right_joint.position": torch.zeros(1, 6),
                "actions.left_gripper.position": torch.zeros(1),
                "actions.right_gripper.position": torch.zeros(1),
            },
            default_signal="epos",
        )
        == "qpos"
    )
    assert infer_interndata_action_signal({"action": torch.zeros(1, 14)}, default_signal="epos") == "qpos"


def test_dataset_factory_passes_interndata_setup_control_suffix_options(monkeypatch):
    _install_lerobot_video_stub()
    from omegaconf import OmegaConf

    import data.lerobot.lerobot_interndata_dataset as interndata_module
    from data.dataset import _create_single_dataset

    captured = {}

    class FakeInternDataDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(interndata_module, "LeRobotInternDataDataset", FakeInternDataDataset)

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
                "type": "lerobot_interndata",
                "task_mode": "multi",
                "task_name": None,
                "max_episodes": None,
                "image_aug": False,
                "use_language_action": False,
                "normalize_actions": False,
                "enable_setup_control_suffix": True,
                "setup_text": "dual-arm InternData robot with grippers",
                "action_stats_signal": "qpos",
                "params": {
                    "root": "/tmp/interndata",
                    "repo_id": "InternData-A1",
                    "output_format": "dual_eef14_joint14",
                },
            },
            "model": {"vlm": {"checkpoint_path": None}},
        }
    )

    _create_single_dataset(config, val=False)

    assert captured["enable_setup_control_suffix"] is True
    assert captured["setup_text"] == "dual-arm InternData robot with grippers"
    assert captured["action_signal"] == "qpos"


def test_multidataset_lap_v4_14a_uses_strict_14d_datasets():
    config = yaml.safe_load(Path("configs/multidataset_lap_v4_14a.yaml").read_text(encoding="utf-8"))

    assert config["common"]["action_dim"] == 14
    assert config["common"]["state_dim"] == 14
    assert config["dataset"]["target_action_dim"] == 14
    assert config["dataset"]["target_state_dim"] == 14
    assert "canonical_format" not in config["dataset"]

    active = {item["name"]: item for item in config["dataset"]["datasets"]}
    assert set(active) == {"agibot_lerobot_split", "interndata_a1"}
    assert active["agibot_lerobot_split"]["params"]["output_format"] == "dual_eef14_joint14"
    assert active["agibot_lerobot_split"]["use_for_val"] is True
    assert active["interndata_a1"]["params"]["output_format"] == "dual_eef14_joint14"
    assert active["interndata_a1"]["use_for_val"] is True
    assert active["interndata_a1"]["use_language_action"] is False
    assert "language_action/episode_000000.txt" not in active["interndata_a1"]["params"]["task_discovery"]["required"]


def test_lerobot_task_discovery_skips_incomplete_local_episodes(tmp_path):
    from data.lerobot.lerobot_dataset import LeRobotMotusDataset

    def make_task(name, complete):
        root = tmp_path / name
        (root / "meta").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "language_action").mkdir()
        (root / "t5_embedding").mkdir()
        (root / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "chunks_size": 1000,
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                }
            ),
            encoding="utf-8",
        )
        (root / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps({"episode_index": idx}) for idx in range(2)) + "\n",
            encoding="utf-8",
        )
        (root / "language_action" / "episode_000000.txt").write_text("ok\n", encoding="utf-8")
        (root / "t5_embedding" / "episode_000000.pt").write_bytes(b"pt")
        (root / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"parquet")
        if complete:
            (root / "data" / "chunk-000" / "episode_000001.parquet").write_bytes(b"parquet")

    make_task("complete_task", complete=True)
    make_task("incomplete_task", complete=False)

    discovered = LeRobotMotusDataset._discover_task_names(
        str(tmp_path),
        {
            "enabled": True,
            "validate_episodes": True,
            "required": [
                "meta/info.json",
                "data",
                "language_action/episode_000000.txt",
                "t5_embedding/episode_000000.pt",
            ],
        },
    )

    assert discovered == ["complete_task"]


def test_lerobot_multi_episode_limit_keeps_deterministic_nonempty_tasks():
    from data.lerobot.lerobot_dataset import LeRobotMotusDataset

    limited = LeRobotMotusDataset._limit_multi_episode_ids(
        {
            "task_a": list(range(4)),
            "task_b": list(range(4)),
            "task_c": list(range(4)),
        },
        max_episodes=5,
    )

    assert sum(len(ids) for ids in limited.values()) == 5
    assert all(ids == sorted(ids) for ids in limited.values())
    assert limited == LeRobotMotusDataset._limit_multi_episode_ids(
        {
            "task_a": list(range(4)),
            "task_b": list(range(4)),
            "task_c": list(range(4)),
        },
        max_episodes=5,
    )
