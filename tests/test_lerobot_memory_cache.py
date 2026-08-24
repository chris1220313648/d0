import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if "lerobot.datasets.video_utils" not in sys.modules:
    lerobot_module = types.ModuleType("lerobot")
    datasets_module = types.ModuleType("lerobot.datasets")
    video_utils_module = types.ModuleType("lerobot.datasets.video_utils")
    lerobot_dataset_module = types.ModuleType("lerobot.datasets.lerobot_dataset")

    class _StubLeRobotDataset:
        pass

    video_utils_module.decode_video_frames = lambda *args, **kwargs: None
    lerobot_dataset_module.LeRobotDataset = _StubLeRobotDataset
    lerobot_dataset_module.LeRobotDatasetMetadata = _StubLeRobotDataset
    lerobot_dataset_module.MultiLeRobotDataset = _StubLeRobotDataset
    lerobot_module.datasets = datasets_module
    datasets_module.video_utils = video_utils_module
    datasets_module.lerobot_dataset = lerobot_dataset_module
    sys.modules["lerobot"] = lerobot_module
    sys.modules["lerobot.datasets"] = datasets_module
    sys.modules["lerobot.datasets.video_utils"] = video_utils_module
    sys.modules["lerobot.datasets.lerobot_dataset"] = lerobot_dataset_module


def _episode_meta_root(tmp_path: Path, episode_index: int = 7) -> SimpleNamespace:
    t5_path = tmp_path / "t5_embedding" / f"episode_{episode_index:06d}.pt"
    t5_path.parent.mkdir(parents=True)
    torch.save(torch.arange(6, dtype=torch.float32).reshape(1, 2, 3), t5_path)

    language_action_path = tmp_path / "language_action" / f"episode_{episode_index:06d}.txt"
    language_action_path.parent.mkdir(parents=True)
    language_action_path.write_text("first\nsecond\n", encoding="utf-8")

    return SimpleNamespace(
        root=tmp_path,
        meta=SimpleNamespace(
            episodes={
                episode_index: {
                    "episode_index": episode_index,
                    "t5_embedding_path": f"t5_embedding/episode_{episode_index:06d}.pt",
                    "language_action_path": f"language_action/episode_{episode_index:06d}.txt",
                }
            }
        ),
    )


def test_agibot_loaders_do_not_keep_worker_memory_caches(tmp_path):
    from data.lerobot.lerobot_agibot_dataset import LeRobotAgiBotDataset

    dataset = object.__new__(LeRobotAgiBotDataset)
    dataset.task_mode = "single"
    dataset.lerobot_dataset = _episode_meta_root(tmp_path)
    dataset.enable_t5_fallback = False
    dataset.language_action_dir_name = "language_action"
    dataset._episode_embedding_cache = {}
    dataset._language_action_cache = {}

    embedding = dataset._load_language_embedding({"episode_index": 7}, task_idx=0)
    language_action = dataset._load_language_action_from_file(7, task_idx=0, condition_frame_idx=1)

    assert embedding.shape == (2, 3)
    assert language_action == "second"
    assert dataset._episode_embedding_cache == {}
    assert dataset._language_action_cache == {}


def test_interndata_loaders_do_not_keep_worker_memory_caches(tmp_path):
    from data.lerobot.lerobot_interndata_dataset import LeRobotInternDataDataset

    dataset = object.__new__(LeRobotInternDataDataset)
    dataset.task_mode = "single"
    dataset.lerobot_dataset = _episode_meta_root(tmp_path)
    dataset.language_action_dir_name = "language_action"
    dataset._episode_embedding_cache = {}
    dataset._language_action_cache = {}

    embedding = dataset._load_language_embedding({"episode_index": 7}, task_idx=0)
    language_action = dataset._load_language_action_from_file(7, task_idx=0, condition_frame_idx=1)

    assert embedding.shape == (2, 3)
    assert language_action == "second"
    assert dataset._episode_embedding_cache == {}
    assert dataset._language_action_cache == {}


def test_robocoin_loaders_do_not_keep_worker_memory_caches(tmp_path):
    from data.lerobot.lerobot_robocoin_dataset import LeRobotRoboCOINDataset

    dataset = object.__new__(LeRobotRoboCOINDataset)
    dataset.task_mode = "single"
    dataset.lerobot_dataset = _episode_meta_root(tmp_path)
    dataset.enable_t5_fallback = False
    dataset.language_action_dir_name = "language_action"
    dataset._episode_embedding_cache = {}
    dataset._language_action_cache = {}

    embedding = dataset._load_language_embedding({"episode_index": 7}, task_idx=0)
    language_action = dataset._load_language_action_from_file(7, task_idx=0, condition_frame_idx=1)

    assert embedding.shape == (2, 3)
    assert language_action == "second"
    assert dataset._episode_embedding_cache == {}
    assert dataset._language_action_cache == {}
