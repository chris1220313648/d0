import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_tiny_lazy_robocoin(
    root: Path,
    task_name: str = "tiny_task",
    num_episodes: int = 1,
    include_language_action: bool = True,
) -> Path:
    task_root = root / task_name
    (task_root / "meta").mkdir(parents=True)
    (task_root / "data" / "chunk-000").mkdir(parents=True)
    (task_root / "videos" / "chunk-000" / "observation.images.cam_high_rgb").mkdir(parents=True)
    (task_root / "t5_embedding").mkdir()
    (task_root / "language_action").mkdir()

    info = {
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": num_episodes,
        "total_frames": 4 * num_episodes,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.cam_high_rgb": {"dtype": "video"},
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["s0", "s1"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["a0", "a1"]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (task_root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episodes = []
    for episode_index in range(num_episodes):
        episode = {
                "episode_index": episode_index,
                "tasks": ["pick object"],
                "length": 4,
                "t5_embedding_path": f"t5_embedding/episode_{episode_index:06d}.pt",
        }
        if include_language_action:
            episode["language_action_path"] = f"language_action/episode_{episode_index:06d}.txt"
        episodes.append(episode)
    _write_jsonl(task_root / "meta" / "episodes.jsonl", episodes)

    for episode_index in range(num_episodes):
        torch.save(torch.ones(1, 2, 3), task_root / "t5_embedding" / f"episode_{episode_index:06d}.pt")
        if include_language_action:
            (task_root / "language_action" / f"episode_{episode_index:06d}.txt").write_text(
                "move 0\nmove 1\nmove 2\nmove 3\n",
                encoding="utf-8",
            )
        (
            task_root
            / "videos"
            / "chunk-000"
            / "observation.images.cam_high_rgb"
            / f"episode_{episode_index:06d}.mp4"
        ).write_bytes(b"")

        pq.write_table(
            pa.table(
                {
                    "observation.state": [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
                    "action": [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5], [1.6, 1.7]],
                    "timestamp": [0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0],
                    "frame_index": [0, 1, 2, 3],
                    "episode_index": [episode_index] * 4,
                }
            ),
            task_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
        )
    return task_root


def test_eager_robocoin_missing_conventional_language_action_returns_none(tmp_path):
    from data.lerobot.lerobot_robocoin_dataset import LeRobotRoboCOINDataset

    dataset = object.__new__(LeRobotRoboCOINDataset)
    dataset.task_mode = "single"
    dataset.language_action_dir_name = "language_action"
    dataset._language_action_cache = {}
    dataset.lerobot_dataset = SimpleNamespace(
        root=tmp_path,
        meta=SimpleNamespace(episodes={0: {"episode_index": 0}}),
    )

    assert dataset._load_language_action_from_file(0, 0, 0) is None


def test_eager_robocoin_declared_language_action_path_still_raises(tmp_path):
    from data.lerobot.lerobot_robocoin_dataset import LeRobotRoboCOINDataset

    dataset = object.__new__(LeRobotRoboCOINDataset)
    dataset.task_mode = "single"
    dataset.language_action_dir_name = "language_action"
    dataset._language_action_cache = {}
    dataset.lerobot_dataset = SimpleNamespace(
        root=tmp_path,
        meta=SimpleNamespace(
            episodes={
                0: {
                    "episode_index": 0,
                    "language_action_path": "language_action/declared-but-missing.txt",
                }
            }
        ),
    )

    with pytest.raises(FileNotFoundError):
        dataset._load_language_action_from_file(0, 0, 0)


def test_lazy_robocoin_builds_episode_manifest_without_lerobot_dataset(tmp_path):
    _make_tiny_lazy_robocoin(tmp_path)

    from data.lerobot.lerobot_lazy_dataset import LazyLeRobotRoboCOINDataset

    dataset = LazyLeRobotRoboCOINDataset(
        root=str(tmp_path),
        repo_id="RoboCOIN",
        task_mode="multi",
        task_discovery={
            "enabled": True,
            "validate_episodes": True,
            "required": [
                "meta/info.json",
                "data",
                "videos",
                "language_action/episode_000000.txt",
                "t5_embedding/episode_000000.pt",
            ],
        },
        num_video_frames=1,
        video_action_freq_ratio=1,
        vlm_checkpoint_path=None,
        use_language_action=True,
    )

    assert dataset.lerobot_dataset is None
    assert len(dataset.lazy_episodes) == 1
    assert dataset.lazy_episodes[0].episode_index == 0
    assert dataset.lazy_episodes[0].length == 4


def test_lazy_robocoin_can_limit_to_one_episode_per_discovered_task(tmp_path):
    _make_tiny_lazy_robocoin(tmp_path, task_name="task_a", num_episodes=2)
    _make_tiny_lazy_robocoin(tmp_path, task_name="task_b", num_episodes=2)

    from data.lerobot.lerobot_lazy_dataset import LazyLeRobotRoboCOINDataset

    dataset = LazyLeRobotRoboCOINDataset(
        root=str(tmp_path),
        repo_id="RoboCOIN",
        task_mode="multi",
        task_discovery={"enabled": True, "validate_episodes": True},
        max_episodes=None,
        max_episodes_per_task=1,
        num_video_frames=1,
        video_action_freq_ratio=1,
        vlm_checkpoint_path=None,
    )

    assert len(dataset.lazy_episodes) == 2
    assert {episode.task_name for episode in dataset.lazy_episodes} == {"task_a", "task_b"}
    assert [episode.episode_index for episode in dataset.lazy_episodes] == [0, 0]


def test_lazy_robocoin_loads_sample_on_demand(tmp_path, monkeypatch):
    _make_tiny_lazy_robocoin(tmp_path)

    import data.lerobot.lerobot_robocoin_dataset as robocoin_module
    from data.lerobot.lerobot_lazy_dataset import LazyLeRobotRoboCOINDataset

    def fake_decode(video_path, timestamps, tolerance_s, backend):
        return torch.ones(1, len(timestamps), 3, 8, 8)

    monkeypatch.setattr(robocoin_module, "decode_video_frames", fake_decode)

    dataset = LazyLeRobotRoboCOINDataset(
        root=str(tmp_path),
        repo_id="RoboCOIN",
        task_mode="multi",
        task_discovery={"enabled": True, "validate_episodes": True},
        num_video_frames=1,
        video_action_freq_ratio=1,
        video_size=(8, 8),
        vlm_checkpoint_path=None,
        use_language_action=True,
    )

    sample = dataset[0]

    assert sample["first_frame"].shape == (3, 8, 8)
    assert sample["video_frames"].shape == (1, 3, 8, 8)
    expected_states = [
        torch.tensor([0.0, 0.1]),
        torch.tensor([0.2, 0.3]),
        torch.tensor([0.4, 0.5]),
    ]
    assert any(torch.allclose(sample["initial_state"], expected) for expected in expected_states)
    assert sample["action_sequence"].shape == (1, 2)
    assert sample["language_embedding"].shape == (2, 3)


def test_lazy_robocoin_missing_language_action_uses_default_vlm_inputs(tmp_path, monkeypatch):
    _make_tiny_lazy_robocoin(tmp_path, include_language_action=False)

    import data.lerobot.lerobot_robocoin_dataset as robocoin_module
    from data.lerobot.lerobot_lazy_dataset import LazyLeRobotRoboCOINDataset

    def fake_decode(video_path, timestamps, tolerance_s, backend):
        return torch.ones(1, len(timestamps), 3, 8, 8)

    def fake_default_preprocess(text_instr, image_pil, processor):
        return {"mode": "default"}

    def reject_lap_preprocess(*args, **kwargs):
        raise AssertionError("missing language_action must not use LAP preprocessing")

    monkeypatch.setattr(robocoin_module, "decode_video_frames", fake_decode)
    monkeypatch.setattr(robocoin_module, "preprocess_vlm_messages", fake_default_preprocess)
    monkeypatch.setattr(robocoin_module, "preprocess_vlm_messages_lap", reject_lap_preprocess)

    dataset = LazyLeRobotRoboCOINDataset(
        root=str(tmp_path),
        repo_id="RoboCOIN",
        task_mode="multi",
        task_discovery={"enabled": True, "validate_episodes": True},
        num_video_frames=1,
        video_action_freq_ratio=1,
        video_size=(8, 8),
        vlm_checkpoint_path=None,
        use_language_action=True,
    )
    dataset.vlm_processor = object()

    sample = dataset[0]

    assert sample["vlm_inputs"] == {"mode": "default"}
    assert sample["action_sequence"].shape == (1, 2)


def test_lazy_robocoin_available_language_action_keeps_lap_inputs(tmp_path, monkeypatch):
    _make_tiny_lazy_robocoin(tmp_path)

    import data.lerobot.lerobot_robocoin_dataset as robocoin_module
    from data.lerobot.lerobot_lazy_dataset import LazyLeRobotRoboCOINDataset

    def fake_decode(video_path, timestamps, tolerance_s, backend):
        return torch.ones(1, len(timestamps), 3, 8, 8)

    def reject_default_preprocess(*args, **kwargs):
        raise AssertionError("available language_action must keep LAP preprocessing")

    def fake_lap_preprocess(
        text_instr,
        image_pil,
        processor,
        language_action,
        supervise_answer,
    ):
        assert language_action.startswith("move ")
        assert supervise_answer is True
        return {"mode": "lap"}

    monkeypatch.setattr(robocoin_module, "decode_video_frames", fake_decode)
    monkeypatch.setattr(robocoin_module, "preprocess_vlm_messages", reject_default_preprocess)
    monkeypatch.setattr(robocoin_module, "preprocess_vlm_messages_lap", fake_lap_preprocess)

    dataset = LazyLeRobotRoboCOINDataset(
        root=str(tmp_path),
        repo_id="RoboCOIN",
        task_mode="multi",
        task_discovery={"enabled": True, "validate_episodes": True},
        num_video_frames=1,
        video_action_freq_ratio=1,
        video_size=(8, 8),
        vlm_checkpoint_path=None,
        use_language_action=True,
    )
    dataset.vlm_processor = object()

    sample = dataset[0]

    assert sample["vlm_inputs"] == {"mode": "lap"}


def test_dataset_factory_exposes_lazy_robocoin_type(tmp_path):
    _make_tiny_lazy_robocoin(tmp_path)

    from omegaconf import OmegaConf

    from data.dataset import _create_single_dataset
    from data.lerobot.lerobot_lazy_dataset import LazyLeRobotRoboCOINDataset

    config = OmegaConf.create(
        {
            "common": {
                "global_downsample_rate": 1,
                "video_action_freq_ratio": 1,
                "num_video_frames": 1,
                "video_height": 8,
                "video_width": 8,
            },
            "dataset": {
                "type": "lerobot_robocoin_lazy",
                "task_mode": "multi",
                "task_name": None,
                "max_episodes": 1,
                "image_aug": False,
                "use_language_action": True,
                "params": {
                    "root": str(tmp_path),
                    "repo_id": "RoboCOIN",
                    "task_discovery": {
                        "enabled": True,
                        "validate_episodes": True,
                    },
                },
            },
            "model": {"vlm": {"checkpoint_path": None}},
        }
    )

    dataset = _create_single_dataset(config, val=False)

    assert isinstance(dataset, LazyLeRobotRoboCOINDataset)
    assert len(dataset.lazy_episodes) == 1
