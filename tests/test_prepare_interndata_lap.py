import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_dataset_root(root: Path, name: str = "task") -> Path:
    dataset_root = root / name
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    (dataset_root / "videos").mkdir()
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "fps": 30, "features": {}}),
        encoding="utf-8",
    )
    _write_jsonl(
        dataset_root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["move object"], "length": 2}],
    )
    return dataset_root


def _read_episodes(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_discover_dataset_roots_finds_unpacked_lerobot_dirs_and_ignores_archives(tmp_path):
    from data.lerobot.prepare_interndata_lap import discover_dataset_roots

    expected = _make_dataset_root(tmp_path / "physical", "A2D")
    (tmp_path / "sim" / "basic_tasks").mkdir(parents=True)
    (tmp_path / "sim" / "basic_tasks" / "task.tar.gz").write_bytes(b"not a real archive")

    roots = discover_dataset_roots(tmp_path, include=("physical", "sim"))

    assert roots == [expected]


def test_select_shard_uses_stable_stride_partition():
    from data.lerobot.prepare_interndata_lap import select_shard

    roots = [Path(f"dataset_{idx}") for idx in range(10)]

    assert select_shard(roots, num_shards=3, shard_index=0) == [roots[0], roots[3], roots[6], roots[9]]
    assert select_shard(roots, num_shards=3, shard_index=1) == [roots[1], roots[4], roots[7]]
    assert select_shard(roots, num_shards=3, shard_index=2) == [roots[2], roots[5], roots[8]]


def test_ensure_language_action_writes_bimanual_tcp_lines_and_metadata(tmp_path):
    from data.lerobot.prepare_interndata_lap import ensure_language_action

    dataset_root = _make_dataset_root(tmp_path)
    left_pose = [[0, 0, 0, 1, 0, 0, 0], [0.01, 0, 0.02, 1, 0, 0, 0]]
    right_pose = [[0, 0, 0, 1, 0, 0, 0], [-0.03, 0, 0, 1, 0, 0, 0]]
    table = pa.table(
        {
            "actions.left_tcp_to_robot_pose": pa.array(left_pose),
            "actions.left_gripper.position": pa.array([0.0, 1.0]),
            "actions.right_tcp_to_robot_pose": pa.array(right_pose),
            "actions.right_gripper.position": pa.array([1.0, 0.0]),
        }
    )
    pq.write_table(table, dataset_root / "data" / "chunk-000" / "episode_000000.parquet")

    stats = ensure_language_action(
        dataset_root,
        folder_name="language_action",
        window_size=2,
        overwrite=False,
        max_episodes=0,
    )

    output = dataset_root / "language_action" / "episode_000000.txt"
    lines = output.read_text(encoding="utf-8").splitlines()
    episodes = _read_episodes(dataset_root / "meta" / "episodes.jsonl")
    assert stats.language_updated == 1
    assert stats.language_failed == 0
    assert len(lines) == 2
    assert "Left arm: move forward 1.0 cm" in lines[0]
    assert "move up 2.0 cm" in lines[0]
    assert "open gripper" in lines[0]
    assert "Right arm: move back 3.0 cm" in lines[0]
    assert "close gripper" in lines[0]
    assert episodes[0]["language_action_path"] == "language_action/episode_000000.txt"


def test_ensure_language_action_skips_joint_only_episode_without_bad_pointer(tmp_path):
    from data.lerobot.prepare_interndata_lap import ensure_language_action

    dataset_root = _make_dataset_root(tmp_path)
    table = pa.table({"actions.joint.position": pa.array([[0.0, 0.1], [0.2, 0.3]])})
    pq.write_table(table, dataset_root / "data" / "chunk-000" / "episode_000000.parquet")

    stats = ensure_language_action(
        dataset_root,
        folder_name="language_action",
        window_size=2,
        overwrite=False,
        max_episodes=0,
    )

    episodes = _read_episodes(dataset_root / "meta" / "episodes.jsonl")
    assert stats.language_updated == 0
    assert stats.language_failed == 1
    assert not (dataset_root / "language_action" / "episode_000000.txt").exists()
    assert "language_action_path" not in episodes[0]


def test_ensure_language_action_writes_single_arm_tcp_lines(tmp_path):
    from data.lerobot.prepare_interndata_lap import ensure_language_action

    dataset_root = _make_dataset_root(tmp_path)
    pose = [[0, 0, 0, 1, 0, 0, 0], [0, -0.02, 0.01, 1, 0, 0, 0]]
    table = pa.table(
        {
            "actions.tcp_to_robot_pose": pa.array(pose),
            "actions.gripper.openness": pa.array([0.0, 1.0]),
        }
    )
    pq.write_table(table, dataset_root / "data" / "chunk-000" / "episode_000000.parquet")

    stats = ensure_language_action(dataset_root, folder_name="language_action", window_size=2)

    lines = (dataset_root / "language_action" / "episode_000000.txt").read_text(encoding="utf-8").splitlines()
    assert stats.language_updated == 1
    assert "move up 1.0 cm" in lines[0]
    assert "move right 2.0 cm" in lines[0]
    assert "open gripper" in lines[0]


def test_ensure_t5_cache_writes_embeddings_and_preserves_unselected_episodes(tmp_path, monkeypatch):
    import data.lerobot.prepare_interndata_lap as module

    dataset_root = _make_dataset_root(tmp_path)
    _write_jsonl(
        dataset_root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["first task"], "length": 2},
            {"episode_index": 1, "tasks": ["second task"], "length": 2},
        ],
    )
    calls = []

    def fake_encode_t5(encoder, instruction: str, device: str):
        calls.append((encoder, instruction, device))
        return torch.ones(3, 4) * len(instruction)

    monkeypatch.setattr(module, "_encode_t5", fake_encode_t5)

    stats = module.ensure_t5_cache(
        dataset_root,
        encoder=object(),
        device="cpu",
        folder_name="t5_embedding",
        overwrite=False,
        max_episodes=1,
    )

    episodes = _read_episodes(dataset_root / "meta" / "episodes.jsonl")
    emb = torch.load(dataset_root / "t5_embedding" / "episode_000000.pt", map_location="cpu")
    assert stats.t5_updated == 1
    assert calls == [(calls[0][0], "first task", "cpu")]
    assert emb.shape == (3, 4)
    assert episodes[0]["t5_embedding_path"] == "t5_embedding/episode_000000.pt"
    assert "t5_embedding_path" not in episodes[1]
