import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_task(
    root: Path,
    task_name: str,
    episodes: list[dict],
    *,
    info_overrides: dict | None = None,
) -> Path:
    task_root = root / task_name
    (task_root / "meta").mkdir(parents=True)
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.cam_high_rgb": {"dtype": "video"},
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_arm_joint_1_rad", "right_arm_joint_1_rad"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_arm_joint_1_rad", "right_arm_joint_1_rad"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
        },
    }
    if info_overrides:
        info.update(info_overrides)
    (task_root / "meta" / "info.json").write_text(
        json.dumps(info),
        encoding="utf-8",
    )
    _write_jsonl(task_root / "meta" / "episodes.jsonl", episodes)
    return task_root


def test_discover_episodes_is_sorted_and_resolves_chunks(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    _make_task(tmp_path, "task_b", [{"episode_index": 1001, "length": 4}])
    _make_task(tmp_path, "task_a", [{"episode_index": 0, "length": 4}])

    episodes, failures = scanner.discover_episodes(tmp_path, None, None)

    assert failures == []
    assert [(item.task, item.episode_index) for item in episodes] == [
        ("task_a", 0),
        ("task_b", 1001),
    ]
    assert Path(episodes[1].parquet_path) == (
        tmp_path / "task_b/data/chunk-001/episode_001001.parquet"
    )


def test_discover_episodes_applies_task_and_episode_filters(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    _make_task(
        tmp_path,
        "keep",
        [
            {"episode_index": 1, "length": 3},
            {"episode_index": 2, "length": 3},
        ],
    )
    _make_task(tmp_path, "skip", [{"episode_index": 2, "length": 3}])

    episodes, failures = scanner.discover_episodes(tmp_path, "keep", 2)

    assert failures == []
    assert [(item.task, item.episode_index) for item in episodes] == [("keep", 2)]


def test_discover_episodes_records_invalid_task_and_continues(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    _make_task(tmp_path, "good", [{"episode_index": 0, "length": 4}])
    bad = tmp_path / "bad" / "meta"
    bad.mkdir(parents=True)
    (bad / "episodes.jsonl").write_text(
        '{"episode_index": 0, "length": 4}\n',
        encoding="utf-8",
    )

    episodes, failures = scanner.discover_episodes(tmp_path, None, None)

    assert [(item.task, item.episode_index) for item in episodes] == [("good", 0)]
    assert [(item.task, item.code) for item in failures] == [
        ("bad", "invalid_task_metadata")
    ]


def test_resolve_episode_path_rejects_escape(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    with pytest.raises(ValueError, match="outside task root"):
        scanner.resolve_episode_path(
            tmp_path,
            "../episode_{episode_index:06d}.parquet",
            3,
            1000,
        )
