import importlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_tiny_robocoin_dataset(root: Path) -> Path:
    dataset_root = root / "Tiny_RoboCOIN_task"
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    (dataset_root / "videos" / "chunk-000" / "observation.images.cam_high_rgb").mkdir(parents=True)
    (dataset_root / "meta").mkdir()

    info = {
        "codebase_version": "v2.1",
        "robot_type": "tiny_robot",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 3,
        "features": {
            "observation.images.cam_high_rgb": {"dtype": "video"},
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["joint_1", "joint_2"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["joint_1", "joint_2"]},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(dataset_root / "meta" / "episodes.jsonl", [{"episode_index": 0, "tasks": ["pick item"], "length": 3}])
    _write_jsonl(dataset_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "pick item"}])

    annotations = {
        "subtask_annotations.jsonl": [
            {"subtask_index": 0, "subtask": "reach the item"},
            {"subtask_index": 1, "subtask": "grasp the item"},
        ],
        "eef_direction_annotation.jsonl": [
            {"eef_direction_index": 0, "eef_direction": "forward"},
            {"eef_direction_index": 1, "eef_direction": "left"},
        ],
        "eef_velocity_annotation.jsonl": [
            {"eef_velocity_index": 0, "eef_velocity": "still"},
            {"eef_velocity_index": 1, "eef_velocity": "slow"},
        ],
        "eef_acc_mag_annotation.jsonl": [
            {"eef_acc_mag_index": 0, "eef_acc_mag": "constant"},
            {"eef_acc_mag_index": 1, "eef_acc_mag": "accelerating"},
        ],
        "gripper_mode_annotation.jsonl": [
            {"gripper_mode_index": 0, "gripper_mode": "open"},
            {"gripper_mode_index": 1, "gripper_mode": "closed"},
        ],
        "gripper_activity_annotation.jsonl": [
            {"gripper_activity_index": 0, "gripper_activity": "holding"},
            {"gripper_activity_index": 1, "gripper_activity": "closing"},
        ],
    }
    for filename, rows in annotations.items():
        _write_jsonl(dataset_root / "annotations" / filename, rows)

    table = pa.table(
        {
            "action": [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
            "subtask_annotation": [[0], [0], [1]],
            "eef_direction_action": [[0, 1], [1, 0], [0, 0]],
            "eef_velocity_action": [[1, 0], [1, 1], [0, 0]],
            "eef_acc_mag_action": [[0, 1], [1, 0], [0, 0]],
            "gripper_mode_action": [[0, 1], [1, 0], [0, 0]],
            "gripper_activity_action": [[0, 1], [1, 0], [0, 0]],
        }
    )
    pq.write_table(table, dataset_root / "data" / "chunk-000" / "episode_000000.parquet")
    return dataset_root


def test_robocoin_language_action_generation_writes_episode_files_and_metadata(tmp_path):
    dataset_root = _make_tiny_robocoin_dataset(tmp_path)

    module = importlib.import_module("data.lerobot.prepare_robocoin_lerobot")
    stats = module.ensure_language_action(
        dataset_root=dataset_root,
        folder_name="language_action",
        overwrite=True,
    )

    assert stats.updated == 1
    output = dataset_root / "language_action" / "episode_000000.txt"
    assert output.exists()
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "Current subtask: reach the item." in lines[0]
    assert "Left arm: move forward at slow speed, constant acceleration, gripper open, holding." in lines[0]
    assert "Right arm: move left, constant speed, accelerating, gripper closed, closing." in lines[0]

    episodes = [json.loads(line) for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert episodes[0]["language_action_path"] == "language_action/episode_000000.txt"


def test_dataset_factory_exposes_lerobot_robocoin_branch():
    source = Path("/root/nas/code/d0/data/dataset.py").read_text(encoding="utf-8")

    assert "lerobot_robocoin" in source
    assert "LeRobotRoboCOINDataset" in source
