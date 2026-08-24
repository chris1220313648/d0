import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fixed_size_list_array(rows):
    values = pa.array([float(v) for row in rows for v in row], type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, len(rows[0]))


def _write_synthetic_agibot_split(root: Path) -> Path:
    split_root = root / "task_split"
    meta_dir = split_root / "meta"
    data_dir = split_root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    state_fields = {
        "state/left_effector/position": {"indices": [0]},
        "state/right_effector/position": {"indices": [1]},
        "state/end/arm_orientation": {"indices": [26, 27, 28, 29, 30, 31, 32, 33]},
        "state/end/arm_position": {"indices": [34, 35, 36, 37, 38, 39]},
        "state/joint/position": {"indices": list(range(40, 54))},
        "state/head/position": {"indices": [82, 83, 84]},
        "state/waist/position": {"indices": [85, 86, 87, 88, 89]},
    }
    action_fields = {
        "action/left_effector/position": {"indices": [0]},
        "action/right_effector/position": {"indices": [1]},
        "action/end/position": {"indices": [2, 3, 4, 5, 6, 7]},
        "action/end/orientation": {"indices": [8, 9, 10, 11, 12, 13, 14, 15]},
        "action/joint/position": {"indices": list(range(16, 30))},
        "action/head/position": {"indices": [30, 31, 32]},
        "action/waist/position": {"indices": [33, 34, 35, 36, 37]},
        "action/robot/velocity": {"indices": [38, 39, 40, 41, 42, 43]},
    }
    info = {
        "total_episodes": 1,
        "total_frames": 3,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [162],
                "field_descriptions": state_fields,
            },
            "action": {
                "dtype": "float32",
                "shape": [44],
                "field_descriptions": action_fields,
            },
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    state_min = [-1000.0] * 162
    state_max = [1000.0] * 162
    action_min = [-2000.0] * 44
    action_max = [2000.0] * 44
    for i in range(40, 54):
        state_min[i] = float(i)
        state_max[i] = float(i + 100)
    for i in range(16, 30):
        action_min[i] = float(i)
        action_max[i] = float(i + 100)
    action_min[38] = -0.5
    action_max[38] = 0.5
    action_min[39] = -0.25
    action_max[39] = 0.25
    stat_row = {
        "episode_index": 0,
        "stats": {
            "observation.state": {
                "min": state_min,
                "max": state_max,
                "mean": [0.0] * 162,
                "std": [1.0] * 162,
                "count": [3],
            },
            "action": {
                "min": action_min,
                "max": action_max,
                "mean": [0.0] * 44,
                "std": [1.0] * 44,
                "count": [3],
            },
        },
    }
    (meta_dir / "episodes_stats.jsonl").write_text(json.dumps(stat_row) + "\n", encoding="utf-8")

    quat_identity = [0.0, 0.0, 0.0, 1.0]
    quat_z90 = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    state_rows = []
    action_rows = []
    for frame_idx, quat in enumerate([quat_identity, quat_z90, quat_identity]):
        state = [0.0] * 162
        state[26:30] = quat
        state[30:34] = quat_identity
        action = [0.0] * 44
        action[8:12] = quat
        action[12:16] = quat_identity
        state_rows.append(state)
        action_rows.append(action)

    table = pa.table(
        {
            "observation.state": _fixed_size_list_array(state_rows),
            "action": _fixed_size_list_array(action_rows),
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "index": pa.array([0, 1, 2], type=pa.int64()),
            "task_index": pa.array([0, 0, 0], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    return split_root


def test_agibot_stats_builds_55d_from_raw_stats_and_derived_orientation(tmp_path):
    from data.utils.calc_stat_agibot_canonical55 import compute_agibot_stats

    _write_synthetic_agibot_split(tmp_path)

    stats = compute_agibot_stats(
        tmp_path,
        suffix="_split",
        max_action_offset=1,
        rotvec_sample_conditions=100,
        seed=0,
    )

    state = stats["agibot"]["state"]
    action = stats["agibot"]["action"]
    assert len(state["min"]) == 55
    assert len(state["max"]) == 55
    assert len(action["min"]) == 55
    assert len(action["max"]) == 55

    assert state["min"][0:14] == [float(i) for i in range(40, 54)]
    assert state["max"][0:14] == [float(i + 100) for i in range(40, 54)]
    assert action["min"][0:14] == [float(i) for i in range(16, 30)]
    assert action["max"][0:14] == [float(i + 100) for i in range(16, 30)]
    assert action["min"][46:48] == [-0.5, -0.25]
    assert action["max"][46:48] == [0.5, 0.25]

    assert np.isclose(state["max"][19], math.pi / 2, atol=1e-5)
    assert np.isclose(action["max"][19], math.pi / 2, atol=1e-5)
    assert state["min"][28:40] == [0.0] * 12
    assert state["max"][28:40] == [1.0] * 12
    assert action["min"][49:55] == [0.0] * 6
    assert action["max"][49:55] == [1.0] * 6


def test_merge_agibot_stats_preserves_existing_stat_keys(tmp_path):
    from data.utils.calc_stat_agibot_canonical55 import merge_agibot_stats

    stat_path = tmp_path / "stat.json"
    stat_path.write_text(json.dumps({"droid": {"qpos": {"min": [0], "max": [1]}}}), encoding="utf-8")
    agibot = {
        "agibot": {
            "state": {"min": [0.0] * 55, "max": [1.0] * 55},
            "action": {"min": [0.0] * 55, "max": [1.0] * 55},
        }
    }

    merge_agibot_stats(stat_path, agibot)

    merged = json.loads(stat_path.read_text(encoding="utf-8"))
    assert merged["droid"]["qpos"] == {"min": [0], "max": [1]}
    assert merged["agibot"]["state"]["min"] == [0.0] * 55


def test_merge_agibot_stats_preserves_existing_file_mode(tmp_path):
    from data.utils.calc_stat_agibot_canonical55 import merge_agibot_stats

    stat_path = tmp_path / "stat.json"
    stat_path.write_text(json.dumps({"droid": {"qpos": {"min": [0], "max": [1]}}}), encoding="utf-8")
    stat_path.chmod(0o755)
    agibot = {
        "agibot": {
            "state": {"min": [0.0] * 55, "max": [1.0] * 55},
            "action": {"min": [0.0] * 55, "max": [1.0] * 55},
        }
    }

    merge_agibot_stats(stat_path, agibot)

    assert stat_path.stat().st_mode & 0o777 == 0o755


def test_agibot_stat_script_can_be_executed_by_path():
    script = REPO_ROOT / "data" / "utils" / "calc_stat_agibot_canonical55.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Build AgiBot canonical55" in result.stdout
