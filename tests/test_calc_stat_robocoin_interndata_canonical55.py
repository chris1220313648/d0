import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_build_robocoin_task_stats_maps_named_fields_to_canonical55():
    from data.utils.calc_stat_robocoin_interndata_canonical55 import build_robocoin_task_stats

    info = {
        "features": {
            "observation.state": {
                "names": [
                    "left_eef_pos_x_m",
                    "left_eef_pos_y_m",
                    "left_eef_pos_z_m",
                    "left_gripper_open",
                ]
            },
            "action": {
                "names": [
                    "right_eef_pos_x_m",
                    "right_eef_pos_y_m",
                    "right_eef_pos_z_m",
                    "right_gripper_open",
                ]
            },
        }
    }
    rows = [
        {
            "stats": {
                "observation.state": {
                    "min": [-1.0, -2.0, -3.0, 900.0],
                    "max": [1.0, 2.0, 3.0, 1000.0],
                    "count": [10],
                },
                "action": {
                    "min": [-4.0, -5.0, -6.0, 800.0],
                    "max": [4.0, 5.0, 6.0, 1100.0],
                    "count": [10],
                },
            }
        }
    ]

    result = build_robocoin_task_stats(info, rows)

    assert result["state"]["min"][14:17] == [-1.0, -2.0, -3.0]
    assert result["state"]["max"][26] == 1000.0
    assert result["action"]["min"][20:23] == [-4.0, -5.0, -6.0]
    assert result["action"]["max"][27] == 1100.0
    assert result["state"]["active_mask"][14:17] == [True, True, True]
    assert result["action"]["active_mask"][27]


def test_build_interndata_task_stats_maps_columns_and_falls_back_to_actions_for_state():
    from data.utils.calc_stat_robocoin_interndata_canonical55 import build_interndata_task_stats

    rows = [
        {
            "stats": {
                "actions.joint.position": {
                    "min": [-2.0] * 7,
                    "max": [2.0] * 7,
                    "count": [20],
                },
                "actions.gripper.position": {
                    "min": [0.1],
                    "max": [0.9],
                    "count": [20],
                },
            }
        }
    ]

    result = build_interndata_task_stats(rows)

    assert result["action"]["min"][0:7] == [-2.0] * 7
    assert result["action"]["max"][26] == pytest.approx(0.9)
    assert result["state"]["min"][0:7] == [-2.0] * 7
    assert result["state"]["max"][26] == pytest.approx(0.9)
    assert result["state"]["source_fallback_rows"] == 1


def test_merge_stats_file_preserves_existing_entries(tmp_path):
    from data.utils.calc_stat_robocoin_interndata_canonical55 import merge_stats_file

    path = tmp_path / "stat.json"
    path.write_text(json.dumps({"agibot": {"sentinel": 1}}), encoding="utf-8")
    updates = {
        "robocoin": {
            "state": {"min": [0.0] * 55, "max": [1.0] * 55, "active_mask": [True] * 55}
        }
    }

    merge_stats_file(path, updates)
    merged = json.loads(path.read_text(encoding="utf-8"))

    assert merged["agibot"] == {"sentinel": 1}
    assert merged["robocoin"] == updates["robocoin"]


def test_stats_builder_cli_can_run_from_repo_root():
    result = subprocess.run(
        [
            sys.executable,
            "data/utils/calc_stat_robocoin_interndata_canonical55.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--robocoin-root" in result.stdout


def test_filtered_episode_stats_reader_only_decodes_requested_features(tmp_path):
    from data.utils.calc_stat_robocoin_interndata_canonical55 import (
        read_filtered_episode_stats,
    )

    path = tmp_path / "episodes_stats.jsonl"
    path.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "stats": {
                    "observation.images.main": {
                        "min": [[[0.0]], [[0.0]], [[0.0]]],
                        "max": [[[1.0]], [[1.0]], [[1.0]]],
                    },
                    "action": {"min": [1.0, 2.0], "max": [3.0, 4.0], "count": [8]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_filtered_episode_stats(path, ["action"])

    assert rows == [
        {"stats": {"action": {"min": [1.0, 2.0], "max": [3.0, 4.0], "count": [8]}}}
    ]
