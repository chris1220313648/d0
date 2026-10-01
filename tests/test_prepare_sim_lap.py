"""Small, CPU-only regression checks for simulation sidecar generation."""
import json
from pathlib import Path
import tempfile
import sys

import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.lerobot.prepare_sim_lap import describe, gr1_source, instruction, la_valid, write_rows, selected_rows, validate_frames


def test_motion_and_window():
    xyz = np.zeros((20, 3))
    xyz[:, 0] = np.arange(20) * .01
    rot = Rotation.identity(20)
    lines = describe([("Arm", xyz, rot, None)], 16)
    aligned = describe([("Arm", xyz, rot, None)], 17)
    assert aligned[0] == "Arm: move +X 16.0 cm."
    assert aligned[-1] == "Arm: hold position."
    assert len(lines) == 20
    assert lines[0] == "Arm: move +X 15.0 cm."
    assert lines[-2] == "Arm: move +X 1.0 cm."
    assert lines[-1] == "Arm: hold position."
    assert describe([("Arm", xyz[:1], rot[:1], None)], 16) == ["Arm: hold position."]


def test_rotation_boundary_and_gripper():
    xyz = np.zeros((2, 3))
    rot = Rotation.from_euler("z", [[179], [-179]], degrees=True)
    lines = describe([("Arm", xyz, rot, np.array([.01, .04]))], 16)
    assert lines[0] == "Arm: rotate 2 degrees about +Z, open gripper."
    assert lines[1] == "Arm: hold position."


def test_metadata_and_resume_validation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [{"episode_index": i, "keep": i} for i in range(3)]
        write_rows(root / "meta/episodes.jsonl", rows)
        all_rows, selected = selected_rows(root, 1)
        selected[0]["language_action_path"] = "a.txt"
        write_rows(root / "meta/episodes.jsonl", all_rows)
        assert len(selected_rows(root, 0)[0]) == 3
        assert all_rows[2] == rows[2]
        txt = root / "a.txt"
        txt.write_text("Arm: hold position.\n")
        assert la_valid(txt, 1)
        assert not la_valid(txt, 2)
        assert not la_valid(txt, 1, "wrong hash")
    row = {"episode_index": 0, "remarks": "full instruction", "tasks": ["short"], "trajectory_id": "Task-00168"}
    assert instruction("gr1", row) == "full instruction"
    assert instruction("xemb", row) == "short"
    path, demo = gr1_source(Path("/a/LeRobot_v21/task"), row)
    assert path == Path("/a/HDF5/Task.hdf5") and demo == "demo_168"
    try:
        instruction("gr1", {"episode_index": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("Empty instruction accepted")


def test_legacy_frame_alignment():
    row = {"episode_index": 189, "length": 3}
    table = pa.table({"episode_index": [189]*3, "index": [40229, 40230, 40231], "timestamp": [0., .05, .10]})
    validate_frames(table, row, 20., legacy=True)
    bad = table.set_column(1, "index", pa.array([40229, 40231, 40230]))
    try:
        validate_frames(bad, row, 20., legacy=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Scrambled trajectory accepted")


if __name__ == "__main__":
    test_motion_and_window()
    test_rotation_boundary_and_gripper()
    test_metadata_and_resume_validation()
    test_legacy_frame_alignment()
    print("4 regression checks passed")
