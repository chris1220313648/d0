import json
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _quat_z(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _active21(left_xyz=(0.0, 0.0, 0.0), right_xyz=(0.0, 0.0, 0.0), left_yaw=0.0, right_yaw=0.0):
    return [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        *left_xyz,
        *_quat_z(left_yaw),
        *right_xyz,
        *_quat_z(right_yaw),
    ]


def test_active21_language_action_uses_64_frame_horizon():
    from data.human_data.generate_active_wrist_language_action import build_active_wrist_language_action_lines

    states = np.asarray(
        [
            _active21(
                left_xyz=(0.01 * idx, 0.0, 0.0),
                right_xyz=(0.0, -0.005 * idx, 0.0),
                left_yaw=float(idx),
            )
            for idx in range(65)
        ],
        dtype=np.float32,
    )

    lines = build_active_wrist_language_action_lines(states, window_size=64)

    assert len(lines) == 65
    assert "Left wrist: move forward 64 cm" in lines[0]
    assert "rotate counterclockwise 64 degrees" in lines[0]
    assert "Right wrist: move right 32 cm" in lines[0]


def test_active21_language_action_reports_hold_position_for_static_window():
    from data.human_data.generate_active_wrist_language_action import build_active_wrist_language_action_lines

    states = np.asarray([_active21() for _ in range(3)], dtype=np.float32)

    lines = build_active_wrist_language_action_lines(states, window_size=64)

    assert lines == ["hold position", "hold position", "hold position"]


def test_process_manifest_writes_sidecar_language_action(tmp_path):
    from data.human_data.generate_active_wrist_language_action import process_manifest

    parquet_path = tmp_path / "episode.parquet"
    states = np.asarray(
        [_active21(left_xyz=(0.01 * idx, 0.0, 0.0)) for idx in range(5)],
        dtype=np.float32,
    )
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array([0, 1, 2, 3, 4]),
                "observation.state": pa.array(states.tolist()),
            }
        ),
        parquet_path,
    )
    manifest_path = tmp_path / "manifests" / "train.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_a",
                "data_parquet": str(parquet_path),
                "start_frame": 1,
                "end_frame": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = process_manifest(
        manifest_path,
        output_root=tmp_path / "language_action",
        split="train",
        window_size=64,
        overwrite=False,
    )

    output_path = tmp_path / "language_action" / "train" / "sample_a.txt"
    assert stats.rows_seen == 1
    assert stats.files_written == 1
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "Left wrist: move forward 2 cm" in lines[0]


def test_process_manifest_treats_start_frame_as_local_when_dataset_indices_exist(tmp_path):
    from data.human_data.generate_active_wrist_language_action import process_manifest

    parquet_path = tmp_path / "episode.parquet"
    states = np.asarray(
        [_active21(left_xyz=(0.01 * idx, 0.0, 0.0)) for idx in range(8)],
        dtype=np.float32,
    )
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array([100, 101, 102, 103, 104, 105, 106, 107]),
                "episode_index": pa.array([7, 7, 7, 7, 7, 7, 7, 7]),
                "observation.state": pa.array(states.tolist()),
            }
        ),
        parquet_path,
    )
    manifest_path = tmp_path / "manifests" / "val.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_b",
                "data_parquet": str(parquet_path),
                "dataset_from_index": 1002,
                "dataset_to_index": 1007,
                "episode_index": 7,
                "start_frame": 100,
                "end_frame": 103,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = process_manifest(
        manifest_path,
        output_root=tmp_path / "language_action",
        split="val",
        window_size=64,
        overwrite=False,
    )

    output_path = tmp_path / "language_action" / "val" / "sample_b.txt"
    assert stats.files_written == 1
    assert stats.rows_failed == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "Left wrist: move forward 2 cm" in lines[0]


def test_process_manifest_reads_shared_parquet_once(tmp_path, monkeypatch):
    from data.human_data import generate_active_wrist_language_action as generator

    parquet_path = tmp_path / "shared.parquet"
    states = np.asarray(
        [_active21(left_xyz=(0.01 * idx, 0.0, 0.0)) for idx in range(8)],
        dtype=np.float32,
    )
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array([0, 1, 2, 3, 0, 1, 2, 3]),
                "episode_index": pa.array([0, 0, 0, 0, 1, 1, 1, 1]),
                "observation.state": pa.array(states.tolist()),
            }
        ),
        parquet_path,
    )
    manifest_path = tmp_path / "manifests" / "train.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"sample_{episode_idx}",
                    "data_parquet": str(parquet_path),
                    "episode_index": episode_idx,
                    "start_frame": 0,
                    "end_frame": 4,
                }
            )
            for episode_idx in (0, 1)
        )
        + "\n",
        encoding="utf-8",
    )

    original_read_table = generator.pq.read_table
    read_count = {"value": 0}

    def counted_read_table(*args, **kwargs):
        read_count["value"] += 1
        return original_read_table(*args, **kwargs)

    monkeypatch.setattr(generator.pq, "read_table", counted_read_table)

    stats = generator.process_manifest(
        manifest_path,
        output_root=tmp_path / "language_action",
        split="train",
        window_size=64,
        overwrite=False,
    )

    assert stats.files_written == 2
    assert stats.rows_failed == 0
    assert read_count["value"] == 1


def test_process_manifest_parallel_matches_serial_output(tmp_path):
    from data.human_data import generate_active_wrist_language_action as generator

    parquet_path = tmp_path / "shared.parquet"
    states = np.asarray(
        [_active21(left_xyz=(0.01 * idx, 0.0, 0.0), right_xyz=(0.0, 0.01 * idx, 0.0)) for idx in range(12)],
        dtype=np.float32,
    )
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array([0, 1, 2, 3] * 3),
                "episode_index": pa.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
                "observation.state": pa.array(states.tolist()),
            }
        ),
        parquet_path,
    )
    manifest_path = tmp_path / "manifests" / "train.jsonl"
    manifest_path.parent.mkdir()
    rows = [
        {
            "id": f"sample_{episode_idx}",
            "data_parquet": str(parquet_path),
            "episode_index": episode_idx,
            "start_frame": 0,
            "end_frame": 4,
        }
        for episode_idx in (0, 1, 2)
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    serial_stats = generator.process_manifest(
        manifest_path,
        output_root=tmp_path / "serial_language_action",
        split="train",
        window_size=2,
        overwrite=False,
    )
    parallel_stats = generator.process_manifest(
        manifest_path,
        output_root=tmp_path / "parallel_language_action",
        split="train",
        window_size=2,
        overwrite=False,
        workers=2,
    )

    assert parallel_stats.rows_seen == serial_stats.rows_seen == 3
    assert parallel_stats.files_written == serial_stats.files_written == 3
    assert parallel_stats.rows_failed == serial_stats.rows_failed == 0
    for row in rows:
        serial_text = (tmp_path / "serial_language_action" / "train" / f"{row['id']}.txt").read_text(encoding="utf-8")
        parallel_text = (tmp_path / "parallel_language_action" / "train" / f"{row['id']}.txt").read_text(encoding="utf-8")
        assert parallel_text == serial_text


def test_parse_args_accepts_workers():
    from data.human_data import generate_active_wrist_language_action as generator

    with mock.patch.object(sys, "argv", ["generate", "--manifest", "train.jsonl", "--workers", "4"]):
        args = generator._parse_args()

    assert args.workers == 4


def test_write_reports_uses_shared_report_root_when_provided(tmp_path):
    from data.human_data import generate_active_wrist_language_action as generator

    processed_root = tmp_path / "processed"
    report_root = tmp_path / "sidecar_output"
    stats = generator.ManifestStats(
        manifest=str(processed_root / "manifests" / "train.jsonl"),
        output_root=str(report_root),
        rows_seen=1,
        files_written=1,
        lines_written=3,
        failures=[],
    )

    generator._write_reports({processed_root: [stats]}, report_root=report_root)

    report_path = report_root / "reports" / "active_wrist_language_action_summary.json"
    assert report_path.exists()
    assert not (processed_root / "reports").exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["totals"]["files_written"] == 1
