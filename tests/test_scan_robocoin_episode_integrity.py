import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


def _write_parquet(
    task_root: Path,
    episode_index: int = 0,
    *,
    columns: dict | None = None,
) -> Path:
    parquet_path = (
        task_root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = {
            "observation.state": [
                [0.0, 0.1],
                [0.2, 0.3],
                [0.4, 0.5],
                [0.6, 0.7],
            ],
            "action": [
                [1.0, 1.1],
                [1.2, 1.3],
                [1.4, 1.5],
                [1.6, 1.7],
            ],
            "timestamp": [0.0, 1 / 30, 2 / 30, 3 / 30],
            "frame_index": [0, 1, 2, 3],
            "episode_index": [episode_index] * 4,
        }
    pq.write_table(pa.table(columns), parquet_path)
    return parquet_path


def _write_video(path: Path, frame_count: int, fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            image = np.full((16, 16, 3), index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _default_video_path(task_root: Path, episode_index: int = 0) -> Path:
    return (
        task_root
        / "videos"
        / f"chunk-{episode_index // 1000:03d}"
        / "observation.images.cam_high_rgb"
        / f"episode_{episode_index:06d}.mp4"
    )


def _discover_one(root: Path):
    from scripts import scan_robocoin_episode_integrity as scanner

    episodes, failures = scanner.discover_episodes(root, None, None)
    assert failures == []
    assert len(episodes) == 1
    return scanner, episodes[0]


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


def test_resolve_episode_path_does_not_stat_target(tmp_path, monkeypatch):
    from scripts import scan_robocoin_episode_integrity as scanner

    def reject_resolve(self, *args, **kwargs):
        raise AssertionError("Path.resolve performs a NAS metadata lookup")

    monkeypatch.setattr(Path, "resolve", reject_resolve)

    result = scanner.resolve_episode_path(
        tmp_path,
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        3,
        1000,
    )

    assert result == tmp_path / "data/chunk-000/episode_000003.parquet"


def test_validate_parquet_accepts_valid_episode_and_maps_canonical55(tmp_path):
    task_root = _make_task(
        tmp_path,
        "task",
        [{"episode_index": 0, "length": 4}],
    )
    _write_parquet(task_root)
    scanner, spec = _discover_one(tmp_path)

    measurement, timestamps, issues = scanner.validate_parquet(spec)

    assert issues == []
    assert measurement is not None
    assert measurement.rows == 4
    assert measurement.timestamp_min == 0.0
    assert measurement.frame_index_max == 3
    assert timestamps == pytest.approx([0.0, 1 / 30, 2 / 30, 3 / 30])


def test_validate_parquet_accumulates_length_frame_and_timestamp_errors(tmp_path):
    task_root = _make_task(
        tmp_path,
        "task",
        [{"episode_index": 0, "length": 5}],
    )
    _write_parquet(
        task_root,
        columns={
            "observation.state": [[0.0, 0.1]] * 4,
            "action": [[1.0, 1.1]] * 4,
            "timestamp": [0.0, 0.2, 0.1, np.nan],
            "frame_index": [0, 2, 1, 3],
            "episode_index": [0] * 4,
        },
    )
    scanner, spec = _discover_one(tmp_path)

    measurement, _, issues = scanner.validate_parquet(spec)

    assert measurement is not None
    assert {issue.code for issue in issues} >= {
        "episode_length_mismatch",
        "invalid_frame_index",
        "invalid_timestamp",
    }


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("missing", "missing_parquet"),
        ("corrupt", "invalid_parquet"),
        ("missing_action", "missing_action"),
        ("wrong_episode", "episode_index_mismatch"),
    ],
)
def test_validate_parquet_reports_structural_failures(
    tmp_path,
    mode,
    expected_code,
):
    task_root = _make_task(
        tmp_path,
        "task",
        [{"episode_index": 0, "length": 4}],
    )
    if mode == "corrupt":
        path = task_root / "data/chunk-000/episode_000000.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not parquet")
    elif mode == "missing_action":
        _write_parquet(
            task_root,
            columns={
                "observation.state": [[0.0, 0.1]] * 4,
                "timestamp": [0.0, 1 / 30, 2 / 30, 3 / 30],
                "frame_index": [0, 1, 2, 3],
                "episode_index": [0] * 4,
            },
        )
    elif mode == "wrong_episode":
        _write_parquet(
            task_root,
            columns={
                "observation.state": [[0.0, 0.1]] * 4,
                "action": [[1.0, 1.1]] * 4,
                "timestamp": [0.0, 1 / 30, 2 / 30, 3 / 30],
                "frame_index": [0, 1, 2, 3],
                "episode_index": [9] * 4,
            },
        )
    scanner, spec = _discover_one(tmp_path)

    _, _, issues = scanner.validate_parquet(spec)

    assert expected_code in {issue.code for issue in issues}


def test_validate_auxiliary_accepts_missing_optional_language_action(tmp_path):
    _make_task(tmp_path, "task", [{"episode_index": 0, "length": 4}])
    scanner, spec = _discover_one(tmp_path)

    assert scanner.validate_auxiliary(spec) == []


def test_validate_auxiliary_rejects_declared_missing_files(tmp_path):
    _make_task(
        tmp_path,
        "task",
        [
            {
                "episode_index": 0,
                "length": 4,
                "language_action_path": "language_action/episode_000000.txt",
                "t5_embedding_path": "t5_embedding/episode_000000.pt",
            }
        ],
    )
    scanner, spec = _discover_one(tmp_path)

    assert {issue.code for issue in scanner.validate_auxiliary(spec)} == {
        "missing_declared_language_action",
        "missing_declared_t5_embedding",
    }


def test_validate_auxiliary_rejects_empty_present_language_action(tmp_path):
    task_root = _make_task(
        tmp_path,
        "task",
        [{"episode_index": 0, "length": 4}],
    )
    path = task_root / "language_action/episode_000000.txt"
    path.parent.mkdir()
    path.write_text("\n", encoding="utf-8")
    scanner, spec = _discover_one(tmp_path)

    assert [issue.code for issue in scanner.validate_auxiliary(spec)] == [
        "invalid_language_action"
    ]


def test_scan_episode_fully_decodes_valid_video(tmp_path):
    task_root = _make_task(
        tmp_path,
        "task",
        [{"episode_index": 0, "length": 4}],
    )
    _write_parquet(task_root)
    _write_video(_default_video_path(task_root), frame_count=4)
    scanner, spec = _discover_one(tmp_path)

    result = scanner.scan_episode(spec)

    assert result.status == "good"
    assert result.issues == ()
    assert result.videos[0]["decoded_frames"] == 4
    assert result.videos[0]["timestamp_max"] >= 3 / 30 - spec.tolerance_s


@pytest.mark.parametrize(
    ("mode", "expected_codes"),
    [
        ("missing", {"missing_video"}),
        ("corrupt", {"video_decode_error"}),
        (
            "short",
            {"video_frame_count_mismatch", "video_too_short"},
        ),
        ("wrong_fps", {"video_fps_mismatch"}),
    ],
)
def test_scan_episode_reports_video_failures(tmp_path, mode, expected_codes):
    task_root = _make_task(
        tmp_path,
        "task",
        [{"episode_index": 0, "length": 4}],
    )
    _write_parquet(task_root)
    video_path = _default_video_path(task_root)
    if mode == "corrupt":
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(b"not video")
    elif mode == "short":
        _write_video(video_path, frame_count=3)
    elif mode == "wrong_fps":
        _write_video(video_path, frame_count=4, fps=15)
    scanner, spec = _discover_one(tmp_path)

    result = scanner.scan_episode(spec)

    assert result.status == "bad"
    assert {issue.code for issue in result.issues} >= expected_codes


def test_scan_episode_decodes_every_declared_video_key(tmp_path):
    task_root = _make_task(
        tmp_path,
        "task",
        [{"episode_index": 0, "length": 4}],
    )
    info_path = task_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.cam_left_wrist_rgb"] = {"dtype": "video"}
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _write_parquet(task_root)
    _write_video(_default_video_path(task_root), frame_count=4)
    _write_video(
        task_root
        / "videos/chunk-000/observation.images.cam_left_wrist_rgb/"
        "episode_000000.mp4",
        frame_count=4,
    )
    scanner, spec = _discover_one(tmp_path)

    result = scanner.scan_episode(spec)

    assert result.status == "good"
    assert {video["key"] for video in result.videos} == {
        "observation.images.cam_high_rgb",
        "observation.images.cam_left_wrist_rgb",
    }


def _result(scanner, episode_index: int, status: str):
    issues = ()
    if status == "bad":
        issues = (
            scanner.Issue(
                code="missing_video",
                message="missing",
                path="/data/video.mp4",
            ),
        )
    return scanner.EpisodeResult(
        task="task",
        episode_index=episode_index,
        status=status,
        elapsed_s=0.1,
        parquet={"rows": 4},
        videos=(),
        issues=issues,
    )


def test_report_writer_flushes_terminal_results_and_summary(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    output = tmp_path / "reports"
    with scanner.ReportWriter(output, overwrite=True) as writer:
        writer.set_discovered(2)
        writer.record_episode(_result(scanner, 0, "good"))
        writer.record_episode(_result(scanner, 1, "bad"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        rows = (output / "episode_results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()

    assert len(rows) == 2
    assert summary["episodes_discovered"] == 2
    assert summary["episodes_scanned"] == 2
    assert summary["good"] == 1
    assert summary["bad"] == 1
    assert summary["errors_by_code"] == {"missing_video": 1}
    assert len((output / "good_episodes.jsonl").read_text().splitlines()) == 1
    assert len((output / "bad_episodes.jsonl").read_text().splitlines()) == 1


def test_report_writer_rejects_duplicate_terminal_key(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    with scanner.ReportWriter(tmp_path / "reports", overwrite=True) as writer:
        result = _result(scanner, 0, "good")
        writer.record_episode(result)
        with pytest.raises(ValueError, match="duplicate terminal result"):
            writer.record_episode(result)


def test_report_writer_resume_rejects_duplicate_task_failure(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    output = tmp_path / "reports"
    failure = scanner.TaskFailure(
        task="broken_task",
        code="invalid_task_metadata",
        message="bad metadata",
    )
    with scanner.ReportWriter(output, overwrite=True) as writer:
        writer.record_task_failure(failure)
    with scanner.ReportWriter(output, resume=True) as writer:
        with pytest.raises(ValueError, match="duplicate task failure"):
            writer.record_task_failure(failure)


def test_load_terminal_keys_ignores_truncated_tail(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    path = tmp_path / "episode_results.jsonl"
    path.write_text(
        '{"task":"task","episode_index":3,"status":"good","issues":[]}\n'
        '{"task":',
        encoding="utf-8",
    )

    assert scanner.load_terminal_keys(path) == {("task", 3)}


def test_cli_help_is_runnable():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/scan_robocoin_episode_integrity.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--resume" in result.stdout
    assert "--progress-interval" in result.stdout


def test_run_scan_writes_good_result_and_resume_skips_it(tmp_path):
    from scripts import scan_robocoin_episode_integrity as scanner

    root = tmp_path / "data"
    task_root = _make_task(
        root,
        "task",
        [{"episode_index": 0, "length": 4}],
    )
    _write_parquet(task_root)
    _write_video(_default_video_path(task_root), frame_count=4)
    output = tmp_path / "reports"
    args = SimpleNamespace(
        root=root,
        output_dir=output,
        workers=1,
        task=None,
        episode=None,
        resume=False,
        overwrite=True,
        progress_interval=1,
    )

    assert scanner.run_scan(args) == 0
    assert scanner.run_scan(
        replace(
            scanner.ScanArguments.from_namespace(args),
            resume=True,
            overwrite=False,
        )
    ) == 0

    rows = (output / "episode_results.jsonl").read_text().splitlines()
    summary = json.loads((output / "summary.json").read_text())
    assert len(rows) == 1
    assert summary["episodes_scanned"] == 1
    assert summary["episodes_skipped_resume"] == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"workers": 0},
        {"resume": True, "overwrite": True},
        {"progress_interval": 0},
    ],
)
def test_run_scan_rejects_invalid_arguments(tmp_path, updates):
    from scripts import scan_robocoin_episode_integrity as scanner

    values = {
        "root": tmp_path / "data",
        "output_dir": tmp_path / "reports",
        "workers": 1,
        "task": None,
        "episode": None,
        "resume": False,
        "overwrite": False,
        "progress_interval": 1,
    }
    values.update(updates)

    assert scanner.run_scan(SimpleNamespace(**values)) == 2


def test_run_scan_persists_summary_when_discovery_is_interrupted(
    tmp_path,
    monkeypatch,
):
    from scripts import scan_robocoin_episode_integrity as scanner

    root = tmp_path / "data"
    root.mkdir()
    output = tmp_path / "reports"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(scanner, "discover_episodes", interrupt)
    args = SimpleNamespace(
        root=root,
        output_dir=output,
        workers=1,
        task=None,
        episode=None,
        resume=False,
        overwrite=True,
        progress_interval=1,
    )

    assert scanner.run_scan(args) == 130
    assert (output / "summary.json").is_file()
