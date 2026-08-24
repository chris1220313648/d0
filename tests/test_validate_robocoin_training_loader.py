import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _good_sample():
    return {
        "first_frame": torch.zeros(3, 8, 8),
        "video_frames": torch.zeros(2, 3, 8, 8),
        "initial_state": torch.zeros(55),
        "state_mask": torch.ones(55, dtype=torch.bool),
        "action_sequence": torch.zeros(4, 55),
        "action_mask": torch.ones(4, 55, dtype=torch.bool),
        "language_embedding": torch.zeros(2, 3),
        "vlm_inputs": None,
        "canonical_format": "canonical55_v2",
    }


def test_validate_target_reports_sample_load_failure_and_continues():
    from data.lerobot.lerobot_robocoin_dataset import RoboCOINEpisodeTarget
    from scripts.validate_robocoin_training_loader import validate_target

    class BrokenDataset:
        def load_episode_sample(self, *_args, **_kwargs):
            raise FileNotFoundError("missing episode video")

    target = RoboCOINEpisodeTarget("task", 0, 2, 7)
    result = validate_target(
        BrokenDataset(),
        canonicalize=lambda sample: sample,
        target=target,
    )

    assert result.status == "bad"
    assert result.stage == "sample_load"
    assert result.error_type == "FileNotFoundError"
    assert "missing episode video" in result.message


def test_validate_target_runs_canonical_and_real_collate():
    from data.lerobot.lerobot_robocoin_dataset import RoboCOINEpisodeTarget
    from scripts.validate_robocoin_training_loader import validate_target

    class Dataset:
        def load_episode_sample(self, *_args, **_kwargs):
            return _good_sample()

    target = RoboCOINEpisodeTarget("task", 0, 0, 3)
    result = validate_target(
        Dataset(),
        canonicalize=lambda sample: sample,
        target=target,
    )

    assert result.status == "good"
    assert result.stage is None
    assert result.condition_frame_idx == "last"
    assert result.measurements["action_sequence"] == [1, 4, 55]
    assert result.measurements["initial_state"] == [1, 55]


def test_report_writer_resume_keeps_unique_terminal_episode_keys(tmp_path):
    from scripts.validate_robocoin_training_loader import ReportWriter, ValidationResult

    good = ValidationResult(
        task="task_a",
        episode_index=1,
        status="good",
        elapsed_s=0.1,
        condition_frame_idx="last",
        stage=None,
        error_type=None,
        message=None,
        traceback=None,
        measurements={"action_sequence": [1, 4, 55]},
    )
    with ReportWriter(tmp_path, overwrite=True, resume=False) as writer:
        writer.set_discovered(2)
        writer.record(good)

    with ReportWriter(tmp_path, overwrite=False, resume=True) as writer:
        assert writer.terminal_keys == {("task_a", 1)}
        writer.set_discovered(2)
        writer.record(
            ValidationResult(
                task="task_b",
                episode_index=2,
                status="bad",
                elapsed_s=0.2,
                condition_frame_idx="last",
                stage="collate",
                error_type="RuntimeError",
                message="bad batch",
                traceback="trace",
                measurements={},
            )
        )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["episodes_discovered"] == 2
    assert summary["episodes_scanned"] == 2
    assert summary["good"] == 1
    assert summary["bad"] == 1
    assert summary["errors_by_stage"] == {"collate": 1}
    rows = [json.loads(line) for line in (tmp_path / "episode_results.jsonl").read_text().splitlines()]
    assert {(row["task"], row["episode_index"]) for row in rows} == {
        ("task_a", 1),
        ("task_b", 2),
    }


def test_report_writer_rejects_existing_output_without_resume_or_overwrite(tmp_path):
    from scripts.validate_robocoin_training_loader import ReportWriter

    (tmp_path / "summary.json").write_text("{}")

    with pytest.raises(ValueError, match="resume or overwrite"):
        ReportWriter(tmp_path, overwrite=False, resume=False)


def test_prepare_validation_config_limits_real_smoke_to_requested_task(tmp_path):
    from scripts.validate_robocoin_training_loader import prepare_validation_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "type": "lerobot_robocoin",
                    "task_mode": "multi",
                    "task_name": None,
                    "max_episodes": 2,
                    "image_aug": True,
                    "params": {
                        "enable_t5_fallback": True,
                        "task_discovery": {"enabled": True},
                    },
                }
            }
        )
    )

    config = prepare_validation_config(config_path, task="task_a")

    assert config.dataset.task_name == "task_a"
    assert config.dataset.max_episodes is None
    assert config.dataset.image_aug is False
    assert config.dataset.params.enable_t5_fallback is False
    assert config.dataset.params.task_discovery.enabled is False


def test_discover_validation_tasks_matches_eager_training_discovery(tmp_path):
    from scripts.validate_robocoin_training_loader import discover_validation_tasks

    task = tmp_path / "task_a"
    (task / "meta").mkdir(parents=True)
    (task / "data" / "chunk-000").mkdir(parents=True)
    (task / "videos").mkdir()
    (task / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "chunks_size": 1000})
    )
    (task / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 3}) + "\n"
        + json.dumps({"episode_index": 1, "length": 4}) + "\n"
    )
    for episode_index in (0, 1):
        (task / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet").touch()
    config = OmegaConf.create(
        {
            "dataset": {
                "params": {
                    "root": str(tmp_path),
                    "task_discovery": {"enabled": True, "validate_episodes": True},
                }
            }
        }
    )

    assert discover_validation_tasks(config) == [("task_a", 2)]


def test_report_writer_resume_restores_task_failures_without_duplicates(tmp_path):
    from scripts.validate_robocoin_training_loader import ReportWriter

    with ReportWriter(tmp_path, overwrite=True, resume=False) as writer:
        writer.record_task_failure("task_a", RuntimeError("broken metadata"))

    with ReportWriter(tmp_path, overwrite=False, resume=True) as writer:
        assert writer.task_failure_keys == {"task_a"}
        assert writer.summary["task_failures"] == 1
        writer.record_task_failure("task_a", RuntimeError("still broken"))

    assert len((tmp_path / "task_failures.jsonl").read_text().splitlines()) == 1
