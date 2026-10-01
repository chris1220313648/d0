import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.utils.per_task_norm import PerTaskNormalizer


def _write_stats(path):
    signal = {
        "min": [-10.0] * 14,
        "max": [10.0] * 14,
        "q01": [-1.0] * 14,
        "q99": [1.0] * 14,
        "active_mask": [True] * 14,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "robocoin",
                "tasks": {"nested/task": {"state": signal, "action": signal}},
            }
        ),
        encoding="utf-8",
    )


def test_per_task_q01_q99_clips_and_respects_mask(tmp_path):
    path = tmp_path / "robocoin.json"
    _write_stats(path)
    normalizer = PerTaskNormalizer(str(path), "robocoin", "q01_q99")
    values = torch.tensor([-10.0, 0.0, 10.0] + [0.0] * 11)
    mask = torch.ones(14, dtype=torch.bool)
    mask[-1] = False

    normalized = normalizer.normalize("nested/task", "action", values, mask)

    assert normalized[:3].tolist() == [0.0, 0.5, 1.0]
    assert normalized[-1].item() == 0.0


def test_per_task_normalizer_has_no_dataset_fallback(tmp_path):
    path = tmp_path / "robocoin.json"
    _write_stats(path)
    normalizer = PerTaskNormalizer(str(path), "robocoin")

    with pytest.raises(KeyError, match="Missing per-task stats"):
        normalizer.normalize(
            "missing/task", "action", torch.zeros(14), torch.ones(14, dtype=torch.bool)
        )
