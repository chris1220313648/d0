import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_stats(path: Path) -> None:
    active_mask = [False] * 55
    active_mask[0] = True
    active_mask[26] = True
    minimum = [0.0] * 55
    maximum = [0.0] * 55
    minimum[0], maximum[0] = -2.0, 2.0
    minimum[26], maximum[26] = 1000.0, 1000.0
    path.write_text(
        json.dumps(
            {
                "robocoin": {
                    "state": {
                        "min": minimum,
                        "max": maximum,
                        "active_mask": active_mask,
                    },
                    "action": {
                        "min": minimum,
                        "max": maximum,
                        "active_mask": active_mask,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_load_and_normalize_masked_canonical_stats(tmp_path):
    from data.utils.norm import load_masked_normalization_stats, normalize_masked

    stats_path = tmp_path / "stat.json"
    _write_stats(stats_path)
    minimum, maximum, active_mask = load_masked_normalization_stats(
        str(stats_path), "robocoin", "action"
    )
    values = torch.zeros(2, 55)
    values[:, 0] = torch.tensor([-4.0, 2.0])
    values[:, 26] = 1000.0
    valid_mask = torch.zeros_like(values, dtype=torch.bool)
    valid_mask[:, [0, 26]] = True

    normalized = normalize_masked(values, valid_mask, minimum, maximum, active_mask)

    assert normalized[:, 0].tolist() == [0.0, 1.0]
    assert normalized[:, 26].tolist() == [0.0, 0.0]
    assert not normalized[:, 1:26].any()


def test_masked_normalization_round_trip_on_nonconstant_dimensions(tmp_path):
    from data.utils.norm import (
        denormalize_masked,
        load_masked_normalization_stats,
        normalize_masked,
    )

    stats_path = tmp_path / "stat.json"
    _write_stats(stats_path)
    minimum, maximum, active_mask = load_masked_normalization_stats(
        str(stats_path), "robocoin", "state"
    )
    values = torch.zeros(55)
    values[0] = 0.5
    valid_mask = torch.zeros(55, dtype=torch.bool)
    valid_mask[0] = True

    normalized = normalize_masked(values, valid_mask, minimum, maximum, active_mask)
    restored = denormalize_masked(normalized, valid_mask, minimum, maximum, active_mask)

    assert torch.allclose(restored[valid_mask], values[valid_mask])
    assert not restored[~valid_mask].any()


def test_masked_normalization_rejects_valid_dimension_without_stats(tmp_path):
    from data.utils.norm import load_masked_normalization_stats, normalize_masked

    stats_path = tmp_path / "stat.json"
    _write_stats(stats_path)
    minimum, maximum, active_mask = load_masked_normalization_stats(
        str(stats_path), "robocoin", "action"
    )
    values = torch.zeros(55)
    valid_mask = torch.zeros(55, dtype=torch.bool)
    valid_mask[3] = True

    with pytest.raises(ValueError, match="valid dimensions missing normalization stats"):
        normalize_masked(values, valid_mask, minimum, maximum, active_mask)


def test_masked_stats_loader_rejects_wrong_dimension(tmp_path):
    from data.utils.norm import load_masked_normalization_stats

    stats_path = tmp_path / "stat.json"
    stats_path.write_text(
        json.dumps(
            {
                "robocoin": {
                    "action": {
                        "min": [0.0, 0.0],
                        "max": [1.0, 1.0],
                        "active_mask": [True, True],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 55"):
        load_masked_normalization_stats(str(stats_path), "robocoin", "action")


def test_masked_normalization_rejects_nonfinite_valid_values():
    from data.utils.norm import normalize_masked

    values = torch.tensor([float("nan")] + [0.0] * 54)
    valid_mask = torch.tensor([True] + [False] * 54)
    minimum = np.zeros(55, dtype=np.float32)
    maximum = np.ones(55, dtype=np.float32)
    active_mask = np.ones(55, dtype=np.bool_)

    with pytest.raises(ValueError, match="non-finite"):
        normalize_masked(values, valid_mask, minimum, maximum, active_mask)
