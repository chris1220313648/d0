import numpy as np
import pytest
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.libero_plus.policy_server import LiberoMotusPolicy


class RecordingNormalizer:
    def __init__(self):
        self.received = None
        self.clip = None

    def denormalize(self, actions, clip=True):
        self.received = np.asarray(actions, dtype=np.float32).copy()
        self.clip = clip
        return self.received + 100.0

    def normalize(self, actions):
        self.received = np.asarray(actions, dtype=np.float32).copy()
        return self.received + 0.5


def make_policy(action_dim, horizon=2):
    policy = LiberoMotusPolicy.__new__(LiberoMotusPolicy)
    policy.model_action_dim = action_dim
    policy.prediction_horizon = horizon
    policy.action_normalizer = RecordingNormalizer()
    policy.flow_source_mode = "history"
    policy.history_action_length = horizon
    policy.device = torch.device("cpu")
    policy.dtype = torch.float32
    return policy


def test_denormalizes_first_seven_dims_from_14d_model_output():
    policy = make_policy(action_dim=14)
    predicted = torch.arange(28, dtype=torch.float32).reshape(1, 2, 14)

    raw_actions = policy._denormalize_libero_plus_actions(predicted)

    expected_normalized = predicted.numpy()[..., :7]
    assert np.array_equal(policy.action_normalizer.received, expected_normalized)
    assert policy.action_normalizer.clip is True
    assert np.array_equal(raw_actions, expected_normalized + 100.0)


def test_denormalizes_7d_model_output_without_cropping():
    policy = make_policy(action_dim=7)
    predicted = torch.arange(14, dtype=torch.float32).reshape(1, 2, 7)

    raw_actions = policy._denormalize_libero_plus_actions(predicted)

    assert np.array_equal(policy.action_normalizer.received, predicted.numpy())
    assert np.array_equal(raw_actions, predicted.numpy() + 100.0)


def test_rejects_model_output_shape_that_does_not_match_model_action_dim():
    policy = make_policy(action_dim=14)
    predicted = torch.zeros(1, 2, 7)

    with pytest.raises(ValueError, match=r"Expected predicted actions \(1, 2, 14\)"):
        policy._denormalize_libero_plus_actions(predicted)


def test_rejects_unsupported_model_action_dim():
    with pytest.raises(ValueError, match="LIBERO-plus supports 7D or 14D"):
        LiberoMotusPolicy._validate_model_action_dim(8)


def test_prepares_normalized_7d_history_for_7d_model():
    policy = make_policy(action_dim=7)
    history = np.arange(14, dtype=np.float32).reshape(2, 7)

    action_source = policy._prepare_action_source({"history_actions": history})

    assert action_source.shape == (1, 2, 7)
    np.testing.assert_allclose(action_source.numpy()[0], history + 0.5)


def test_pads_normalized_history_for_14d_model():
    policy = make_policy(action_dim=14)
    history = np.arange(14, dtype=np.float32).reshape(2, 7)

    action_source = policy._prepare_action_source({"history_actions": history})

    np.testing.assert_allclose(action_source.numpy()[0, :, :7], history + 0.5)
    np.testing.assert_array_equal(action_source.numpy()[0, :, 7:], 0.0)


def test_rejects_history_with_wrong_shape():
    policy = make_policy(action_dim=7)

    with pytest.raises(ValueError, match=r"Expected LIBERO history_actions \(2, 7\)"):
        policy._prepare_action_source({"history_actions": np.zeros((1, 7), dtype=np.float32)})


def test_gaussian_policy_does_not_require_history():
    policy = make_policy(action_dim=7)
    policy.flow_source_mode = "gaussian"

    assert policy._prepare_action_source({}) is None
