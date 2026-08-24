import json
import threading

import numpy as np
import pytest
import torch

from deploy import d0_protocol
from deploy import d0_policy as d0_policy_module
from deploy.d0_policy import D0Policy
from deploy.d0_preprocessing import D0Normalizer, TASK_SPECS, TaskSpec, resolve_task, stitch_three_views
from deploy.serve_d0 import load_deployment


def test_protocol_numpy_roundtrip():
    source = {"x": np.arange(12, dtype=np.float32).reshape(3, 4), "task_id": "pick_keys"}
    decoded = d0_protocol.unpackb(d0_protocol.packb(source))
    np.testing.assert_array_equal(decoded["x"], source["x"])
    assert decoded["task_id"] == "pick_keys"


def test_stitch_three_views_matches_layout():
    front = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)
    left = np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8)
    right = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)
    mosaic = stitch_three_views(front, left, right)
    assert mosaic.shape == (3, 384, 320)
    assert torch.allclose(mosaic[:, 50, 160], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(mosaic[:, 350, 50], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(mosaic[:, 350, 270], torch.tensor([0.0, 0.0, 1.0]))


def test_task_registry_is_strict():
    for task_id, spec in TASK_SPECS.items():
        assert resolve_task(task_id, spec.instruction) == spec
    try:
        resolve_task("pick_keys", "wrong prompt")
    except ValueError:
        pass
    else:
        raise AssertionError("prompt mismatch was not rejected")


def test_normalizer_roundtrip_shapes(tmp_path):
    stats = {
        "state": {"q01": [0.0] * 14, "q99": [1.0] * 14},
        "action": {"q01": [-1.0] * 14, "q99": [1.0] * 14},
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    normalizer = D0Normalizer(path)
    state = np.zeros(14, dtype=np.float32)
    state[[6, 13]] = 100.0
    normalized = normalizer.normalize_state(state)
    assert normalized[[6, 13]].tolist() == [1.0, 1.0]
    actions = normalizer.denormalize_action(np.full((48, 14), 0.5, dtype=np.float32))
    assert actions.shape == (48, 14)
    np.testing.assert_allclose(actions[:, :6], 0.0)
    history = normalizer.normalize_action(actions)
    np.testing.assert_allclose(history[:, :6], 0.5)


def test_load_deployment_resolves_checkpoint_relative_paths(tmp_path):
    checkpoint = tmp_path / "checkpoint_step_10"
    checkpoint.mkdir()
    (checkpoint / "training_config.yaml").write_text("common: {}\n", encoding="utf-8")
    (checkpoint / "deployment.yaml").write_text(
        "version: 1\nconfig: training_config.yaml\nstats: /tmp/stats.json\nnum_inference_steps: 4\n",
        encoding="utf-8",
    )
    deployment = load_deployment(str(checkpoint))
    assert deployment["config"] == str((checkpoint / "training_config.yaml").resolve())
    assert deployment["stats"] == "/tmp/stats.json"
    assert deployment["num_inference_steps"] == 4


def test_legacy_checkpoint_requires_explicit_config_and_stats(tmp_path):
    checkpoint = tmp_path / "checkpoint_step_10"
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match="legacy checkpoints"):
        load_deployment(str(checkpoint))


def test_history_actions_are_normalized_and_passed_as_action_source(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "state": {"q01": [0.0] * 14, "q99": [1.0] * 14},
                "action": {"q01": [-1.0] * 14, "q99": [1.0] * 14},
            }
        ),
        encoding="utf-8",
    )
    embedding_path = tmp_path / "task.pt"
    torch.save(torch.zeros((2, 4), dtype=torch.float32), embedding_path)
    captured = {}

    class FakeModel:
        def inference_step(self, **kwargs):
            captured.update(kwargs)
            return torch.empty(0), torch.full((1, 48, 14), 0.5)

    policy = object.__new__(D0Policy)
    policy.config = {
        "common": {"num_video_frames": 8, "video_action_freq_ratio": 6},
        "model": {"flow_source": {"mode": "history"}},
    }
    policy.normalizer = D0Normalizer(stats_path)
    policy.task_specs = {"test": TaskSpec("test", "do it", str(embedding_path))}
    policy._embedding_cache = {}
    policy._infer_lock = threading.Lock()
    policy.device = torch.device("cpu")
    policy.vlm_processor = object()
    policy.model = FakeModel()
    policy.num_inference_steps = 4
    policy.model_name = "test"
    monkeypatch.setattr(d0_policy_module, "stitch_three_views", lambda *args: torch.zeros((3, 4, 4)))
    monkeypatch.setattr(d0_policy_module, "tensor_to_pil", lambda image: object())
    monkeypatch.setattr(d0_policy_module, "preprocess_vlm_messages", lambda *args: {})

    result = policy.infer(
        {
            "state": np.zeros(14, dtype=np.float32),
            "front_camera": object(),
            "left_wrist_camera": object(),
            "right_wrist_camera": object(),
            "task_id": "test",
            "prompt": "do it",
            "history_actions": np.zeros((48, 14), dtype=np.float32),
        }
    )

    assert captured["action_source"].shape == (1, 48, 14)
    torch.testing.assert_close(captured["action_source"], torch.full((1, 48, 14), 0.5))
    assert captured["num_inference_steps"] == 4
    assert captured["decode_video"] is False
    assert result["actions"].shape == (48, 14)
