import numpy as np
import torch
from pathlib import Path

from data.libero.action_normalization import RawOscActionNormalizer
from data.libero.generate_language_action import raw_osc_window_to_lap
from data.libero.image_utils import compose_libero_views
from data.libero.libero_dataset import Episode, LiberoMotusDataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compose_libero_views_stacks_third_person_above_wrist():
    third = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.zeros((256, 256, 3), dtype=np.uint8)
    third[..., 0] = 255
    wrist[..., 1] = 255

    result = compose_libero_views(third, wrist)

    assert result.shape == (384, 320, 3)
    assert np.array_equal(result[96, 160], np.array([255, 0, 0], dtype=np.uint8))
    assert np.array_equal(result[288, 160], np.array([0, 255, 0], dtype=np.uint8))


def test_raw_osc_lap_uses_controller_scales_and_last_gripper():
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[0, 0] = 1.0
    actions[0, 5] = 1.0
    actions[-1, 6] = 1.0

    text = raw_osc_window_to_lap(actions)

    assert "move forward 5 cm" in text
    assert "rotate counterclockwise 30 degrees" in text
    assert text.endswith("open gripper")


def test_raw_osc_normalization_round_trip_keeps_binary_gripper():
    normalizer = RawOscActionNormalizer(
        minimum=np.asarray([-0.2, -0.4, -0.6, -0.8, -1.0, -1.2, 0.0], dtype=np.float32),
        maximum=np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.0], dtype=np.float32),
        mask=np.asarray([True, True, True, True, True, True, False]),
    )
    raw = np.asarray(
        [
            [-0.2, 0.0, 0.6, -0.4, 0.5, 1.2, 0.0],
            [0.2, 0.4, -0.6, 0.8, -1.0, -1.2, 1.0],
        ],
        dtype=np.float32,
    )

    normalized = normalizer.normalize(raw)
    restored = normalizer.denormalize(normalized)

    assert np.allclose(restored, raw)
    assert np.array_equal(normalized[:, -1], raw[:, -1])


def test_libero_dataset_pads_raw_osc_actions_to_target_action_dim(monkeypatch):
    normalizer = RawOscActionNormalizer(
        minimum=np.asarray([-1.0] * 7, dtype=np.float32),
        maximum=np.asarray([1.0] * 7, dtype=np.float32),
        mask=np.asarray([True, True, True, True, True, True, False]),
    )
    raw_actions = np.asarray(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, 0.0],
            [1.0, 0.5, 0.0, -0.5, -1.0, -0.25, 1.0],
        ],
        dtype=np.float32,
    )
    states = np.zeros((3, 8), dtype=np.float32)
    dataset = LiberoMotusDataset.__new__(LiberoMotusDataset)
    dataset.samples = [(0, 0)]
    dataset.episodes = [
        Episode(
            suite="libero_spatial",
            root=Path("/tmp/libero_spatial_no_noops"),
            episode_index=0,
            length=3,
            task_index=0,
            instruction="pick up the object",
            chunk_size=1000,
        )
    ]
    dataset.action_chunk_size = 2
    dataset.num_video_frames = 1
    dataset.video_action_freq_ratio = 1
    dataset.action_normalizer = normalizer
    dataset.use_language_action = False
    dataset.vlm_processor = None
    dataset.target_action_dim = 14
    dataset.include_history_actions = False

    monkeypatch.setattr(dataset, "_load_episode_arrays", lambda _path: (states, raw_actions))
    monkeypatch.setattr(dataset, "_load_composite_frames", lambda _episode, _indices: torch.zeros(2, 3, 8, 8))
    monkeypatch.setattr(dataset, "_t5_path", lambda _episode: Path("/tmp/missing_t5.pt"))
    monkeypatch.setattr(dataset, "_lap_path", lambda _episode: Path("/tmp/missing_lap.txt"))

    sample = dataset[0]

    assert sample["action_sequence"].shape == (2, 14)
    assert torch.allclose(sample["action_sequence"][:, :7], torch.from_numpy(normalizer.normalize(raw_actions)))
    assert torch.equal(sample["action_sequence"][:, 7:], torch.zeros(2, 7))
    assert sample["action_mask"].shape == (2, 14)
    assert torch.equal(sample["action_mask"][:, :7], torch.ones(2, 7, dtype=torch.bool))
    assert torch.equal(sample["action_mask"][:, 7:], torch.zeros(2, 7, dtype=torch.bool))


def test_libero_history_is_causal_and_zero_padded_before_normalization():
    normalizer = RawOscActionNormalizer(
        minimum=np.asarray([-1.0] * 7, dtype=np.float32),
        maximum=np.asarray([1.0] * 7, dtype=np.float32),
        mask=np.asarray([True, True, True, True, True, True, False]),
    )
    actions = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1.0],
            [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dataset = LiberoMotusDataset.__new__(LiberoMotusDataset)
    dataset.history_action_length = 3
    dataset.target_action_dim = 7
    dataset.action_normalizer = normalizer

    at_episode_start = dataset._history_action_sequence(actions, frame_idx=0)
    before_third_action = dataset._history_action_sequence(actions, frame_idx=2)

    expected_raw = np.zeros((3, 7), dtype=np.float32)
    expected_raw[-2:] = actions[:2]
    torch.testing.assert_close(
        at_episode_start,
        torch.from_numpy(normalizer.normalize(np.zeros((3, 7), dtype=np.float32))),
    )
    torch.testing.assert_close(
        before_third_action,
        torch.from_numpy(normalizer.normalize(expected_raw)),
    )
    assert not torch.equal(before_third_action[-1], torch.from_numpy(normalizer.normalize(actions[2:3]))[0])


def test_history_init_launcher_uses_independent_config_and_fresh_pretrain():
    config = (REPO_ROOT / "configs" / "libero_raw_osc_lap_h16_gradacc4_40k_hisinit_future_noise.yaml").read_text(
        encoding="utf-8"
    )
    script = (REPO_ROOT / "scripts" / "train_libero_lap_h16_gradacc4_40k_hisinit_future_noise.sh").read_text(
        encoding="utf-8"
    )

    assert "mode: history" in config
    assert "video_mode: gaussian" in config
    assert "history_length: 16" in config
    assert "future_video_noise_augmentation:" in config
    assert "enabled: true" in config
    assert "checkpoint_path: null" in config
    assert "pretrained_models/d0_v/mp_rank_00_model_states.pt" in config
    assert "libero_raw_osc_lap_h16_gradacc4_40k_hisinit_future_noise.yaml" in script


def test_libero_a14_launcher_uses_a14_names_and_log():
    script = (REPO_ROOT / "scripts" / "train_libero_lap_h16_gradacc4_40k_14a.sh").read_text(
        encoding="utf-8"
    )

    assert 'TASK="${TASK:-libero_raw_osc_h16_a14}"' in script
    assert 'CONFIG_FILE="${CONFIG_FILE:-configs/libero_raw_osc_lap_h16_gradacc4_40k_a14.yaml}"' in script
    assert 'LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_a14.log}"' in script
    assert '> "$LOG_FILE" 2>&1' in script
