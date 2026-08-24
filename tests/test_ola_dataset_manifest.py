from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.lerobot.ola_dataset_manifest import generated_dataset_name, load_manifest
from scripts.build_ola_manifest_config import build_config, select_episode_indices


def test_diverse_manifest_exclusions_and_weights():
    manifest = load_manifest(REPO_ROOT / "configs/dataset_manifests/pick_diverse_action.yaml")
    assert len(manifest.datasets) == 5
    assert all(item.weight == 1.0 for item in manifest.datasets)
    press = next(item for item in manifest.datasets if item.name == "press_calculator")
    assert press.excluded_episode_indices == (10, 34)
    assert press.path.name == "press-calcultor-trim"


def test_mixture_keeps_duplicate_tasks_as_separate_children():
    manifest_path = REPO_ROOT / "configs/dataset_mixtures/pick_anything_plus_diverse.yaml"
    manifest = load_manifest(manifest_path)
    assert len(manifest.datasets) == 13
    assert all(item.weight == 1.0 for item in manifest.datasets)
    pen_names = [generated_dataset_name(item, True) for item in manifest.datasets if item.name == "pick_pen"]
    assert pen_names == ["pick_anything__pick_pen", "pick_diverse_action__pick_pen"]
    assert manifest.stats_path == Path(
        "/root/nas/piper_data/motus_mixtures/pick_anything_plus_diverse/stats.json"
    )


def test_generated_config_uses_one_stats_file_and_preserves_exclusions():
    config, _, _ = build_config(
        REPO_ROOT / "configs/dataset_mixtures/pick_anything_plus_diverse.yaml",
        REPO_ROOT / "configs/ola_lap_hisinit_future_noise_template.yaml",
    )
    children = config["dataset"]["datasets"]
    assert len(children) == 13
    assert {child["params"]["stats_path"] for child in children} == {
        "/root/nas/piper_data/motus_mixtures/pick_anything_plus_diverse/stats.json"
    }
    press = next(child for child in children if child["name"].endswith("press_calculator"))
    assert press["params"]["excluded_episode_indices"] == [10, 34]
    assert config["common"]["video_action_freq_ratio"] == 6
    assert config["model"]["flow_source"]["history_length"] == 48
    assert config["training"]["max_steps"] == 20000
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert config["system"]["save_interval"] == 5000


def test_episode_subsets_are_deterministic_and_nested():
    episodes = list(range(50))
    first = select_episode_indices(episodes, 0.25, "collection/task")
    repeated = select_episode_indices(episodes, 0.25, "collection/task")
    larger = select_episode_indices(episodes, 0.50, "collection/task")
    assert first == repeated
    assert len(first) == 13
    assert len(larger) == 25
    assert set(first).issubset(larger)
