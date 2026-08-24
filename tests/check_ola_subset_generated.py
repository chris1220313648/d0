#!/usr/bin/env python3
"""Validate generated OLA subset configs without loading videos or models."""

import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.lerobot.lerobot_ola_v3_dataset import LeRobotOLAV3Dataset


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: check_ola_subset_generated.py F25_JSON F50_JSON F25_YAML EXPECTED_STATS")
    report25 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    report50 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    config25 = OmegaConf.to_container(OmegaConf.load(sys.argv[3]), resolve=True)
    expected_stats = str(Path(sys.argv[4]))

    assert report25["seed"] == report50["seed"] == 42
    assert report25["normalization_recomputed"] is False
    assert report25["normalization_stats_path"] == expected_stats
    by_name50 = {item["name"]: item for item in report50["datasets"]}
    selected_from_config = {
        item["name"]: item["params"]["included_episode_indices"]
        for item in config25["dataset"]["datasets"]
    }
    stats_paths = {
        item["params"]["stats_path"] for item in config25["dataset"]["datasets"]
    }
    assert stats_paths == {expected_stats}
    assert len(report25["datasets"]) == len(selected_from_config) == 13
    for item25 in report25["datasets"]:
        expected_count = math.ceil(item25["eligible_episode_count"] * 0.25)
        assert item25["selected_episode_count"] == expected_count
        assert selected_from_config[item25["name"]] == item25["included_episode_indices"]
        assert set(item25["included_episode_indices"]).issubset(
            by_name50[item25["name"]]["included_episode_indices"]
        )

    first_child = config25["dataset"]["datasets"][0]
    loader_params = dict(first_child["params"])
    loader_params.update(
        {
            "normalize_state": False,
            "normalize_actions": False,
            "include_history_actions": True,
            "history_action_length": 48,
        }
    )
    dataset = LeRobotOLAV3Dataset(
        dataset_dir=first_child["dataset_dir"],
        global_downsample_rate=1,
        video_action_freq_ratio=6,
        num_video_frames=8,
        max_episodes=None,
        vlm_checkpoint_path=None,
        **loader_params,
    )
    assert [episode.episode_index for episode in dataset.episodes] == selected_from_config[first_child["name"]]
    print(
        json.dumps(
            {
                "datasets": len(report25["datasets"]),
                "selected_25_total": sum(item["selected_episode_count"] for item in report25["datasets"]),
                "selected_50_total": sum(item["selected_episode_count"] for item in report50["datasets"]),
                "seed": report25["seed"],
                "nested": True,
                "loader_episode_filter": True,
                "shared_stats": expected_stats,
            }
        )
    )


if __name__ == "__main__":
    main()
