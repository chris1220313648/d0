#!/usr/bin/env python3
"""Generate one Motus training config from an OLA dataset manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.lerobot.ola_dataset_manifest import generated_dataset_name, load_manifest


SUBSET_SEED = 42


def _episode_rows(dataset_path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted((dataset_path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata under {dataset_path / 'meta' / 'episodes'}")
    rows: list[dict] = []
    for path in paths:
        rows.extend(pq.read_table(path, columns=["episode_index", "tasks"]).to_pylist())
    return rows


def select_episode_indices(
    episode_indices: list[int],
    fraction: float,
    dataset_identity: str,
    seed: int = SUBSET_SEED,
) -> list[int]:
    """Return a deterministic nested random subset for one child dataset."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("episode_fraction must be in the interval (0, 1]")
    ordered = sorted({int(index) for index in episode_indices})
    if not ordered:
        raise ValueError(f"No eligible episodes for {dataset_identity}")
    digest = hashlib.sha256(f"{seed}:{dataset_identity}".encode("utf-8")).digest()
    child_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    random.Random(child_seed).shuffle(ordered)
    count = min(len(ordered), max(1, math.ceil(len(ordered) * float(fraction))))
    return sorted(ordered[:count])


def _fraction_tag(fraction: float) -> str:
    value = f"{float(fraction):.8f}".rstrip("0").rstrip(".")
    return f"f{value.replace('.', 'p')}_s{SUBSET_SEED}"


def _eligible_episode_indices(item) -> tuple[list[int], int]:
    rows = _episode_rows(item.path)
    excluded = set(item.excluded_episode_indices)
    eligible = []
    for row in rows:
        episode_index = int(row["episode_index"])
        tasks = {str(task) for task in (row.get("tasks") or [])}
        if episode_index in excluded:
            continue
        if item.task_text is not None and item.task_text not in tasks:
            continue
        eligible.append(episode_index)
    return sorted(set(eligible)), len({int(row["episode_index"]) for row in rows})


def build_config(
    manifest_path: str | Path,
    template_path: str | Path,
    episode_fraction: float | None = None,
) -> tuple[dict, Path, str]:
    resolved = load_manifest(manifest_path)
    template_path = Path(template_path).expanduser().resolve()
    config = OmegaConf.to_container(OmegaConf.load(template_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Template must contain a YAML mapping: {template_path}")
    dataset_config = config["dataset"]
    base_params = copy.deepcopy(dataset_config.pop("ola_params"))
    is_mixture = len({item.collection for item in resolved.datasets}) > 1
    children = []
    subset_children = []
    for item in resolved.datasets:
        params = copy.deepcopy(base_params)
        params["stats_path"] = str(resolved.stats_path)
        if item.excluded_episode_indices:
            params["excluded_episode_indices"] = list(item.excluded_episode_indices)
        if item.task_text:
            params["task_text_filter"] = item.task_text
        if episode_fraction is not None:
            eligible, source_count = _eligible_episode_indices(item)
            identity = f"{item.collection}/{item.name}"
            selected = select_episode_indices(eligible, episode_fraction, identity)
            params["included_episode_indices"] = selected
            subset_children.append(
                {
                    "name": generated_dataset_name(item, is_mixture),
                    "dataset_dir": str(item.path),
                    "source_episode_count": source_count,
                    "eligible_episode_count": len(eligible),
                    "selected_episode_count": len(selected),
                    "actual_fraction_of_eligible": len(selected) / len(eligible),
                    "included_episode_indices": selected,
                }
            )
        children.append(
            {
                "name": generated_dataset_name(item, is_mixture),
                "type": "lerobot_ola_v3",
                "weight": item.weight,
                "use_for_val": item.use_for_val,
                "dataset_dir": str(item.path),
                "max_episodes": None,
                "image_aug": False,
                "use_language_action": True,
                "params": params,
            }
        )
    dataset_config["datasets"] = children
    name = resolved.name
    if episode_fraction is not None:
        name = f"{name}_{_fraction_tag(episode_fraction)}"
        dataset_config["episode_subset"] = {
            "requested_fraction": float(episode_fraction),
            "seed": SUBSET_SEED,
            "selection_unit": "episode",
            "rounding": "ceil",
            "normalization_stats_path": str(resolved.stats_path),
            "normalization_recomputed": False,
            "datasets": subset_children,
        }
    return config, resolved.source_path, name


def check_artifacts(manifest_path: str | Path) -> None:
    resolved = load_manifest(manifest_path)
    missing: list[str] = []
    if not resolved.stats_path.is_file():
        missing.append(str(resolved.stats_path))
    for item in resolved.datasets:
        motus = item.path / "motus"
        for path in (motus / "schema.json", motus / "language_action", motus / "t5_embeddings"):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing preprocessing artifacts:\n  " + "\n  ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--template",
        default=str(REPO_ROOT / "configs" / "ola_lap_hisinit_future_noise_template.yaml"),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument("--episode-fraction", type=float, default=None)
    args = parser.parse_args()
    if args.check_artifacts:
        check_artifacts(args.manifest)
    config, source_path, name = build_config(
        args.manifest,
        args.template,
        episode_fraction=args.episode_fraction,
    )
    output = Path(args.output) if args.output else REPO_ROOT / "configs" / "generated" / f"{name}.yaml"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), output)
    selection_path = None
    if args.episode_fraction is not None:
        selection_path = output.with_suffix(".selection.json")
        selection_path.write_text(
            json.dumps(config["dataset"]["episode_subset"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "name": name,
                "manifest": str(source_path),
                "config": str(output),
                "selection": str(selection_path) if selection_path else None,
            }
        )
    )


if __name__ == "__main__":
    main()
