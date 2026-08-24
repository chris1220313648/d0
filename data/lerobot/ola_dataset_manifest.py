"""Load and validate OLA collection and mixture manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ManifestDataset:
    name: str
    collection: str
    path: Path
    weight: float
    use_for_val: bool
    excluded_episode_indices: tuple[int, ...]
    task_text: str | None


@dataclass(frozen=True)
class ResolvedManifest:
    name: str
    source_path: Path
    stats_path: Path
    datasets: tuple[ManifestDataset, ...]


def _plain_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must contain a YAML mapping: {path}")
    return value


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _load_collection(path: Path) -> tuple[str, Path | None, list[ManifestDataset]]:
    raw = _plain_yaml(path)
    if int(raw.get("version", 0)) != 1 or "datasets" not in raw:
        raise ValueError(f"Not an OLA collection manifest (version 1): {path}")
    name = str(raw.get("collection_name") or path.stem)
    root = _resolve_path(str(raw.get("root", ".")), path.parent)
    stats_value = raw.get("normalization_stats_path")
    stats_path = _resolve_path(str(stats_value), path.parent) if stats_value else None
    datasets: list[ManifestDataset] = []
    seen_names: set[str] = set()
    for item in raw["datasets"]:
        if not isinstance(item, dict) or "name" not in item or "path" not in item:
            raise ValueError(f"Each dataset needs name and path in {path}")
        item_name = str(item["name"])
        if item_name in seen_names:
            raise ValueError(f"Duplicate dataset name {item_name!r} in {path}")
        seen_names.add(item_name)
        dataset_path = _resolve_path(str(item["path"]), root)
        excluded = tuple(sorted({int(index) for index in item.get("excluded_episode_indices", [])}))
        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"Dataset weight must be positive: {name}/{item_name}")
        datasets.append(
            ManifestDataset(
                name=item_name,
                collection=name,
                path=dataset_path,
                weight=weight,
                use_for_val=bool(item.get("use_for_val", False)),
                excluded_episode_indices=excluded,
                task_text=str(item["task_text"]) if item.get("task_text") is not None else None,
            )
        )
    if not datasets:
        raise ValueError(f"Collection has no datasets: {path}")
    return name, stats_path, datasets


def load_manifest(manifest_path: str | Path) -> ResolvedManifest:
    """Resolve either one collection or a mixture into a flat dataset list."""
    path = Path(manifest_path).expanduser().resolve()
    raw = _plain_yaml(path)
    if "datasets" in raw:
        name, stats_path, datasets = _load_collection(path)
        if stats_path is None:
            raise ValueError(f"Collection must define normalization_stats_path: {path}")
        return ResolvedManifest(name, path, stats_path, tuple(datasets))

    if int(raw.get("version", 0)) != 1 or "collections" not in raw:
        raise ValueError(f"Not an OLA collection or mixture manifest: {path}")
    name = str(raw.get("mixture_name") or path.stem)
    normalization = raw.get("normalization") or {}
    stats_value = normalization.get("stats_path")
    if not stats_value:
        raise ValueError(f"Mixture must define normalization.stats_path: {path}")
    stats_path = _resolve_path(str(stats_value), path.parent)
    datasets: list[ManifestDataset] = []
    collection_names: set[str] = set()
    for collection_value in raw["collections"]:
        collection_path = _resolve_path(str(collection_value), path.parent)
        collection_name, _, children = _load_collection(collection_path)
        if collection_name in collection_names:
            raise ValueError(f"Collection listed more than once: {collection_name}")
        collection_names.add(collection_name)
        datasets.extend(children)
    if not datasets:
        raise ValueError(f"Mixture has no datasets: {path}")
    return ResolvedManifest(name, path, stats_path, tuple(datasets))


def generated_dataset_name(dataset: ManifestDataset, is_mixture: bool) -> str:
    return f"{dataset.collection}__{dataset.name}" if is_mixture else dataset.name
