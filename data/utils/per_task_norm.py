"""Strict per-task normalization for native 14D robot signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from data.utils.norm import normalize_masked


class PerTaskNormalizer:
    def __init__(self, stats_path: str, dataset_key: str, mode: str = "q01_q99") -> None:
        path = Path(stats_path)
        if path.is_dir():
            path = path / f"{dataset_key}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Per-task normalization stats not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("dataset") != dataset_key:
            raise ValueError(f"Invalid per-task stats schema/dataset in {path}")
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError(f"No task stats in {path}")
        if mode not in {"q01_q99", "minmax"}:
            raise ValueError(f"Unsupported per-task normalization mode: {mode}")
        self.path = path
        self.dataset_key = dataset_key
        self.mode = mode
        self.tasks: Dict[str, Dict[str, Any]] = tasks

    def normalize(
        self,
        task_key: str,
        signal: str,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        try:
            stats = self.tasks[str(task_key)][signal]
        except KeyError as exc:
            raise KeyError(
                f"Missing per-task stats for {self.dataset_key}/{task_key}/{signal} in {self.path}"
            ) from exc
        low_key, high_key = ("q01", "q99") if self.mode == "q01_q99" else ("min", "max")
        minimum = np.asarray(stats[low_key], dtype=np.float32)
        maximum = np.asarray(stats[high_key], dtype=np.float32)
        active_mask = np.asarray(stats["active_mask"], dtype=np.bool_)
        return normalize_masked(values, valid_mask, minimum, maximum, active_mask)
