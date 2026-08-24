"""Training-parity preprocessing and postprocessing for Piper D0 inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    embedding_path: str


DATA_ROOT = Path("/root/nas/piper_data/pick_anything")
TASK_SPECS = {
    "pick_keys": TaskSpec(
        "pick_keys",
        "Pick up keys and put it on the plate",
        str(DATA_ROOT / "pick-keys-trim/motus/t5_embeddings/task_000000.pt"),
    ),
    "pick_orange_cup": TaskSpec(
        "pick_orange_cup",
        "Pick up orange cup and put it on the plate",
        str(DATA_ROOT / "pick-orange-cup-trim/motus/t5_embeddings/task_000000.pt"),
    ),
    "pick_pen": TaskSpec(
        "pick_pen",
        "Pick up pen and put it on the plate",
        str(DATA_ROOT / "pick-pen-trim/motus/t5_embeddings/task_000000.pt"),
    ),
    "pick_pink_cup": TaskSpec(
        "pick_pink_cup",
        "Pick up pink cup and put it on the plate",
        str(DATA_ROOT / "pick-pink-cup-trim/motus/t5_embeddings/task_000000.pt"),
    ),
    "pick_tape": TaskSpec(
        "pick_tape",
        "Pick up tape and put it on the plate",
        str(DATA_ROOT / "pick-tape-trim/motus/t5_embeddings/task_000000.pt"),
    ),
    "pick_tissue": TaskSpec(
        "pick_tissue",
        "Pick up tissue and put it on the plate",
        str(DATA_ROOT / "pick-tissue-trim/motus/t5_embeddings/task_000000.pt"),
    ),
    "pick_watch": TaskSpec(
        "pick_watch",
        "Pick up watch and put it on the plate",
        str(DATA_ROOT / "pick-watch-trim/motus/t5_embeddings/task_000000.pt"),
    ),
    "pick_water_bottle": TaskSpec(
        "pick_water_bottle",
        "Pick up water bottle and put it on the plate",
        str(DATA_ROOT / "pick-water-bottle-trim/motus/t5_embeddings/task_000000.pt"),
    ),
}


def load_task_specs(items: list[dict[str, Any]] | None = None) -> dict[str, TaskSpec]:
    if items is None:
        return dict(TASK_SPECS)
    specs: dict[str, TaskSpec] = {}
    for item in items:
        spec = TaskSpec(
            task_id=str(item["task_id"]),
            instruction=str(item["instruction"]),
            embedding_path=str(item["embedding_path"]),
        )
        if not spec.task_id or not spec.instruction:
            raise ValueError("D0 deployment tasks require non-empty task_id and instruction")
        if spec.task_id in specs:
            raise ValueError(f"Duplicate D0 task_id={spec.task_id!r}")
        specs[spec.task_id] = spec
    if not specs:
        raise ValueError("D0 deployment must advertise at least one task")
    return specs


def task_metadata(task_specs: dict[str, TaskSpec] | None = None) -> list[dict[str, str]]:
    task_specs = TASK_SPECS if task_specs is None else task_specs
    return [
        {"task_id": spec.task_id, "instruction": spec.instruction}
        for spec in task_specs.values()
    ]


def resolve_task(
    task_id: str,
    prompt: str,
    task_specs: dict[str, TaskSpec] | None = None,
) -> TaskSpec:
    task_specs = TASK_SPECS if task_specs is None else task_specs
    try:
        spec = task_specs[str(task_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported D0 task_id={task_id!r}") from exc
    if str(prompt) != spec.instruction:
        raise ValueError(
            f"Prompt mismatch for task_id={task_id!r}: expected {spec.instruction!r}, got {prompt!r}"
        )
    if not Path(spec.embedding_path).is_file():
        raise FileNotFoundError(f"T5 embedding not found: {spec.embedding_path}")
    return spec


def _image_to_chw_float(image: Any, name: str) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must be HWC or 1HWC, got {array.shape}")
    if array.shape[-1] not in (3, 4):
        raise ValueError(f"{name} must have 3 or 4 channels, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {array.dtype}")
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255.0)


def stitch_three_views(
    front: Any,
    left_wrist: Any,
    right_wrist: Any,
    output_hw: tuple[int, int] = (384, 320),
) -> torch.Tensor:
    """Match the OLA training loader's RobotWin T-shaped mosaic exactly."""
    front_t = _image_to_chw_float(front, "front_camera")
    left_t = _image_to_chw_float(left_wrist, "left_wrist_camera")
    right_t = _image_to_chw_float(right_wrist, "right_wrist_camera")
    source_h, source_w = int(front_t.shape[-2]), int(front_t.shape[-1])
    wrist_h = max(1, source_h // 2)
    left_w = source_w // 2
    right_w = source_w - left_w
    left_t = F.interpolate(left_t.unsqueeze(0), size=(wrist_h, left_w), mode="bilinear", align_corners=False)
    right_t = F.interpolate(right_t.unsqueeze(0), size=(wrist_h, right_w), mode="bilinear", align_corners=False)
    bottom = torch.cat((left_t, right_t), dim=-1)
    mosaic = torch.cat((front_t.unsqueeze(0), bottom), dim=-2)
    return F.interpolate(mosaic, size=output_hw, mode="bilinear", align_corners=False).squeeze(0)


def tensor_to_pil(image_chw: torch.Tensor) -> Image.Image:
    array = image_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


class D0Normalizer:
    def __init__(self, stats_path: str | Path):
        with Path(stats_path).open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        self.state_low = np.asarray(stats["state"]["q01"], dtype=np.float32)
        self.state_high = np.asarray(stats["state"]["q99"], dtype=np.float32)
        self.action_low = np.asarray(stats["action"]["q01"], dtype=np.float32)
        self.action_high = np.asarray(stats["action"]["q99"], dtype=np.float32)
        for name, value in (
            ("state.q01", self.state_low),
            ("state.q99", self.state_high),
            ("action.q01", self.action_low),
            ("action.q99", self.action_high),
        ):
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} stats shape/values: {value.shape}")

    @staticmethod
    def _scale(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        width = np.maximum(high - low, 1e-6)
        return np.clip((values - low) / width, 0.0, 1.0)

    def normalize_state(self, state: Any) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(f"D0 state must be finite float[14], got {values.shape}")
        values = values.copy()
        values[[6, 13]] = np.clip(values[[6, 13]] / 100.0, 0.0, 1.0)
        return self._scale(values, self.state_low, self.state_high).astype(np.float32)

    def normalize_action(self, actions: Any) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
            raise ValueError(f"D0 history actions must be finite float[H,14], got {values.shape}")
        values = values.copy()
        values[:, [6, 13]] = np.clip(values[:, [6, 13]], 0.0, 1.0)
        return self._scale(values, self.action_low, self.action_high).astype(np.float32)

    def denormalize_action(self, actions: Any) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
            raise ValueError(f"D0 actions must be finite float[H,14], got {values.shape}")
        values = np.clip(values, 0.0, 1.0)
        result = values * (self.action_high - self.action_low) + self.action_low
        result[:, [6, 13]] = np.clip(result[:, [6, 13]], 0.0, 1.0)
        return result.astype(np.float32)
