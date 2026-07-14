from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from data.utils.image_utils import resize_with_padding
from utils.image_qa_vlm_utils import preprocess_image_qa_messages


class ImageQADataset(Dataset):
    """LLaVA/Qwen-style image QA dataset adapter.

    Expected data structure:
      <image_root>/
        image_000001.jpg
        subdir/
          image_000002.png
          ...

      <json_path>.json

    JSON records should be a list of QA samples. Supported image fields are:
      - image: "relative/path.jpg"
      - image: ["relative/path1.jpg", "relative/path2.jpg"]
      - images: ["relative/path1.jpg", ...]

    Supports records with either:
      - image: "path.jpg"
      - image: ["path1.jpg", "path2.jpg", ...]
      - images: ["path1.jpg", ...]
    Multi-image records are reduced to one image per sample for VLM input.
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        vlm_checkpoint_path: str,
        max_samples: Optional[int] = None,
        max_length: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = False,
        num_video_frames: int = 8,
        video_action_freq_ratio: int = 1,
        action_dim: int = 14,
        state_dim: int = 14,
        video_size: Tuple[int, int] = (384, 320),
        t5_text_len: int = 512,
        t5_dim: int = 4096,
        val: bool = False,
    ):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)
        self.vlm_checkpoint_path = str(vlm_checkpoint_path)
        self.max_length = int(max_length) if max_length is not None else None
        self.num_video_frames = int(num_video_frames)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        self.t5_text_len = int(t5_text_len)
        self.t5_dim = int(t5_dim)
        self.val = val
        if not self.json_path.exists():
            raise FileNotFoundError(f"json_path not found: {self.json_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"image_root not found: {self.image_root}")

        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list, got {type(data).__name__}: {self.json_path}")
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(data)
        if max_samples is not None and int(max_samples) > 0:
            data = data[: int(max_samples)]
        self.samples: List[Dict[str, Any]] = data
        self.processor = AutoProcessor.from_pretrained(self.vlm_checkpoint_path, trust_remote_code=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.samples[idx]
        try:
            image_paths = self._extract_image_paths(sample)
            image_path = self._select_image_path(image_paths)
            question, answer = self._extract_qa(sample)
            raw_image = self._load_image(image_path)
            image = self._resize_image_with_padding(raw_image, self.video_size)
            vlm_inputs = preprocess_image_qa_messages(
                question,
                answer,
                [image],
                self.processor,
                max_length=self.max_length,
            )
            first_frame = self._image_to_frame_tensor(image)
            video_frames = first_frame.unsqueeze(0).repeat(self.num_video_frames, 1, 1, 1)
            action_sequence = torch.zeros(self.action_chunk_size, self.action_dim, dtype=torch.float32)
            return {
                # "qa_only": True,
                "id": sample.get("id", str(idx)),
                "image_paths": [str(image_path)],
                "question": question,
                "answer": answer,
                "first_frame": first_frame,
                "video_frames": video_frames,
                "action_sequence": action_sequence,
                "action_mask": torch.zeros_like(action_sequence, dtype=torch.bool),
                "video_mask": torch.tensor(False, dtype=torch.bool),
                "initial_state": torch.zeros(self.state_dim, dtype=torch.float32),
                "language_embedding": torch.zeros(self.t5_text_len, self.t5_dim, dtype=torch.float32),
                "history_action_sequence": torch.zeros_like(action_sequence),
                "vlm_inputs": vlm_inputs,
            }
        except Exception as exc:
            print(f"[ImageQADataset] skip idx={idx} id={sample.get('id')} error={type(exc).__name__}: {exc}")
            return None

    def _extract_image_paths(self, sample: Dict[str, Any]) -> List[Path]:
        raw = sample.get("image", None)
        if raw is None:
            raw = sample.get("images", None)
        if raw is None:
            raise ValueError("missing image/images field")
        if isinstance(raw, str):
            raw_paths: Sequence[str] = [raw]
        elif isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            raw_paths = raw
        else:
            raise TypeError(f"unsupported image field type: {type(raw).__name__}")
        paths = [self._resolve_media_path(path) for path in raw_paths]
        missing = [str(path) for path in paths if not path.exists() or path.stat().st_size <= 0]
        if missing:
            raise FileNotFoundError(f"missing/empty image files: {missing[:3]}")
        return paths

    def _select_image_path(self, image_paths: Sequence[Path]) -> Path:
        if not image_paths:
            raise ValueError("empty image path list")
        if self.val or len(image_paths) == 1:
            return image_paths[0]
        return random.choice(list(image_paths))

    def _resolve_media_path(self, path_text: str) -> Path:
        path = Path(path_text)
        if path.is_absolute():
            return path
        return self.image_root / path

    @staticmethod
    def _extract_qa(sample: Dict[str, Any]) -> Tuple[str, str]:
        conversations = sample.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError("missing conversations list")
        human = next((turn for turn in conversations if turn.get("from") == "human"), None)
        gpt = next((turn for turn in conversations if turn.get("from") == "gpt"), None)
        if human is None or gpt is None:
            raise ValueError("conversations must contain human and gpt turns")
        question = str(human.get("value", "")).strip()
        answer = str(gpt.get("value", "")).strip()
        if not question:
            raise ValueError("empty human question")
        if not answer:
            raise ValueError("empty gpt answer")
        return question, answer

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    @staticmethod
    def _resize_image_with_padding(image: Image.Image, video_size: Tuple[int, int]) -> Image.Image:
        frame_np = np.asarray(image.convert("RGB"))
        frame_np = resize_with_padding(frame_np, video_size)
        return Image.fromarray(frame_np, mode="RGB")

    @staticmethod
    def _image_to_frame_tensor(image: Image.Image) -> torch.Tensor:
        frame_np = np.asarray(image.convert("RGB"))
        return torch.from_numpy(frame_np.copy()).permute(2, 0, 1).float() / 255.0
