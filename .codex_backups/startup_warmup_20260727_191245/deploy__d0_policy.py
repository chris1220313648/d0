"""Single-GPU Motus/D0 policy adapter for the Piper checkpoint."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoProcessor

from deploy.d0_preprocessing import D0Normalizer, load_task_specs, resolve_task, stitch_three_views, tensor_to_pil
from models.motus import Motus, MotusConfig
from utils.vlm_utils import preprocess_vlm_messages


logger = logging.getLogger(__name__)


class D0Policy:
    def __init__(
        self,
        *,
        checkpoint_dir: str,
        config_path: str,
        stats_path: str,
        device: str = "cuda:0",
        num_inference_steps: int = 4,
        tasks: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_path = Path(config_path)
        self.device = torch.device(device)
        self.num_inference_steps = int(num_inference_steps)
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.normalizer = D0Normalizer(stats_path)
        self.task_specs = load_task_specs(tasks)
        self.model_name = model_name or self.checkpoint_dir.parent.name
        self._embedding_cache: dict[str, torch.Tensor] = {}
        self._infer_lock = threading.Lock()
        self.model = self._load_model()
        vlm_path = self.config["model"]["vlm"]["checkpoint_path"]
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)

    def _model_config(self) -> MotusConfig:
        common = self.config["common"]
        model = self.config["model"]
        flow = model.get("flow_source", {})
        video_noise = model.get("future_video_noise_augmentation", {})
        return MotusConfig(
            wan_checkpoint_path=model["wan"]["checkpoint_path"],
            vae_path=model["wan"]["vae_path"],
            wan_config_path=model["wan"]["config_path"],
            vlm_checkpoint_path=model["vlm"]["checkpoint_path"],
            video_precision=model["wan"]["precision"],
            action_state_dim=int(common["state_dim"]),
            action_dim=int(common["action_dim"]),
            action_expert_dim=int(model["action_expert"]["hidden_size"]),
            action_expert_ffn_dim_multiplier=int(model["action_expert"]["ffn_dim_multiplier"]),
            action_expert_norm_eps=float(model["action_expert"]["norm_eps"]),
            und_expert_hidden_size=int(model["und_expert"]["hidden_size"]),
            und_expert_ffn_dim_multiplier=int(model["und_expert"]["ffn_dim_multiplier"]),
            und_expert_norm_eps=float(model["und_expert"]["norm_eps"]),
            vlm_adapter_input_dim=int(model["und_expert"]["vlm"]["input_dim"]),
            vlm_adapter_projector_type=model["und_expert"]["vlm"]["projector_type"],
            global_downsample_rate=int(common["global_downsample_rate"]),
            video_action_freq_ratio=int(common["video_action_freq_ratio"]),
            num_video_frames=int(common["num_video_frames"]),
            video_height=int(common["video_height"]),
            video_width=int(common["video_width"]),
            batch_size=1,
            video_loss_weight=float(model["loss_weights"]["video_loss_weight"]),
            action_loss_weight=float(model["loss_weights"]["action_loss_weight"]),
            flow_source_mode=flow.get("mode", "gaussian"),
            flow_source_video_mode=flow.get("video_mode", flow.get("mode", "gaussian")),
            flow_source_action_noise_std=float(flow.get("action_noise_std", 0.0)),
            future_video_noise_aug_enabled=bool(video_noise.get("enabled", False)),
            future_video_noise_aug_probability=float(video_noise.get("probability", 0.5)),
            future_video_noise_aug_min_scale=float(video_noise.get("min_scale", 0.5)),
            future_video_noise_aug_max_scale=float(video_noise.get("max_scale", 1.0)),
            future_video_noise_aug_start_index=int(video_noise.get("future_start_index", 1)),
            training_mode="finetune",
            load_pretrained_backbones=False,
            vlm_frozen=bool(model["vlm"]["frozen"]),
        )

    def _checkpoint_file(self) -> Path:
        candidates = (
            self.checkpoint_dir / "pytorch_model_0.bin",
            self.checkpoint_dir / "mp_rank_00_model_states.pt",
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"No supported model checkpoint under {self.checkpoint_dir}")

    def _load_model(self) -> Motus:
        logger.info("Building Motus model on %s", self.device)
        model = Motus(self._model_config()).to(self.device)
        checkpoint_file = self._checkpoint_file()
        logger.info("Loading D0 weights from %s", checkpoint_file)
        if checkpoint_file.name.startswith("pytorch_model_"):
            state = torch.load(checkpoint_file, map_location="cpu", weights_only=True, mmap=True)
        else:
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            state = checkpoint.get("module", checkpoint)
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        torch.cuda.empty_cache()
        return model

    def _language_embedding(self, task_id: str, embedding_path: str) -> torch.Tensor:
        cached = self._embedding_cache.get(task_id)
        if cached is None:
            cached = torch.load(embedding_path, map_location="cpu", weights_only=True)
            if cached.ndim == 3 and cached.shape[0] == 1:
                cached = cached.squeeze(0)
            if cached.ndim != 2:
                raise ValueError(f"T5 embedding must be [S,D], got {tuple(cached.shape)}")
            cached = cached.to(dtype=torch.bfloat16)
            self._embedding_cache[task_id] = cached
        return cached

    @staticmethod
    def _move_vlm_inputs(inputs: Any, device: torch.device) -> dict[str, Any]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in dict(inputs).items()
        }

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        required = {
            "state",
            "front_camera",
            "left_wrist_camera",
            "right_wrist_camera",
            "task_id",
            "prompt",
        }
        missing = sorted(required.difference(observation))
        if missing:
            raise KeyError(f"D0 observation missing keys: {missing}")
        spec = resolve_task(str(observation["task_id"]), str(observation["prompt"]), self.task_specs)
        first_frame = stitch_three_views(
            observation["front_camera"],
            observation["left_wrist_camera"],
            observation["right_wrist_camera"],
        )
        normalized_state = self.normalizer.normalize_state(observation["state"])
        image_pil = tensor_to_pil(first_frame)
        vlm_inputs = preprocess_vlm_messages(spec.instruction, image_pil, self.vlm_processor)
        vlm_inputs = self._move_vlm_inputs(vlm_inputs, self.device)
        language_embedding = self._language_embedding(spec.task_id, spec.embedding_path)
        flow_mode = str(self.config.get("model", {}).get("flow_source", {}).get("mode", "gaussian"))
        action_source = None
        if flow_mode == "history":
            history = np.asarray(observation.get("history_actions"), dtype=np.float32)
            expected_horizon = int(self.config["common"]["num_video_frames"]) * int(
                self.config["common"]["video_action_freq_ratio"]
            )
            if history.shape != (expected_horizon, 14):
                raise ValueError(f"D0 history_actions must be [{expected_horizon},14], got {history.shape}")
            action_source = torch.from_numpy(self.normalizer.normalize_action(history)).unsqueeze(0)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        infer_started = time.perf_counter()
        with self._infer_lock:
            _, predicted_actions = self.model.inference_step(
                first_frame=first_frame.unsqueeze(0).to(self.device),
                state=torch.from_numpy(normalized_state).unsqueeze(0).to(self.device),
                num_inference_steps=self.num_inference_steps,
                language_embeddings=language_embedding.unsqueeze(0),
                vlm_inputs=vlm_inputs,
                action_source=action_source,
                decode_video=False,
            )
        infer_ms = (time.perf_counter() - infer_started) * 1000.0

        post_started = time.perf_counter()
        actions = self.normalizer.denormalize_action(predicted_actions.squeeze(0).cpu().numpy())
        expected_horizon = int(self.config["common"]["num_video_frames"]) * int(
            self.config["common"]["video_action_freq_ratio"]
        )
        if actions.shape != (expected_horizon, 14):
            raise ValueError(f"Unexpected D0 action shape: {actions.shape}")
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        return {
            "actions": actions,
            "task_id": spec.task_id,
            "model_name": self.model_name,
            "policy_timing": {
                "preprocess_ms": preprocess_ms,
                "infer_ms": infer_ms,
                "postprocess_ms": postprocess_ms,
            },
        }
