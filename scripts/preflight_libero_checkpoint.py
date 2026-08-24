#!/usr/bin/env python3
"""Verify that the shared Motus pretrain checkpoint fits the LIBERO heads."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.motus import Motus, MotusConfig


def build_config(config_path: str) -> MotusConfig:
    config = OmegaConf.load(config_path)
    return MotusConfig(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        action_expert_dim=config.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=config.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=config.model.action_expert.norm_eps,
        und_expert_hidden_size=config.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=config.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=config.model.und_expert.norm_eps,
        vlm_adapter_input_dim=config.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=config.model.und_expert.vlm.projector_type,
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=1,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        training_mode="finetune",
        load_pretrained_backbones=False,
        vlm_frozen=config.model.vlm.frozen,
    )


def resolve_checkpoint(path: str) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    candidates = [
        checkpoint / "pytorch_model" / "mp_rank_00_model_states.pt",
        checkpoint / "mp_rank_00_model_states.pt",
    ] if checkpoint.is_dir() else [checkpoint]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint file found under {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/libero_raw_osc_lap.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    model = Motus(build_config(args.config))
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("module", checkpoint)
    filtered = {
        key: value
        for key, value in state_dict.items()
        if "action_expert.input_encoder" not in key and "action_expert.decoder" not in key
    }
    incompatible = model.load_state_dict(filtered, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key for key in missing
        if "action_expert.input_encoder" not in key and "action_expert.decoder" not in key
    ]
    print(f"checkpoint={checkpoint_path}")
    print(f"loaded_keys={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")
    print("missing_keys:")
    for key in missing:
        print(f"  {key}")
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: invalid_missing={invalid_missing}, unexpected={unexpected}"
        )
    print("PREFLIGHT_OK: only the 8D state encoder and 7D action decoder are reinitialized")


if __name__ == "__main__":
    main()
