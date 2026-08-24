#!/usr/bin/env python3
"""Tiny distributed ZeRO-2 smoke test for the production accumulation helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.gradient_accumulation import backward_and_step


def main() -> None:
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=str(REPO_ROOT / "configs/zero2.json")),
        gradient_accumulation_steps=2,
        mixed_precision="bf16",
    )
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.full((4, 1), 2.0)), batch_size=1)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    initial = accelerator.unwrap_model(model).weight.detach().float().clone()
    flags: list[bool] = []
    unchanged_after_first = False

    for micro_step, (inputs, targets) in enumerate(loader):
        if micro_step == 2:
            break
        inputs = inputs.to(accelerator.device, dtype=torch.bfloat16)
        targets = targets.to(accelerator.device, dtype=torch.bfloat16)
        with accelerator.accumulate(model):
            loss = (model(inputs) - targets).square().mean()
            flags.append(
                backward_and_step(
                    loss=loss,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    accelerator=accelerator,
                    grad_clip_norm=1.0,
                )
            )
        if micro_step == 0:
            unchanged_after_first = torch.equal(
                accelerator.unwrap_model(model).weight.detach().float(), initial
            )

    changed_after_second = not torch.equal(
        accelerator.unwrap_model(model).weight.detach().float(), initial
    )
    passed = flags == [False, True] and unchanged_after_first and changed_after_second
    result = torch.tensor([int(passed)], device=accelerator.device)
    gathered = accelerator.gather(result)
    if accelerator.is_main_process:
        summary = {
            "world_size": accelerator.num_processes,
            "gradient_accumulation_steps": 2,
            "update_flags": flags,
            "unchanged_after_first_microbatch": unchanged_after_first,
            "changed_after_second_microbatch": changed_after_second,
            "all_ranks_passed": bool(gathered.bool().all().item()),
        }
        print(json.dumps(summary))
    if not bool(gathered.bool().all().item()):
        raise SystemExit(1)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
