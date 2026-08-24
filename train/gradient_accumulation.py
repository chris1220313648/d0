"""Shared optimizer update logic for Accelerate gradient accumulation."""

from __future__ import annotations

from typing import Any

import torch


def backward_and_step(
    *,
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    accelerator: Any | None,
    grad_clip_norm: float,
) -> bool:
    """Backpropagate one micro-batch and update only at an accumulation boundary.

    With Accelerate, ``optimizer.step`` and ``zero_grad`` are accumulation-aware
    wrappers. Calling them for every micro-batch is the documented pattern; the
    wrapped optimizer only mutates parameters when ``sync_gradients`` is true.
    The scheduler and clipping are explicitly limited to that same boundary.
    """
    if accelerator is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        return True

    accelerator.backward(loss)
    is_update_step = bool(accelerator.sync_gradients)
    if is_update_step:
        accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    if scheduler is not None and is_update_step:
        scheduler.step()
    optimizer.zero_grad()
    return is_update_step
