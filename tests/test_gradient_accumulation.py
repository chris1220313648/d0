import sys
from pathlib import Path

import torch
from accelerate import Accelerator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.gradient_accumulation import backward_and_step


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_parameters_and_scheduler_update_only_on_accumulation_boundary():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = CountingScheduler()
    initial = accelerator.unwrap_model(model).weight.detach().clone()
    update_flags = []

    for _ in range(2):
        with accelerator.accumulate(model):
            prediction = model(torch.ones(1, 1))
            loss = (prediction - 2.0).square().mean()
            update_flags.append(
                backward_and_step(
                    loss=loss,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    accelerator=accelerator,
                    grad_clip_norm=1.0,
                )
            )
        if len(update_flags) == 1:
            assert torch.equal(accelerator.unwrap_model(model).weight.detach(), initial)

    assert update_flags == [False, True]
    assert scheduler.steps == 1
    assert not torch.equal(accelerator.unwrap_model(model).weight.detach(), initial)
