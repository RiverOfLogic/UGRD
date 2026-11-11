"""Short student rollouts with fresh teacher labels and detached histories."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch


@contextmanager
def generation_mode(model):
    """Generate the deployment policy without dropout, preserving module modes."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def shift_append(history: torch.Tensor, forecast: torch.Tensor, steps: int) -> torch.Tensor:
    if history.ndim != 3 or forecast.ndim != 3:
        raise ValueError("History and forecast must be [batch, time, channels].")
    if history.shape[0] != forecast.shape[0] or history.shape[2] != forecast.shape[2]:
        raise ValueError("Rollout must preserve batch size and all input channels.")
    if steps < 1 or steps > min(history.shape[1], forecast.shape[1]):
        raise ValueError("Rollout block must fit both history and forecast lengths.")
    return torch.cat((history[:, steps:], forecast[:, :steps].detach()), dim=1).detach()


def short_rollout_loss(
    model: torch.nn.Module,
    history: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    block_size: int,
    rollout_steps: int,
    predict: Callable,
    teacher_label: Callable,
    distillation_loss: Callable,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Teacher and student share each generated context; no real suffix labels.

    Callbacks accept full histories/full-channel forecasts. The loss callback
    performs any dataset-specific output selection. Sampling and teacher labels
    are detached, while the student's predictions at visited states have gradients.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive.")
    visited = history.detach()
    losses = []
    for _ in range(rollout_steps):
        with generation_mode(model):
            generated = predict(visited, channel_mask)
        visited = shift_append(visited, generated, block_size)
        with torch.no_grad():
            labels = teacher_label(visited)
        prediction = predict(visited, channel_mask)
        losses.append(distillation_loss(prediction, labels, channel_mask))
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite short-rollout loss.")
    return loss, {"teacher_queries": rollout_steps, "samples": history.shape[0] * rollout_steps}
