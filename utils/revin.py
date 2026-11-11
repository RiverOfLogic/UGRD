from dataclasses import dataclass

import torch


@dataclass
class RevINStats:
    mean: torch.Tensor
    stdev: torch.Tensor


def revin_normalize(
    x: torch.Tensor,
    *,
    eps: float = 1e-5,
    detach_stats: bool = True,
) -> tuple[torch.Tensor, RevINStats]:
    if x.ndim != 3:
        raise ValueError(f"RevIN expects [batch, time, channels], got {tuple(x.shape)}.")

    mean = x.mean(dim=1, keepdim=True)
    variance = (x - mean).square().mean(dim=1, keepdim=True)
    stdev = torch.sqrt(variance + eps)
    if detach_stats:
        mean = mean.detach()
        stdev = stdev.detach()
    return (x - mean) / stdev, RevINStats(mean=mean, stdev=stdev)


def revin_denormalize(x: torch.Tensor, stats: RevINStats) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"RevIN expects [batch, time, channels], got {tuple(x.shape)}.")
    return x * stats.stdev + stats.mean
