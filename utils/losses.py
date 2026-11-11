from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


QUANTILE_PENALTY_ALPHA = 0.5


def expand_channel_mask(channel_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if channel_mask.ndim != 2:
        raise ValueError(f"channel_mask must have shape [batch, channels], got {tuple(channel_mask.shape)}.")
    if target.ndim != 3:
        raise ValueError(f"target must have shape [batch, time, channels], got {tuple(target.shape)}.")
    if channel_mask.shape[0] != target.shape[0] or channel_mask.shape[1] != target.shape[2]:
        raise ValueError(
            f"channel_mask shape {tuple(channel_mask.shape)} does not match target shape {tuple(target.shape)}."
        )
    return channel_mask[:, None, :].to(device=target.device, dtype=target.dtype).expand_as(target)


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    mask = expand_channel_mask(channel_mask, target)
    squared_error = (prediction - target).square() * mask
    denominator = mask.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def distilts_horizon_weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    tau: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}.")
    if prediction.ndim != 3:
        raise ValueError(f"prediction must have shape [batch, time, channels], got {tuple(prediction.shape)}.")
    if prediction.shape[1] <= 0:
        raise ValueError("time length must be positive for DistilTS horizon-weighted MSE.")
    if not np.isfinite(float(tau)):
        raise ValueError(f"tau must be finite, got {tau}.")

    mask = expand_channel_mask(channel_mask, target)
    time_length = prediction.shape[1]
    if time_length == 1:
        horizon_weights = prediction.new_ones(1)
    else:
        horizon_index = torch.arange(time_length, device=prediction.device, dtype=prediction.dtype)
        horizon_weights = torch.exp(prediction.new_tensor(float(tau)) * horizon_index / float(time_length - 1))
        horizon_weights = horizon_weights / horizon_weights.mean().clamp_min(1e-12)
    squared_error = (prediction - target).square() * horizon_weights[None, :, None] * mask
    denominator = mask.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def _teacher_quantile_bounds(
    uncertainty_weight: Any | None,
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if uncertainty_weight is None:
        raise ValueError("uncertainty_weight must provide teacher_q01 and teacher_q09.")
    if isinstance(uncertainty_weight, dict):
        lower = uncertainty_weight.get("teacher_q01", uncertainty_weight.get("q01"))
        upper = uncertainty_weight.get("teacher_q09", uncertainty_weight.get("q09"))
    elif isinstance(uncertainty_weight, (tuple, list)) and len(uncertainty_weight) == 2:
        lower, upper = uncertainty_weight
    elif torch.is_tensor(uncertainty_weight) and uncertainty_weight.shape == (*prediction.shape, 2):
        lower = uncertainty_weight[..., 0]
        upper = uncertainty_weight[..., 1]
    else:
        raise ValueError("uncertainty_weight must provide teacher_q01 and teacher_q09.")
    if lower is None or upper is None:
        raise ValueError("uncertainty_weight must provide teacher_q01 and teacher_q09.")
    lower = lower.to(device=prediction.device, dtype=prediction.dtype)
    upper = upper.to(device=prediction.device, dtype=prediction.dtype)
    if lower.shape != prediction.shape or upper.shape != prediction.shape:
        raise ValueError(
            "teacher_q01 and teacher_q09 shapes must match prediction shape: "
            f"{tuple(lower.shape)} / {tuple(upper.shape)} != {tuple(prediction.shape)}"
        )
    return lower, upper


def quantile_penalty_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    uncertainty_weight: Any | None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}.")
    if prediction.ndim != 3:
        raise ValueError(f"prediction must have shape [batch, time, channels], got {tuple(prediction.shape)}.")
    if channel_mask.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError(
            f"channel_mask shape {tuple(channel_mask.shape)} does not match prediction shape {tuple(prediction.shape)}."
        )
    alpha_tensor = prediction.new_tensor(float(QUANTILE_PENALTY_ALPHA))
    if not torch.isfinite(alpha_tensor) or float(QUANTILE_PENALTY_ALPHA) < 0.0:
        raise ValueError(f"QUANTILE_PENALTY_ALPHA must be non-negative and finite, got {QUANTILE_PENALTY_ALPHA}.")

    mask = expand_channel_mask(channel_mask, target)
    lower, upper = _teacher_quantile_bounds(uncertainty_weight, prediction)
    center_error = prediction - target
    center_abs = center_error.abs()
    alpha_tensor = alpha_tensor.to(dtype=prediction.dtype)
    lower_penalty = alpha_tensor * (lower - prediction).clamp_min(0.0).square()
    upper_penalty = alpha_tensor * (prediction - upper).clamp_min(0.0).square()
    loss = torch.where(
        prediction < lower,
        center_abs + lower_penalty,
        torch.where(
            prediction > upper,
            center_abs + upper_penalty,
            center_abs,
        ),
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _fixed_stride2_channelwise_downsample(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"sequence must have shape [batch, time, channels], got {tuple(sequence.shape)}.")
    if sequence.shape[1] < 2:
        raise ValueError(f"cannot downsample time length {sequence.shape[1]} with kernel_size=2 and stride=2.")
    downsampled = F.avg_pool1d(sequence.transpose(1, 2), kernel_size=2, stride=2)
    return downsampled.transpose(1, 2)


def multi_scale_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    num_downsamples: int = 3,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}.")
    if prediction.ndim != 3:
        raise ValueError(f"prediction must have shape [batch, time, channels], got {tuple(prediction.shape)}.")
    if int(num_downsamples) < 0:
        raise ValueError(f"num_downsamples must be non-negative, got {num_downsamples}.")

    num_downsamples = int(num_downsamples)
    min_time_length = 2**num_downsamples
    if prediction.shape[1] < min_time_length:
        raise ValueError(
            f"time length {prediction.shape[1]} is too short for {num_downsamples} stride-2 downsamples; "
            f"expected at least {min_time_length}."
        )

    losses = []
    scale_prediction = prediction
    scale_target = target
    for scale_index in range(num_downsamples + 1):
        losses.append(masked_mse_loss(scale_prediction, scale_target, channel_mask))
        if scale_index < num_downsamples:
            scale_prediction = _fixed_stride2_channelwise_downsample(scale_prediction)
            scale_target = _fixed_stride2_channelwise_downsample(scale_target)
    weights = prediction.new_tensor([1.0 / float(scale_index + 1) for scale_index in range(num_downsamples + 1)])
    return (torch.stack(losses) * weights).sum() / weights.sum().clamp_min(1e-12)


def time_distill_scale_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    num_downsamples: int = 3,
    down_sampling_window: int = 2,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}.")
    if prediction.ndim != 3:
        raise ValueError(f"prediction must have shape [batch, time, channels], got {tuple(prediction.shape)}.")
    if int(num_downsamples) < 0:
        raise ValueError(f"num_downsamples must be non-negative, got {num_downsamples}.")
    if int(down_sampling_window) < 1:
        raise ValueError(f"down_sampling_window must be positive, got {down_sampling_window}.")

    num_downsamples = int(num_downsamples)
    down_sampling_window = int(down_sampling_window)
    min_time_length = down_sampling_window**num_downsamples
    if prediction.shape[1] < min_time_length:
        raise ValueError(
            f"time length {prediction.shape[1]} is too short for {num_downsamples} "
            f"kernel_size={down_sampling_window} downsamples; "
            f"expected at least {min_time_length}."
        )

    losses = []
    scale_prediction = prediction
    scale_target = target
    for scale_index in range(num_downsamples + 1):
        losses.append(masked_mse_loss(scale_prediction, scale_target, channel_mask))
        if scale_index < num_downsamples:
            scale_prediction = F.avg_pool1d(
                scale_prediction.transpose(1, 2),
                kernel_size=down_sampling_window,
                stride=down_sampling_window,
            ).transpose(1, 2)
            scale_target = F.avg_pool1d(
                scale_target.transpose(1, 2),
                kernel_size=down_sampling_window,
                stride=down_sampling_window,
            ).transpose(1, 2)
    return torch.stack(losses).mean()


def period_distribution_kl_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}.")
    if prediction.ndim != 3:
        raise ValueError(f"prediction must have shape [batch, time, channels], got {tuple(prediction.shape)}.")
    if prediction.shape[1] < 2:
        raise ValueError(f"time length must be at least 2 for period distribution KL, got {prediction.shape[1]}.")
    if channel_mask.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError(
            f"channel_mask shape {tuple(channel_mask.shape)} does not match prediction shape {tuple(prediction.shape)}."
        )
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    scale = float(temperature)
    prediction_amplitude = torch.fft.rfft(prediction, dim=1).abs()[:, 1:, :]
    target_amplitude = torch.fft.rfft(target.detach(), dim=1).abs()[:, 1:, :]
    student_log_probs = F.log_softmax(prediction_amplitude / scale, dim=1)
    teacher_probs = F.softmax(target_amplitude / scale, dim=1)
    per_frequency_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    per_channel_kl = per_frequency_kl.sum(dim=1)
    mask = channel_mask.to(device=prediction.device, dtype=per_channel_kl.dtype)
    return (per_channel_kl * mask).sum() / mask.sum().clamp_min(1.0)


def time_distill_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    num_downsamples: int = 3,
    period_temperature: float = 0.5,
    scale_weight: float = 1.0,
    period_weight: float = 1.0,
    down_sampling_window: int = 2,
) -> torch.Tensor:
    scale_loss = time_distill_scale_mse_loss(
        prediction,
        target,
        channel_mask,
        num_downsamples=num_downsamples,
        down_sampling_window=down_sampling_window,
    )
    period_loss = period_distribution_kl_loss(
        prediction,
        target,
        channel_mask,
        temperature=period_temperature,
    )
    return float(scale_weight) * scale_loss + float(period_weight) * period_loss


def _off_diagonal_relation_weights(
    relation: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    relation_layout: str = "source_target",
    absolute: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if relation.ndim not in (3, 4):
        raise ValueError(
            "relation must have shape [batch, channels, channels] or "
            f"[batch, heads, channels, channels], got {tuple(relation.shape)}."
        )
    if channel_mask.ndim != 2:
        raise ValueError(f"channel_mask must have shape [batch, channels], got {tuple(channel_mask.shape)}.")
    batch_size, num_channels = channel_mask.shape
    channel_axis = relation.shape[-1]
    if relation.shape[0] != batch_size or relation.shape[-2] != channel_axis or channel_axis != num_channels:
        raise ValueError(
            f"relation shape {tuple(relation.shape)} does not match channel_mask shape {tuple(channel_mask.shape)}."
        )
    if relation_layout not in {"source_target", "target_source"}:
        raise ValueError("relation_layout must be 'source_target' or 'target_source'.")

    weights = torch.nan_to_num(relation, nan=0.0, posinf=0.0, neginf=0.0)
    if absolute:
        weights = weights.abs()
    weights = weights.to(device=channel_mask.device)
    mask = channel_mask.to(device=weights.device, dtype=weights.dtype)
    if relation.ndim == 3:
        pair_mask = mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(num_channels, device=weights.device, dtype=torch.bool)[None]
        off_diag = (weights * pair_mask).masked_fill(eye, 0.0)
        channel_weight = off_diag.sum(dim=1 if relation_layout == "source_target" else 2)
    else:
        pair_mask = mask[:, None, :, None] * mask[:, None, None, :]
        eye = torch.eye(num_channels, device=weights.device, dtype=torch.bool)[None, None]
        off_diag = (weights * pair_mask).masked_fill(eye, 0.0)
        channel_weight = off_diag.sum(dim=2 if relation_layout == "source_target" else 3).sum(dim=1)
    return channel_weight, off_diag


def relation_weighted_mse_parts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    relation: torch.Tensor,
    *,
    relation_layout: str = "source_target",
    absolute: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return numerator and denominator for relation-weighted MSE.

    For relation layout [source, target], target channel i receives weight
    sum_j relation[j, i], excluding the diagonal self relation. The final loss
    is sum_i(weight_i * MSE_i) / sum_{i != j} relation[j, i].
    """

    mask = expand_channel_mask(channel_mask, target)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}.")
    channel_weight, _ = _off_diagonal_relation_weights(
        relation,
        channel_mask,
        relation_layout=relation_layout,
        absolute=absolute,
    )
    squared_error = (prediction - target).square() * mask
    per_channel_count = mask.sum(dim=1).clamp_min(1.0)
    per_channel_mse = squared_error.sum(dim=1) / per_channel_count
    valid_weight = channel_weight.to(device=per_channel_mse.device, dtype=per_channel_mse.dtype)
    valid_weight = valid_weight * channel_mask.to(device=per_channel_mse.device, dtype=per_channel_mse.dtype)
    numerator = (valid_weight * per_channel_mse).sum()
    denominator = valid_weight.sum()
    return numerator, denominator


def relation_weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    relation: torch.Tensor,
    *,
    relation_layout: str = "source_target",
    absolute: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    numerator, denominator = relation_weighted_mse_parts(
        prediction,
        target,
        channel_mask,
        relation,
        relation_layout=relation_layout,
        absolute=absolute,
    )
    return numerator / denominator.clamp_min(float(eps))


def masked_mae_loss(prediction: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    mask = expand_channel_mask(channel_mask, target)
    absolute_error = (prediction - target).abs() * mask
    denominator = mask.sum().clamp_min(1.0)
    return absolute_error.sum() / denominator


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    mask = expand_channel_mask(channel_mask, target)
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=float(beta)) * mask
    denominator = mask.sum().clamp_min(1.0)
    return loss.sum() / denominator


def supervised_forecasting_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    loss_name: str = "mse",
    smooth_l1_beta: float = 1.0,
    relation: torch.Tensor | None = None,
    relation_layout: str = "source_target",
    multi_scale_downsamples: int = 3,
    period_distribution_temperature: float = 1.0,
    distilts_tau: float = 1.0,
    time_distill_num_downsamples: int = 3,
    time_distill_period_temperature: float = 0.5,
    time_distill_scale_weight: float = 1.0,
    time_distill_period_weight: float = 1.0,
    time_distill_down_sampling_window: int = 2,
    uncertainty_weight: Any | None = None,
) -> torch.Tensor:
    if loss_name == "mse":
        return masked_mse_loss(prediction, target, channel_mask)
    if loss_name in ("distilts", "distilts_mse", "distilts_horizon_mse", "distilts_horizon_weighted_mse"):
        return distilts_horizon_weighted_mse_loss(
            prediction,
            target,
            channel_mask,
            tau=distilts_tau,
        )
    if loss_name in ("time_distill_prediction", "timedistill", "timedistill_prediction"):
        return time_distill_prediction_loss(
            prediction,
            target,
            channel_mask,
            num_downsamples=time_distill_num_downsamples,
            period_temperature=time_distill_period_temperature,
            scale_weight=time_distill_scale_weight,
            period_weight=time_distill_period_weight,
            down_sampling_window=time_distill_down_sampling_window,
        )
    if loss_name == "quantile_penalty_loss":
        return quantile_penalty_loss(
            prediction,
            target,
            channel_mask,
            uncertainty_weight,
        )
    if loss_name in ("multi_scale_mse", "multiscale_mse", "ms_mse"):
        return multi_scale_mse_loss(
            prediction,
            target,
            channel_mask,
            num_downsamples=multi_scale_downsamples,
        )
    if loss_name in ("period_distribution_kl", "period_kl", "frequency_distribution_kl"):
        return period_distribution_kl_loss(
            prediction,
            target,
            channel_mask,
            temperature=period_distribution_temperature,
        )
    if loss_name == "wmse":
        if relation is None:
            raise ValueError("relation is required when loss_name='wmse'.")
        return relation_weighted_mse_loss(
            prediction,
            target,
            channel_mask,
            relation,
            relation_layout=relation_layout,
            absolute=True,
        )
    if loss_name == "mae":
        return masked_mae_loss(prediction, target, channel_mask)
    if loss_name in ("smooth_l1", "huber"):
        return masked_smooth_l1_loss(prediction, target, channel_mask, beta=smooth_l1_beta)
    raise ValueError(f"Unsupported supervised loss: {loss_name}")


def soft_distillation_loss(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    channel_mask: torch.Tensor,
    loss_name: str = "mse",
    multi_scale_downsamples: int = 3,
    period_distribution_temperature: float = 1.0,
    distilts_tau: float = 1.0,
    time_distill_num_downsamples: int = 3,
    time_distill_period_temperature: float = 0.5,
    time_distill_scale_weight: float = 1.0,
    time_distill_period_weight: float = 1.0,
    time_distill_down_sampling_window: int = 2,
    uncertainty_weight: Any | None = None,
) -> torch.Tensor:
    if teacher_prediction.shape != student_prediction.shape:
        raise ValueError(
            "teacher_prediction shape must match student_prediction shape: "
            f"{tuple(teacher_prediction.shape)} != {tuple(student_prediction.shape)}"
        )
    return supervised_forecasting_loss(
        student_prediction,
        teacher_prediction.detach(),
        channel_mask,
        loss_name=loss_name,
        multi_scale_downsamples=multi_scale_downsamples,
        period_distribution_temperature=period_distribution_temperature,
        distilts_tau=distilts_tau,
        time_distill_num_downsamples=time_distill_num_downsamples,
        time_distill_period_temperature=time_distill_period_temperature,
        time_distill_scale_weight=time_distill_scale_weight,
        time_distill_period_weight=time_distill_period_weight,
        time_distill_down_sampling_window=time_distill_down_sampling_window,
        uncertainty_weight=uncertainty_weight,
    )


def blended_supervised_distillation_loss(
    student_prediction: torch.Tensor,
    hard_target: torch.Tensor,
    teacher_prediction: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    hard_weight: float = 1.0,
    distillation_weight: float = 1.0,
    hard_loss_name: str = "mse",
    distillation_loss_name: str = "mse",
) -> tuple[torch.Tensor, dict[str, float]]:
    hard_loss = supervised_forecasting_loss(
        student_prediction,
        hard_target,
        channel_mask,
        loss_name=hard_loss_name,
    )
    distill_loss = soft_distillation_loss(
        student_prediction,
        teacher_prediction,
        channel_mask,
        loss_name=distillation_loss_name,
    )
    total = float(hard_weight) * hard_loss + float(distillation_weight) * distill_loss
    return total, {
        "hard_loss": float(hard_loss.detach().cpu().item()),
        "distillation_loss": float(distill_loss.detach().cpu().item()),
        "total_loss": float(total.detach().cpu().item()),
    }


def masked_hidden_mse_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            "student_hidden shape must match teacher_hidden shape: "
            f"{tuple(student_hidden.shape)} != {tuple(teacher_hidden.shape)}"
        )
    if student_hidden.ndim not in (3, 4):
        raise ValueError(
            "hidden tensors must have shape [batch, channels, dim] or "
            f"[batch, channels, tokens, dim], got {tuple(student_hidden.shape)}."
        )
    if channel_mask.shape != student_hidden.shape[:2]:
        raise ValueError(
            f"channel_mask shape {tuple(channel_mask.shape)} does not match hidden shape {tuple(student_hidden.shape)}."
        )
    if student_hidden.ndim == 3:
        mask = channel_mask[:, :, None].to(device=student_hidden.device, dtype=student_hidden.dtype)
        denominator = (mask.sum() * student_hidden.shape[-1]).clamp_min(1.0)
    else:
        mask = channel_mask[:, :, None, None].to(device=student_hidden.device, dtype=student_hidden.dtype)
        denominator = (mask.sum() * student_hidden.shape[-2] * student_hidden.shape[-1]).clamp_min(1.0)
    squared_error = (student_hidden - teacher_hidden.detach()).square() * mask
    return squared_error.sum() / denominator


def masked_hidden_cosine_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    channel_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            "student_hidden shape must match teacher_hidden shape: "
            f"{tuple(student_hidden.shape)} != {tuple(teacher_hidden.shape)}"
        )
    if student_hidden.ndim not in (3, 4):
        raise ValueError(
            "hidden tensors must have shape [batch, channels, dim] or "
            f"[batch, channels, tokens, dim], got {tuple(student_hidden.shape)}."
        )
    if channel_mask.shape != student_hidden.shape[:2]:
        raise ValueError(
            f"channel_mask shape {tuple(channel_mask.shape)} does not match hidden shape {tuple(student_hidden.shape)}."
        )
    cos_sim = torch.nn.functional.cosine_similarity(
        student_hidden, teacher_hidden.detach(), dim=-1, eps=eps
    )
    if student_hidden.ndim == 3:
        mask = channel_mask.to(device=student_hidden.device, dtype=student_hidden.dtype)
        denominator = mask.sum().clamp_min(1.0)
    else:
        mask = channel_mask[:, :, None].to(device=student_hidden.device, dtype=student_hidden.dtype)
        denominator = mask.sum().clamp_min(1.0)
    cosine_loss = (1.0 - cos_sim) * mask
    return cosine_loss.sum() / denominator


def _relation_pair_mask(channel_mask: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    if channel_mask.ndim != 2:
        raise ValueError(f"channel_mask must have shape [batch, channels], got {tuple(channel_mask.shape)}.")
    if relation.ndim == 3:
        if channel_mask.shape != relation.shape[:2] or relation.shape[1] != relation.shape[2]:
            raise ValueError(
                f"channel_mask shape {tuple(channel_mask.shape)} does not match relation shape {tuple(relation.shape)}."
            )
        return (channel_mask[:, :, None] * channel_mask[:, None, :]).to(device=relation.device, dtype=relation.dtype)
    if relation.ndim == 4:
        if channel_mask.shape[0] != relation.shape[0] or channel_mask.shape[1] != relation.shape[2]:
            raise ValueError(
                f"channel_mask shape {tuple(channel_mask.shape)} does not match relation shape {tuple(relation.shape)}."
            )
        if relation.shape[2] != relation.shape[3]:
            raise ValueError(f"relation must have square channel axes, got {tuple(relation.shape)}.")
        pair_mask = channel_mask[:, None, :, None] * channel_mask[:, None, None, :]
        return pair_mask.to(device=relation.device, dtype=relation.dtype)
    raise ValueError(
        "relation must have shape [batch, channels, channels] or "
        f"[batch, heads, channels, channels], got {tuple(relation.shape)}."
    )


def _normalize_relation_rows(
    relation: torch.Tensor,
    channel_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    pair_mask = _relation_pair_mask(channel_mask, relation)
    masked = relation * pair_mask
    return masked / (masked.sum(dim=-1, keepdim=True) + float(eps))


def relation_mse_loss(
    adapter_relation: torch.Tensor,
    teacher_relation: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    if adapter_relation.shape != teacher_relation.shape:
        raise ValueError(
            "adapter_relation shape must match teacher_relation shape: "
            f"{tuple(adapter_relation.shape)} != {tuple(teacher_relation.shape)}"
        )
    pair_mask = _relation_pair_mask(channel_mask, adapter_relation)
    squared_error = (adapter_relation - teacher_relation.detach()).square() * pair_mask
    return squared_error.sum() / pair_mask.sum().clamp_min(1.0)


def relation_kl_loss(
    adapter_relation: torch.Tensor,
    teacher_relation: torch.Tensor,
    channel_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(A_teacher || A_adapter) over valid channel rows."""

    if adapter_relation.shape != teacher_relation.shape:
        raise ValueError(
            "adapter_relation shape must match teacher_relation shape: "
            f"{tuple(adapter_relation.shape)} != {tuple(teacher_relation.shape)}"
        )
    adapter_probs = _normalize_relation_rows(adapter_relation, channel_mask, eps)
    teacher_probs = _normalize_relation_rows(teacher_relation.detach(), channel_mask, eps)
    if adapter_relation.ndim == 3:
        query_mask = channel_mask.to(device=adapter_relation.device, dtype=adapter_relation.dtype)
    else:
        query_mask = channel_mask[:, None, :].to(device=adapter_relation.device, dtype=adapter_relation.dtype).expand(
            adapter_relation.shape[0],
            adapter_relation.shape[1],
            adapter_relation.shape[2],
        )
    per_entry = F.kl_div(torch.log(adapter_probs + float(eps)), teacher_probs, reduction="none")
    per_row = per_entry.sum(dim=-1) * query_mask
    return per_row.sum() / query_mask.sum().clamp_min(1.0)


def relation_topk_loss(
    adapter_relation: torch.Tensor,
    teacher_relation: torch.Tensor,
    channel_mask: torch.Tensor,
    k: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-entropy on teacher top-k channels for each valid query row."""

    if adapter_relation.shape != teacher_relation.shape:
        raise ValueError(
            "adapter_relation shape must match teacher_relation shape: "
            f"{tuple(adapter_relation.shape)} != {tuple(teacher_relation.shape)}"
        )
    adapter_probs = _normalize_relation_rows(adapter_relation, channel_mask, eps)
    teacher_probs = _normalize_relation_rows(teacher_relation.detach(), channel_mask, eps)
    original_ndim = adapter_probs.ndim
    if original_ndim == 3:
        flat_adapter = adapter_probs
        flat_teacher = teacher_probs
        flat_channel_mask = channel_mask.to(device=adapter_probs.device)
    else:
        batch_size, num_heads, num_channels, _ = adapter_probs.shape
        flat_adapter = adapter_probs.reshape(batch_size * num_heads, num_channels, num_channels)
        flat_teacher = teacher_probs.reshape(batch_size * num_heads, num_channels, num_channels)
        flat_channel_mask = channel_mask.to(device=adapter_probs.device).repeat_interleave(num_heads, dim=0)

    batch_size, num_channels, _ = flat_adapter.shape
    key_mask = flat_channel_mask[:, None, :].to(device=flat_adapter.device, dtype=torch.bool)
    query_mask = flat_channel_mask.to(device=flat_adapter.device, dtype=flat_adapter.dtype)
    valid_counts = flat_channel_mask.to(device=flat_adapter.device, dtype=torch.long).sum(dim=-1)
    max_k = min(max(int(k), 1), num_channels)
    teacher_for_topk = flat_teacher.masked_fill(~key_mask, -1.0)
    topk_idx = torch.topk(teacher_for_topk, k=max_k, dim=-1).indices
    selected_adapter = torch.gather(flat_adapter, dim=-1, index=topk_idx)
    selected_teacher = torch.gather(flat_teacher, dim=-1, index=topk_idx)
    selected_valid = torch.gather(key_mask.expand(batch_size, num_channels, num_channels), dim=-1, index=topk_idx)
    selected_teacher = selected_teacher * selected_valid.to(dtype=selected_teacher.dtype)
    weights = selected_teacher / (selected_teacher.sum(dim=-1, keepdim=True) + float(eps))
    per_row = -(weights * torch.log(selected_adapter + float(eps))).sum(dim=-1)
    enough_keys = (valid_counts >= 1).to(dtype=query_mask.dtype)[:, None]
    valid_query = query_mask * enough_keys
    return (per_row * valid_query).sum() / valid_query.sum().clamp_min(1.0)


def relation_distribution_loss(
    adapter_relation: torch.Tensor,
    teacher_relation: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    kl_weight: float = 1.0,
    topk_weight: float = 0.5,
    mse_weight: float = 0.0,
    topk: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    kl = relation_kl_loss(adapter_relation, teacher_relation, channel_mask, eps=eps)
    topk_loss = relation_topk_loss(adapter_relation, teacher_relation, channel_mask, k=topk, eps=eps)
    mse = relation_mse_loss(adapter_relation, teacher_relation, channel_mask)
    total = float(kl_weight) * kl + float(topk_weight) * topk_loss + float(mse_weight) * mse
    return total, {
        "relation_kl_loss": kl,
        "relation_topk_loss": topk_loss,
        "relation_mse_loss": mse,
    }


def divide_no_nan(a, b):
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, insample: torch.Tensor, freq: int, forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        del insample, freq
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, insample: torch.Tensor, freq: int, forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        del insample, freq
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target), torch.abs(forecast) + torch.abs(target)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, insample: torch.Tensor, freq: int, forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)
