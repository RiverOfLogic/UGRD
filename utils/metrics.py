from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ErrorSums:
    abs_error: float = 0.0
    squared_error: float = 0.0
    abs_percentage_error: float = 0.0
    squared_percentage_error: float = 0.0
    count: int = 0

    def update(self, prediction: np.ndarray, target: np.ndarray, channel_mask: np.ndarray) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction shape {prediction.shape} != target shape {target.shape}")
        expanded_mask = np.broadcast_to(channel_mask[:, None, :].astype(bool), target.shape)
        prediction_values = prediction[expanded_mask]
        target_values = target[expanded_mask]
        errors = prediction_values - target_values
        with np.errstate(divide="ignore", invalid="ignore"):
            percentage_errors = errors / target_values
        self.abs_error += float(np.abs(errors).sum())
        self.squared_error += float(np.square(errors).sum())
        self.abs_percentage_error += float(np.abs(percentage_errors).sum())
        self.squared_percentage_error += float(np.square(percentage_errors).sum())
        self.count += int(errors.size)

    def as_metrics(self) -> dict[str, float | int]:
        denominator = max(self.count, 1)
        mse = self.squared_error / denominator
        return {
            "mae": self.abs_error / denominator,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mape": self.abs_percentage_error / denominator,
            "mspe": self.squared_percentage_error / denominator,
            "count": self.count,
        }


def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def inverse_transform_values(dataset: Any, values: np.ndarray) -> np.ndarray:
    full_dim = len(getattr(dataset, "feature_names", []))
    if values.shape[-1] == full_dim:
        return np.asarray(dataset.inverse_transform(values), dtype=np.float32)
    if values.shape[-1] == 1:
        return np.asarray(dataset.inverse_transform(values, target_only=True), dtype=np.float32)
    raise ValueError(f"Cannot inverse-transform values with last dimension {values.shape[-1]} for full_dim={full_dim}.")


def inverse_transform_channel(dataset: Any, values: np.ndarray, channel_idx: int) -> np.ndarray:
    scaler = getattr(dataset, "scaler", None)
    if scaler is None:
        return values
    return values * float(scaler.scale_[channel_idx]) + float(scaler.mean_[channel_idx])


def select_output(values: torch.Tensor, dataset: Any) -> torch.Tensor:
    return values[..., dataset.output_slice]


def select_output_mask(channel_mask: torch.Tensor, dataset: Any) -> torch.Tensor:
    return channel_mask[:, dataset.output_slice]


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe
