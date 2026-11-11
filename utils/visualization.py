from pathlib import Path

import numpy as np


def plot_forecast_sample(
    path: str | Path,
    *,
    history: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    channel_name: str,
    title: str,
    dpi: int = 150,
    scale_label: str = "original scale",
    connect_from_history: bool = True,
    quantile_lower: np.ndarray | None = None,
    quantile_upper: np.ndarray | None = None,
    quantile_label: str = "quantile interval",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_x = np.arange(-len(history) + 1, 1)
    future_x = np.arange(1, len(truth) + 1)
    truth_x = future_x
    pred_x = future_x
    truth_y = truth
    pred_y = prediction
    if connect_from_history and len(history) > 0:
        truth_x = np.concatenate(([0], future_x))
        pred_x = np.concatenate(([0], future_x))
        truth_y = np.concatenate(([history[-1]], truth))
        pred_y = np.concatenate(([history[-1]], prediction))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(history_x, history, color="#111827", linewidth=1.8, label="history")
    if quantile_lower is not None and quantile_upper is not None:
        lower_y = quantile_lower
        upper_y = quantile_upper
        interval_x = future_x
        if connect_from_history and len(history) > 0:
            interval_x = np.concatenate(([0], future_x))
            lower_y = np.concatenate(([history[-1]], quantile_lower))
            upper_y = np.concatenate(([history[-1]], quantile_upper))
        ax.fill_between(interval_x, lower_y, upper_y, color="#9ecae1", alpha=0.35, label=quantile_label)
    ax.plot(truth_x, truth_y, color="#1f77b4", linewidth=2.0, label="truth")
    ax.plot(pred_x, pred_y, color="#d62728", linewidth=2.0, label="prediction")
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time relative to forecast start")
    ax.set_ylabel(f"{channel_name} ({scale_label})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_forecast_sample_csv(
    path: str | Path,
    *,
    history: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    quantile_lower: np.ndarray | None = None,
    quantile_upper: np.ndarray | None = None,
) -> None:
    import pandas as pd

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time_index = np.arange(-len(history) + 1, len(truth) + 1)
    values = pd.DataFrame({"time": time_index})
    values["history"] = np.nan
    values["truth"] = np.nan
    values["prediction"] = np.nan
    if quantile_lower is not None:
        values["quantile_lower"] = np.nan
    if quantile_upper is not None:
        values["quantile_upper"] = np.nan

    history_mask = values["time"] <= 0
    future_mask = values["time"] > 0
    values.loc[history_mask, "history"] = history
    values.loc[future_mask, "truth"] = truth
    values.loc[future_mask, "prediction"] = prediction
    if quantile_lower is not None:
        values.loc[future_mask, "quantile_lower"] = quantile_lower
    if quantile_upper is not None:
        values.loc[future_mask, "quantile_upper"] = quantile_upper

    values.to_csv(output_path, index=False)
