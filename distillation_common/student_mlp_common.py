import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    ChannelIndependentMLP,
    ChannelIndependentMLPConfig,
    ThreeTapChannelIndependentMLP,
    ThreeTapChannelIndependentMLPConfig,
)
from research_data_loader import data_provider  # noqa: E402
from research_data_loader.cache_dataset import base_cache_fingerprint  # noqa: E402
from utils.losses import supervised_forecasting_loss  # noqa: E402
from utils.tools import adjust_learning_rate  # noqa: E402
from utils.metrics import (  # noqa: E402
    ErrorSums,
    inverse_transform_channel,
    select_output,
    select_output_mask,
    to_numpy,
)
from utils.revin import revin_denormalize, revin_normalize  # noqa: E402
from utils.visualization import plot_forecast_sample, write_forecast_sample_csv  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "mlp_distillation_suite" / "configs" / "01_only_soft.yaml"
METRIC_NAME_ALIASES = {
    "scaled_mae": "mae",
    "scaled_mse": "mse",
    "scaled_rmse": "rmse",
    "scaled_mape": "mape",
    "scaled_mspe": "mspe",
    "original_mae": "mae",
    "original_mse": "mse",
    "original_rmse": "rmse",
    "original_mape": "mape",
    "original_mspe": "mspe",
}
EVALUATION_METRICS = {"mae", "mse", "rmse", "mape", "mspe"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enforce_standardized_protocol(config: dict[str, Any]) -> None:
    if not bool(config.get("data", {}).get("scale", False)):
        raise ValueError("Project protocol requires data.scale=true: use train-split per-channel Z-score standardization.")
    if bool(config.get("model", {}).get("revin", {}).get("enabled", False)):
        raise ValueError(
            "Project protocol requires model inputs to remain in train Z-score space. "
            "Set model.revin.enabled=false for this baseline."
        )


def normalize_best_metric(config: dict[str, Any]) -> str:
    train_cfg = config.setdefault("train", {})
    raw_name = str(train_cfg.get("best_metric", "mse"))
    metric_name = METRIC_NAME_ALIASES.get(raw_name, raw_name)
    if metric_name not in EVALUATION_METRICS:
        raise ValueError(
            f"Unsupported train.best_metric: {raw_name}. "
            f"Use one of {sorted(EVALUATION_METRICS)}."
        )
    train_cfg["best_metric"] = metric_name
    return metric_name


def lr_scheduler_args(config: dict[str, Any]) -> SimpleNamespace:
    train_cfg = config["train"]
    return SimpleNamespace(
        lradj=str(train_cfg.get("lradj", "type1")),
        learning_rate=float(train_cfg["learning_rate"]),
        train_epochs=int(train_cfg["epochs"]),
    )


def adjust_distillation_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, config: dict[str, Any]) -> None:
    if str(config["train"].get("lradj", "type1")) == "constant":
        return
    adjust_learning_rate(optimizer, epoch, lr_scheduler_args(config))


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = config["train"].get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def dataset_spec(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    data_cfg = config["data"]
    spec = dict(data_cfg["datasets"][dataset_name])
    spec.setdefault("name", dataset_name)
    spec["root_path"] = str(resolve_path(data_cfg["root_path"]))
    spec["scale"] = bool(data_cfg.get("scale", True))
    spec["embed"] = data_cfg.get("embed", "timeF")
    return spec


def teacher_cache_args(config: dict[str, Any]) -> dict[str, Any]:
    teacher_cfg = config.get("teacher", {})
    cache_cfg = dict(teacher_cfg.get("cache", {}))
    if not cache_cfg:
        return {}
    return {
        "enabled": bool(cache_cfg.get("enabled", False)),
        "path": cache_cfg.get("path"),
        "dir": cache_cfg.get("dir", "outputs/teacher_cache/chronos2"),
        "name": cache_cfg.get("name"),
        "format": cache_cfg.get("format"),
        "validate_config": bool(cache_cfg.get("validate_config", True)),
        "required_fields": list(cache_cfg.get("required_fields", ["teacher_prediction"])),
        "required_hidden_targets": dict(cache_cfg.get("required_hidden_targets", {})),
        "required_hidden_specs": dict(cache_cfg.get("required_hidden_specs", {})),
        "shuffle_mode": cache_cfg.get("shuffle_mode", "chunk"),
        "chunk_size": int(cache_cfg.get("chunk_size", 2048)),
        "pin_memory": bool(cache_cfg.get("pin_memory", False)),
        "persistent_workers": bool(cache_cfg.get("persistent_workers", False)),
        "prefetch_factor": cache_cfg.get("prefetch_factor"),
        "expected_seq_len": data_context_len(config),
        "expected_student_seq_len": student_seq_len(config),
        "expected_data_context_length": data_context_len(config),
        "expected_pred_len": int(config["model"]["pred_len"]),
        "expected_label_len": int(config["data"]["label_len"]),
        "expected_teacher_context_length": teacher_context_len(config),
        "expected_teacher_prediction_length": int(config.get("teacher", {}).get("prediction_length") or config["model"]["pred_len"]),
    }


def student_seq_len(config: dict[str, Any]) -> int:
    return int(config["model"]["seq_len"])


def teacher_context_len(config: dict[str, Any]) -> int:
    return int(config.get("teacher", {}).get("context_length") or student_seq_len(config))


def data_context_len(config: dict[str, Any]) -> int:
    return max(student_seq_len(config), teacher_context_len(config))


def student_history(seq_x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    length = student_seq_len(config)
    if seq_x.shape[1] < length:
        raise ValueError(f"seq_x history length {seq_x.shape[1]} is shorter than model.seq_len={length}.")
    return seq_x[:, -length:, :]


def base_args(config: dict[str, Any]) -> SimpleNamespace:
    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    return SimpleNamespace(
        data=None,
        root_path=None,
        data_path=None,
        features="M",
        target=None,
        scale=bool(data_cfg.get("scale", True)),
        embed=data_cfg.get("embed", "timeF"),
        freq="h",
        seq_len=data_context_len(config),
        label_len=int(data_cfg["label_len"]),
        pred_len=int(model_cfg["pred_len"]),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        split=None,
        multi_dataset_strategy=data_cfg.get("multi_dataset_strategy", "round_robin"),
        teacher_cache=teacher_cache_args(config),
    )


def attach_teacher_cache_fingerprint(config: dict[str, Any], args: SimpleNamespace, spec: dict[str, Any]) -> None:
    teacher_cache = getattr(args, "teacher_cache", None)
    if not isinstance(teacher_cache, dict) or not teacher_cache.get("enabled", False):
        return
    dataset_name = spec.get("name", spec["data"])
    teacher_cache["base_fingerprint"] = base_cache_fingerprint(
        model_config=config["model"],
        data_config=config["data"],
        dataset_spec=spec,
        teacher_config=config.get("teacher", {}),
        dataset_name=dataset_name,
    )


def build_train_provider(config: dict[str, Any], dataset_names: list[str]):
    args = base_args(config)
    if len(dataset_names) == 1:
        spec = dataset_spec(config, dataset_names[0])
        attach_teacher_cache_fingerprint(config, args, spec)
        for key, value in spec.items():
            setattr(args, key, value)
        return data_provider(args, "train")

    args.train_datasets = []
    for name in dataset_names:
        spec = dataset_spec(config, name)
        args_for_spec = SimpleNamespace(**vars(args))
        attach_teacher_cache_fingerprint(config, args_for_spec, spec)
        spec = dict(spec)
        spec["teacher_cache"] = args_for_spec.teacher_cache
        args.train_datasets.append(spec)
    return data_provider(args, "train")


def build_eval_providers(config: dict[str, Any], dataset_names: list[str], flag: str) -> list[dict[str, Any]]:
    providers = []
    for dataset_name in dataset_names:
        args = base_args(config)
        args.seq_len = student_seq_len(config)
        args.teacher_cache = {"enabled": False}
        spec = dataset_spec(config, dataset_name)
        for key, value in spec.items():
            setattr(args, key, value)
        dataset, loader = data_provider(args, flag)
        if getattr(dataset, "is_teacher_cache", False):
            raise RuntimeError(f"{flag} evaluation must use the raw TSL-style dataset split, not teacher cache.")
        providers.append({"name": dataset_name, "dataset": dataset, "loader": loader})
    return providers


def unpack_train_item(item: Any, default_dataset: Any, default_name: str) -> tuple[str, Any, tuple]:
    if isinstance(item, dict):
        return item["name"], item["dataset"], item["batch"]
    return default_name, default_dataset, item


def batch_target(seq_y: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    label_len = int(config["data"]["label_len"])
    pred_len = int(config["model"]["pred_len"])
    return seq_y[:, label_len : label_len + pred_len, :]


def predict_full(
    model: torch.nn.Module,
    seq_x: torch.Tensor,
    channel_mask: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    seq_x = student_history(seq_x, config)
    revin_cfg = config["model"].get("revin", {})
    if revin_cfg.get("enabled", False):
        normalized_x, stats = revin_normalize(
            seq_x,
            eps=float(revin_cfg.get("eps", 1e-5)),
            detach_stats=bool(revin_cfg.get("detach_stats", True)),
        )
        normalized_prediction = model(normalized_x, channel_mask=channel_mask)
        return revin_denormalize(normalized_prediction, stats)
    return model(seq_x, channel_mask=channel_mask)


def forward_selected(
    model: torch.nn.Module,
    dataset: Any,
    seq_x: torch.Tensor,
    seq_y: torch.Tensor,
    channel_mask: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_x = seq_x.to(device)
    seq_y = seq_y.to(device)
    channel_mask = channel_mask.to(device)
    prediction = predict_full(model, seq_x, channel_mask, config)
    target = batch_target(seq_y, config)
    prediction = select_output(prediction, dataset)
    target = select_output(target, dataset)
    selected_mask = select_output_mask(channel_mask, dataset)
    return prediction, target, selected_mask


def train_one_epoch(
    model: torch.nn.Module,
    train_provider: tuple[Any, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    *,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.train()
    default_dataset, train_loader = train_provider
    default_name = getattr(default_dataset, "dataset_name", "train")
    total_loss = 0.0
    total_batches = 0
    log_interval = int(config["train"].get("log_interval", 50))
    expected_batches = len(train_loader)
    if max_batches is not None:
        expected_batches = min(expected_batches, int(max_batches))
    expected_batches = max(expected_batches, 1)

    for step, item in enumerate(train_loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        dataset_name, dataset, batch = unpack_train_item(item, default_dataset, default_name)
        seq_x, seq_y, _, _, channel_mask = batch
        prediction, target, selected_mask = forward_selected(model, dataset, seq_x, seq_y, channel_mask, device, config)
        loss = supervised_forecasting_loss(
            prediction,
            target,
            selected_mask,
            loss_name=config["train"].get("hard_loss", config["train"].get("loss", "mse")),
            smooth_l1_beta=float(config["train"].get("smooth_l1_beta", 1.0)),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        if step == 1 or step == expected_batches or (log_interval > 0 and step % log_interval == 0):
            avg_loss = total_loss / max(total_batches, 1)
            log(
                f"train {step}/{expected_batches} | dataset={dataset_name} | "
                f"loss={loss.item():.6g} | avg_loss={avg_loss:.6g}"
            )

    return {"train_loss": total_loss / max(total_batches, 1), "train_batches": total_batches}


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    provider: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
    *,
    max_batches: int | None,
) -> dict[str, float | int | str]:
    model.eval()
    dataset = provider["dataset"]
    loader = provider["loader"]
    log(f"eval {provider['name']} start")
    metric_sums = ErrorSums()
    batch_count = 0

    for step, batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        seq_x, seq_y, _, _, channel_mask = batch
        prediction, target, selected_mask = forward_selected(model, dataset, seq_x, seq_y, channel_mask, device, config)

        prediction_np = to_numpy(prediction)
        target_np = to_numpy(target)
        mask_np = to_numpy(selected_mask).astype(bool)
        metric_sums.update(prediction_np, target_np, mask_np)
        batch_count += 1

    metrics = metric_sums.as_metrics()
    row = {
        "dataset": provider["name"],
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "rmse": float(metrics["rmse"]),
        "mape": float(metrics["mape"]),
        "mspe": float(metrics["mspe"]),
        "count": int(metrics["count"]),
        "batches": batch_count,
    }
    log(
        f"eval {provider['name']} done | batches={batch_count} | "
        f"mse={row['mse']:.6g} | mae={row['mae']:.6g}"
    )
    return row


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    total_count = sum(int(row["count"]) for row in rows)
    denominator = max(total_count, 1)
    mse = sum(float(row["mse"]) * int(row["count"]) for row in rows) / denominator
    metrics: dict[str, float | int] = {
        "mae": sum(float(row["mae"]) * int(row["count"]) for row in rows) / denominator,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape": sum(float(row["mape"]) * int(row["count"]) for row in rows) / denominator,
        "mspe": sum(float(row["mspe"]) * int(row["count"]) for row in rows) / denominator,
        "count": total_count,
    }
    return metrics


@torch.no_grad()
def write_visualizations(
    model: torch.nn.Module,
    providers: list[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    if not config.get("visualization", {}).get("enabled", True):
        return []

    model.eval()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    max_samples = int(config["visualization"].get("num_samples", 1))
    dpi = int(config["visualization"].get("dpi", 150))
    value_space = config["visualization"].get("value_space", "scaled")
    if value_space not in {"scaled", "original"}:
        raise ValueError("visualization.value_space must be either 'scaled' or 'original'.")
    generated: list[str] = []

    for provider in providers:
        dataset_name = provider["name"]
        dataset = provider["dataset"]
        dataset_cfg = config["data"]["datasets"][dataset_name]
        channel_name = dataset_cfg.get("visualize_channel", dataset_cfg["target"])
        feature_names = list(getattr(dataset, "feature_names", []))
        channel_idx = feature_names.index(channel_name) if channel_name in feature_names else dataset.target_idx

        batch = next(iter(provider["loader"]))
        seq_x, seq_y, _, _, channel_mask = batch
        prediction, target, _ = forward_selected(model, dataset, seq_x, seq_y, channel_mask, device, config)
        full_prediction = predict_full(
            model,
            seq_x.to(device),
            channel_mask.to(device),
            config,
        ).detach().cpu().numpy()
        target_future = batch_target(seq_y, config).numpy()
        history = student_history(seq_x, config).numpy()

        for sample_idx in range(min(max_samples, history.shape[0])):
            history_values = history[sample_idx, :, channel_idx]
            truth_values = target_future[sample_idx, :, channel_idx]
            pred_values = full_prediction[sample_idx, :, channel_idx]
            scale_label = "train Z-score"
            if value_space == "original":
                history_values = inverse_transform_channel(dataset, history_values, channel_idx)
                truth_values = inverse_transform_channel(dataset, truth_values, channel_idx)
                pred_values = inverse_transform_channel(dataset, pred_values, channel_idx)
                scale_label = "original scale"
            path = plot_dir / f"{dataset_name}_{channel_name}_sample{sample_idx}.png"
            plot_forecast_sample(
                path,
                history=history_values,
                truth=truth_values,
                prediction=pred_values,
                channel_name=channel_name,
                title=f"{dataset_name} {channel_name} student MLP forecast ({scale_label})",
                dpi=dpi,
                scale_label=scale_label,
                connect_from_history=True,
            )
            csv_path = plot_dir / f"{dataset_name}_{channel_name}_sample{sample_idx}.csv"
            write_forecast_sample_csv(
                csv_path,
                history=history_values,
                truth=truth_values,
                prediction=pred_values,
            )
            generated.append(str(path))

        del prediction, target

    return generated


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict[str, Any], epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = dict(config["model"])
    model_cfg.pop("revin", None)
    architecture = str(model_cfg.pop("architecture", "channel_independent_mlp")).lower()
    if architecture in {"lmpnet", "student_lmpnet"}:
        from models.student_lmpnet import StudentLMPNet, StudentLMPNetConfig
        return StudentLMPNet(StudentLMPNetConfig(**model_cfg))
    if architecture in {"patchtsmixer", "student_patchtsmixer"}:
        from models.student_patchtsmixer import StudentPatchTSMixer, StudentPatchTSMixerConfig
        return StudentPatchTSMixer(StudentPatchTSMixerConfig(**model_cfg))
    if architecture in {"itransformer", "student_itransformer"}:
        from models.student_itransformer import StudentITransformer, StudentITransformerConfig
        return StudentITransformer(StudentITransformerConfig(**model_cfg))
    if architecture in {"channel_independent_mlp", "student_mlp", "mlp"}:
        return ChannelIndependentMLP(ChannelIndependentMLPConfig(**model_cfg))
    if architecture in {"three_tap_mlp", "student_mlp_three_tap", "three_layer_mlp"}:
        return ThreeTapChannelIndependentMLP(ThreeTapChannelIndependentMLPConfig(**model_cfg))
    raise ValueError(f"Unsupported student model architecture: {architecture}")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.max_train_batches is not None:
        config["train"]["max_train_batches"] = args.max_train_batches
    if args.max_eval_batches is not None:
        config["train"]["max_eval_batches"] = args.max_eval_batches
    lradj_override = getattr(args, "lradj", None)
    if lradj_override is not None:
        config["train"]["lradj"] = lradj_override
    loss_override = getattr(args, "loss", None)
    if loss_override is not None:
        config["train"]["loss"] = loss_override
        config["train"]["hard_loss"] = loss_override
    if args.datasets is not None:
        config["data"]["enabled_datasets"] = args.datasets


def run(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    apply_cli_overrides(config, args)
    normalize_best_metric(config)
    enforce_standardized_protocol(config)
    set_seed(int(config["train"]["seed"]))

    output_dir = resolve_path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    enabled_datasets = list(config["data"]["enabled_datasets"])
    device = choose_device(config)
    log(
        f"Student MLP training started | datasets={enabled_datasets} | device={device} | "
        f"epochs={config['train']['epochs']} | batch_size={config['train']['batch_size']} | "
        f"revin={config['model'].get('revin', {}).get('enabled', False)}"
    )

    if config["model"].get("channel_weight_mode", "shared") != "shared" and len(enabled_datasets) > 1:
        raise ValueError("Multi-dataset training requires channel_weight_mode='shared' for variable channel counts.")

    train_provider = build_train_provider(config, enabled_datasets)
    val_providers = build_eval_providers(config, enabled_datasets, "val")
    test_providers = build_eval_providers(config, enabled_datasets, "test")

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    checkpoints_dir = output_dir / "checkpoints"
    history_rows: list[dict[str, Any]] = []
    best_metric_name = normalize_best_metric(config)
    best_value = float("inf")
    best_epoch = -1
    patience = int(config["train"].get("early_stopping_patience", 0))
    stale_epochs = 0

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        log(f"epoch {epoch}/{config['train']['epochs']} start")
        train_metrics = train_one_epoch(
            model,
            train_provider,
            optimizer,
            device,
            config,
            max_batches=config["train"].get("max_train_batches"),
        )
        val_rows = [
            evaluate_dataset(
                model,
                provider,
                device,
                config,
                max_batches=config["train"].get("max_eval_batches"),
            )
            for provider in val_providers
        ]
        val_metrics = aggregate_metrics(val_rows)
        current_value = float(val_metrics[best_metric_name])
        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history_rows.append(row)
        log(
            f"epoch {epoch}/{config['train']['epochs']} done | "
            f"train_loss={train_metrics['train_loss']:.6g} | "
            f"val_mae={val_metrics['mae']:.6g} | "
            f"val_mse={val_metrics['mse']:.6g}"
        )

        if current_value < best_value:
            best_value = current_value
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(checkpoints_dir / "best.pt", model, config, epoch, val_metrics)
            log(f"new best | epoch={epoch} | {best_metric_name}={best_value:.6g}")
        else:
            stale_epochs += 1
            log(f"no improvement | stale_epochs={stale_epochs}/{patience}")

        if patience > 0 and stale_epochs >= patience:
            log(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break

        adjust_distillation_learning_rate(optimizer, epoch, config)

    last_metrics = history_rows[-1] if history_rows else {}
    save_checkpoint(checkpoints_dir / "last.pt", model, config, int(last_metrics.get("epoch", 0)), last_metrics)
    pd.DataFrame(history_rows).to_csv(output_dir / "epoch_metrics.csv", index=False)

    if (checkpoints_dir / "best.pt").exists():
        checkpoint = torch.load(checkpoints_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log(f"loaded best checkpoint from epoch {checkpoint['epoch']} for final test")

    log("final test start")
    test_rows = [
        evaluate_dataset(
            model,
            provider,
            device,
            config,
            max_batches=config["train"].get("max_eval_batches"),
        )
        for provider in test_providers
    ]
    test_metrics = aggregate_metrics(test_rows)
    log(
        f"final test done | mae={test_metrics['mae']:.6g} | "
        f"mse={test_metrics['mse']:.6g}"
    )
    pd.DataFrame(test_rows).to_csv(output_dir / "test_metrics_by_dataset.csv", index=False)
    plot_paths = write_visualizations(model, test_providers, device, config, output_dir)

    final_metrics = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "enabled_datasets": enabled_datasets,
        "best_epoch": best_epoch,
        "best_validation_metric": {best_metric_name: best_value},
        "test": test_metrics,
        "test_by_dataset": test_rows,
        "visualizations": plot_paths,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"training finished. metrics written to {output_dir / 'metrics.json'}")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the channel-independent MLP student baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--lradj", choices=["type1", "type2", "type3", "cosine"], default=None)
    parser.add_argument("--loss", choices=["mse", "mae", "smooth_l1", "huber"], default=None)
    parser.add_argument("--datasets", nargs="+", choices=["ECL", "ETTh1", "ETTm1", "WTH"])
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run(resolve_path(cli_args.config), cli_args)
