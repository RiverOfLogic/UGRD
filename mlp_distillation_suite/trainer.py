from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from distillation_common.teacher_cache import cache_enabled
from distillation_common.teacher_backend import teacher_backend, teacher_structure
from mlp_distillation_suite.config import prepare_suite_config
from mlp_distillation_suite.hidden import (
    build_projectors_for_layers,
    compute_hidden_distillation_loss,
    distillation_mode,
    student_forward_selected_with_hiddens,
)
from mlp_distillation_suite.teacher import load_teacher, teacher_predict_for_suite
from mlp_distillation_suite.on_policy import short_rollout_loss
from mlp_distillation_suite.schedules import hidden_weight_at_epoch
from distillation_common.student_mlp_common import (
    aggregate_metrics,
    adjust_distillation_learning_rate,
    build_eval_providers,
    build_model,
    build_train_provider,
    choose_device,
    evaluate_dataset,
    load_config,
    resolve_path,
    set_seed,
    unpack_train_item,
    normalize_best_metric,
    predict_full,
    write_visualizations,
)
from utils.losses import soft_distillation_loss, supervised_forecasting_loss
from utils.metrics import select_output, select_output_mask


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _output_dir(config: dict[str, Any], config_path: Path) -> Path:
    output_cfg = config.setdefault("output", {})
    if output_cfg.get("dir"):
        return resolve_path(output_cfg["dir"])
    return resolve_path(Path("outputs") / "mlp_distillation_suite" / config_path.stem)


def _output_dir_for_itr(base_dir: Path, args: Any) -> Path:
    itr = int(getattr(args, "itr", 1) or 1)
    if itr <= 1:
        return base_dir
    itr_index = int(getattr(args, "itr_index", 0) or 0)
    return base_dir / f"itr{itr_index}"


def _validate_teacher_lengths(config: dict[str, Any]) -> None:
    teacher_pred_len = int(config["teacher"].get("prediction_length") or config["model"]["pred_len"])
    if teacher_pred_len != int(config["model"]["pred_len"]):
        raise ValueError("teacher.prediction_length must match model.pred_len for MLP distillation.")
    teacher_context_len = int(config["teacher"].get("context_length") or config["model"]["seq_len"])
    model_seq_len = int(config["model"]["seq_len"])
    if teacher_context_len < model_seq_len:
        raise ValueError(
            "teacher.context_length must be greater than or equal to model.seq_len. "
            "When it is larger, the teacher sees the full history and the student uses the last model.seq_len steps."
        )


def compute_soft_loss(
    prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    selected_mask: torch.Tensor,
    distill_cfg: dict[str, Any],
    teacher_quantiles: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    soft_losses = distill_cfg.get("soft_losses")
    if not soft_losses:
        loss_name = distill_cfg.get("soft_loss", "mse")
        loss = soft_distillation_loss(
            prediction,
            teacher_prediction,
            selected_mask,
            loss_name=loss_name,
            distilts_tau=float(distill_cfg.get("distilts_tau", 1.0)),
            time_distill_num_downsamples=int(distill_cfg.get("num_downsamples", distill_cfg.get("time_distill_num_downsamples", 3))),
            time_distill_period_temperature=float(distill_cfg.get("period_temperature", distill_cfg.get("time_distill_period_temperature", 0.5))),
            time_distill_scale_weight=float(distill_cfg.get("scale_weight", distill_cfg.get("time_distill_scale_weight", 1.0))),
            time_distill_period_weight=float(distill_cfg.get("period_weight", distill_cfg.get("time_distill_period_weight", 1.0))),
            time_distill_down_sampling_window=int(distill_cfg.get("down_sampling_window", distill_cfg.get("time_distill_down_sampling_window", 2))),
            uncertainty_weight=teacher_quantiles,
        )
        return loss, {f"soft_{loss_name}_loss": loss}

    total = torch.zeros((), device=prediction.device)
    parts: dict[str, torch.Tensor] = {}
    for item in soft_losses:
        name = str(item["name"])
        weight = float(item.get("weight", 1.0))
        loss = soft_distillation_loss(
            prediction,
            teacher_prediction,
            selected_mask,
            loss_name=name,
            distilts_tau=float(item.get("tau", item.get("distilts_tau", distill_cfg.get("distilts_tau", 1.0)))),
            time_distill_num_downsamples=int(item.get("num_downsamples", item.get("time_distill_num_downsamples", distill_cfg.get("time_distill_num_downsamples", 3)))),
            time_distill_period_temperature=float(item.get("period_temperature", item.get("time_distill_period_temperature", distill_cfg.get("time_distill_period_temperature", 0.5)))),
            time_distill_scale_weight=float(item.get("scale_weight", item.get("time_distill_scale_weight", distill_cfg.get("time_distill_scale_weight", 1.0)))),
            time_distill_period_weight=float(item.get("period_weight", item.get("time_distill_period_weight", distill_cfg.get("time_distill_period_weight", 1.0)))),
            time_distill_down_sampling_window=int(item.get("down_sampling_window", item.get("time_distill_down_sampling_window", distill_cfg.get("time_distill_down_sampling_window", 2)))),
            uncertainty_weight=teacher_quantiles,
        )
        parts[f"soft_{name}_loss"] = loss
        total = total + weight * loss
    return total, parts

def compute_hard_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected_mask: torch.Tensor,
    distill_cfg: dict[str, Any],
    config: dict[str, Any],
    teacher_quantiles: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hard_losses = distill_cfg.get("hard_losses")
    if not hard_losses:
        loss_name = distill_cfg.get("hard_loss", config["train"].get("loss", "mse"))
        loss = supervised_forecasting_loss(
            prediction,
            target,
            selected_mask,
            loss_name=loss_name,
            uncertainty_weight=teacher_quantiles,
        )
        return loss, {f"hard_{loss_name}_loss": loss}

    total = torch.zeros((), device=prediction.device)
    parts: dict[str, torch.Tensor] = {}
    for item in hard_losses:
        name = str(item["name"])
        weight = float(item.get("weight", 1.0))
        loss = supervised_forecasting_loss(
            prediction,
            target,
            selected_mask,
            loss_name=name,
            uncertainty_weight=teacher_quantiles,
        )
        parts[f"hard_{name}_loss"] = loss
        total = total + weight * loss
    return total, parts


def train_one_epoch(
    model: torch.nn.Module,
    projectors: torch.nn.ModuleDict,
    teacher: Any | None,
    train_provider: tuple[Any, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    layers: list[dict[str, Any]],
    *,
    max_batches: int | None,
    epoch: int = 1,
    extra_objective=None,
) -> dict[str, float | int]:
    model.train()
    projectors.train()
    default_dataset, train_loader = train_provider
    default_name = getattr(default_dataset, "dataset_name", "train")
    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    total_hidden_loss = 0.0
    total_on_policy_loss = 0.0
    on_policy_queries = 0
    on_policy_samples = 0
    hard_part_totals: dict[str, float] = {}
    soft_part_totals: dict[str, float] = {}
    per_layer_totals = {layer["name"]: 0.0 for layer in layers}
    total_batches = 0

    log_interval = int(config["train"].get("log_interval", 50))
    expected_batches = len(train_loader)
    if max_batches is not None:
        expected_batches = min(expected_batches, int(max_batches))
    expected_batches = max(expected_batches, 1)

    distill_cfg = config["distillation"]
    hard_weight = float(distill_cfg.get("hard_weight", 1.0))
    soft_weight = float(distill_cfg.get("soft_weight", 0.5))
    hidden_weight = hidden_weight_at_epoch(distill_cfg, epoch)
    hidden_loss_name = distill_cfg.get("hidden_loss", "mse")
    opd = config.get("on_policy", {})
    opd_active = bool(opd.get("enabled", False)) and epoch >= int(opd.get("start_epoch", 1))

    for step, item in enumerate(train_loader, start=1):
        if max_batches is not None and step > max_batches:
            break

        if isinstance(item, dict) and "seq_x" in item:
            dataset_name, dataset, batch = default_name, default_dataset, item
        else:
            dataset_name, dataset, batch = unpack_train_item(item, default_dataset, default_name)
        if isinstance(batch, dict):
            seq_x = batch["seq_x"]
            seq_y = batch["seq_y"]
            channel_mask = batch["channel_mask"]
            teacher_prediction = batch["teacher_prediction"].to(device)
            teacher_quantiles = {}
            teacher_q01 = batch.get("teacher_q01")
            teacher_q09 = batch.get("teacher_q09")
            if teacher_q01 is not None and teacher_q09 is not None:
                teacher_quantiles = {
                    "teacher_q01": teacher_q01.to(device),
                    "teacher_q09": teacher_q09.to(device),
                }
            teacher_hiddens = {
                name: value.to(device)
                for name, value in batch.get("teacher_hiddens", {}).items()
            }
        else:
            if teacher is None:
                raise RuntimeError("Chronos2 teacher is required when teacher cache is disabled.")
            seq_x, seq_y, _, _, channel_mask = batch
            teacher_prediction, teacher_hiddens, teacher_quantiles = teacher_predict_for_suite(
                teacher,
                dataset,
                seq_x,
                config,
                layers,
                device,
            )

        prediction, target, selected_mask, student_hiddens = student_forward_selected_with_hiddens(
            model,
            dataset,
            seq_x,
            seq_y,
            channel_mask,
            device,
            config,
            layers,
            extra_student_keys=getattr(extra_objective, "student_keys", None),
        )
        if teacher_prediction.shape != prediction.shape:
            raise ValueError(
                f"Teacher prediction shape {tuple(teacher_prediction.shape)} does not match "
                f"student prediction shape {tuple(prediction.shape)}."
            )
        if teacher_quantiles:
            for name, value in teacher_quantiles.items():
                if value.shape != prediction.shape:
                    raise ValueError(
                        f"Teacher quantile {name} shape {tuple(value.shape)} does not match "
                        f"student prediction shape {tuple(prediction.shape)}."
                    )

        hard_loss, hard_loss_parts = compute_hard_loss(
            prediction,
            target,
            selected_mask,
            distill_cfg,
            config,
            teacher_quantiles=teacher_quantiles,
        )
        soft_loss, soft_loss_parts = compute_soft_loss(
            prediction,
            teacher_prediction,
            selected_mask,
            distill_cfg,
            teacher_quantiles=teacher_quantiles,
        )
        hidden_loss, layer_loss_values = compute_hidden_distillation_loss(
            projectors,
            layers,
            dataset_name,
            student_hiddens,
            teacher_hiddens,
            selected_mask,
            hidden_loss_name=hidden_loss_name,
            log_fn=log,
            step=step,
        )
        loss = hard_weight * hard_loss + soft_weight * soft_loss + hidden_weight * hidden_loss
        if extra_objective is not None:
            extra_kwargs = {"student_hiddens": student_hiddens} if getattr(extra_objective, "student_keys", None) else {}
            extra_loss, extra_parts = extra_objective(
                prediction, selected_mask, batch, teacher_prediction, teacher_quantiles, **extra_kwargs
            )
            loss = extra_loss if getattr(extra_objective, "replaces_base_loss", False) else loss + extra_loss
            soft_loss_parts.update(extra_parts)
        on_policy_loss = loss.new_zeros(())
        if opd_active and (step - 1) % int(opd.get("every_n_batches", 4)) == 0:
            if teacher is None:
                raise RuntimeError("On-policy histories require a live teacher even when real-history cache is enabled.")
            size = min(seq_x.shape[0], int(opd.get("max_samples", 8)))
            history = seq_x[:size].to(device)
            mask = channel_mask[:size].to(device)

            def fresh_labels(visited):
                median, _, quantiles = teacher_predict_for_suite(teacher, dataset, visited, config, [], device)
                return median, quantiles

            def output_loss(full_prediction, labels, full_mask):
                median, quantiles = labels
                return compute_soft_loss(select_output(full_prediction, dataset), median,
                                         select_output_mask(full_mask, dataset), distill_cfg,
                                         teacher_quantiles=quantiles)[0]

            on_policy_loss, details = short_rollout_loss(
                model, history, mask,
                block_size=int(opd.get("block_size", 16)),
                rollout_steps=int(opd.get("rollout_steps", 1)),
                predict=lambda context, valid: predict_full(model, context, valid, config),
                teacher_label=fresh_labels, distillation_loss=output_loss,
            )
            loss = loss + float(opd.get("weight", 0.1)) * on_policy_loss
            on_policy_queries += int(details["teacher_queries"])
            on_policy_samples += int(details["samples"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_hard_loss += float(hard_loss.detach().cpu().item())
        for name, value in hard_loss_parts.items():
            hard_part_totals[name] = hard_part_totals.get(name, 0.0) + float(value.detach().cpu().item())
        total_soft_loss += float(soft_loss.detach().cpu().item())
        for name, value in soft_loss_parts.items():
            soft_part_totals[name] = soft_part_totals.get(name, 0.0) + float(value.detach().cpu().item())
        total_hidden_loss += float(hidden_loss.detach().cpu().item())
        total_on_policy_loss += float(on_policy_loss.detach().cpu().item())
        for name, layer_loss in layer_loss_values.items():
            per_layer_totals[name] += float(layer_loss.detach().cpu().item())
        total_batches += 1

        if step == 1 or step == expected_batches or (log_interval > 0 and step % log_interval == 0):
            layer_summary = " | ".join(f"{name}={value.item():.6g}" for name, value in layer_loss_values.items())
            hard_summary = " | ".join(f"{name}={value.item():.6g}" for name, value in hard_loss_parts.items())
            soft_summary = " | ".join(f"{name}={value.item():.6g}" for name, value in soft_loss_parts.items())
            suffix = f" | {layer_summary}" if layer_summary else ""
            if soft_summary:
                suffix = f" | {soft_summary}{suffix}"
            if hard_summary:
                suffix = f" | {hard_summary}{suffix}"
            log(
                f"train {step}/{expected_batches} | dataset={dataset_name} | total={loss.item():.6g} | "
                f"hard={hard_loss.item():.6g} | soft={soft_loss.item():.6g} | "
                f"hidden={hidden_loss.item():.6g} ({hidden_loss_name}) | hidden_weight={hidden_weight:g}{suffix}"
                f" | on_policy={on_policy_loss.item():.6g}"
            )

    denom = max(total_batches, 1)
    metrics: dict[str, float | int] = {
        "train_loss": total_loss / denom,
        "train_hard_loss": total_hard_loss / denom,
        "train_soft_loss": total_soft_loss / denom,
        "train_hidden_loss": total_hidden_loss / denom,
        "train_weighted_hidden_loss": hidden_weight * total_hidden_loss / denom,
        "train_hidden_weight": hidden_weight,
        "train_batches": total_batches,
        "train_on_policy_loss": total_on_policy_loss / denom,
        "train_on_policy_teacher_queries": on_policy_queries,
        "train_on_policy_samples": on_policy_samples,
    }
    for name, value in hard_part_totals.items():
        metrics[f"train_{name}"] = value / denom
    for name, value in per_layer_totals.items():
        metrics[f"train_hidden_{name}_loss"] = value / denom
    for name, value in soft_part_totals.items():
        metrics[f"train_{name}"] = value / denom
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    projectors: torch.nn.ModuleDict,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "projectors_state_dict": projectors.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def forward_eval(provider: dict[str, Any], model: torch.nn.Module, device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    return evaluate_dataset(
        model,
        provider,
        device,
        config,
        max_batches=config["train"].get("max_eval_batches"),
    )


def run(config_path: Path, args: Any, *, objective_factory=None, train_provider_factory=None) -> None:
    config = load_config(config_path)
    layers = prepare_suite_config(config, args)
    _validate_teacher_lengths(config)
    run_seed = int(getattr(args, "run_seed", 2026))
    config.setdefault("train", {})["seed"] = run_seed
    config["train"]["itr_index"] = int(getattr(args, "itr_index", 0) or 0)
    config["train"]["itr"] = int(getattr(args, "itr", 1) or 1)
    set_seed(int(config["train"]["seed"]))

    output_dir = _output_dir_for_itr(_output_dir(config, config_path), args)
    config["output"]["dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    enabled_datasets = list(config["data"]["enabled_datasets"])
    device = choose_device(config)
    if objective_factory is None and config.get("heterogeneous_ftr", {}).get("enabled", False):
        from extensions.heterogeneous_ftr_v2.objective import create_objective, paired_provider

        objective_factory, train_provider_factory = create_objective, paired_provider
    if objective_factory is None and config.get("ftr_pd_v2", {}).get("enabled", False):
        from extensions.timestral_ftr_v2.config import validate_config
        from extensions.timestral_ftr_v2.objective import create_objective

        validate_config(config)
        objective_factory = create_objective
    extra_objective = objective_factory(config, device) if objective_factory else None
    mode = distillation_mode(layers)
    use_teacher_cache = cache_enabled(config)
    log(
        f"MLP distillation suite started | mode={mode} | datasets={enabled_datasets} | "
        f"device={device} | epochs={config['train']['epochs']} | "
        f"itr={config['train']['itr_index']}/{config['train']['itr']} | seed={run_seed} | "
        f"teacher_cache={'enabled' if use_teacher_cache else 'disabled'}"
    )

    if config["model"].get("channel_weight_mode", "shared") != "shared" and len(enabled_datasets) > 1:
        raise ValueError("Multi-dataset training requires channel_weight_mode='shared' for variable channel counts.")

    train_provider = (train_provider_factory or build_train_provider)(config, enabled_datasets)
    val_providers = build_eval_providers(config, enabled_datasets, "val")
    validation_only = bool(config["train"].get("validation_only", False))
    test_providers = [] if validation_only else build_eval_providers(config, enabled_datasets, "test")

    teacher = None
    num_blocks = None
    if use_teacher_cache and not config.get("on_policy", {}).get("enabled", False):
        log(f"Using offline {teacher_backend(config)} teacher cache; skipping teacher model load")
    else:
        log(f"Loading {teacher_backend(config)} teacher")
        teacher = load_teacher(config)
        _, _, num_blocks = teacher_structure(teacher, config)
        if teacher_backend(config) == "chronos2" and layers and num_blocks != 12:
            raise ValueError(f"Expected Chronos2 to have 12 encoder blocks, got {num_blocks}.")

    model = build_model(config).to(device)
    if extra_objective is not None and hasattr(extra_objective, "on_model_ready"):
        extra_objective.on_model_ready(model)
    projectors = build_projectors_for_layers(layers, config)
    if extra_objective is not None and hasattr(extra_objective, 'configure_projectors'):
        extra_objective.configure_projectors(projectors)
    projectors = projectors.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projectors.parameters()),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
        foreach=config["train"].get("optimizer_foreach"),
    )

    checkpoints_dir = output_dir / "checkpoints"
    history_rows: list[dict[str, Any]] = []
    best_metric_name = normalize_best_metric(config)
    best_value = float("inf")
    best_epoch = -1
    patience = int(config["train"].get("early_stopping_patience", 0))
    stale_epochs = 0

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        sampler = getattr(train_provider[1], "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        log(f"epoch {epoch}/{config['train']['epochs']} start")
        train_metrics = train_one_epoch(
            model,
            projectors,
            teacher,
            train_provider,
            optimizer,
            device,
            config,
            layers,
            max_batches=config["train"].get("max_train_batches"),
            epoch=epoch,
            extra_objective=extra_objective,
        )
        val_rows = [forward_eval(provider, model, device, config) for provider in val_providers]
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
            f"train_hard={train_metrics['train_hard_loss']:.6g} | "
            f"train_soft={train_metrics['train_soft_loss']:.6g} | "
            f"train_hidden={train_metrics['train_hidden_loss']:.6g} | "
            f"val_mse={val_metrics['mse']:.6g}"
        )

        if current_value < best_value:
            best_value = current_value
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(checkpoints_dir / "best.pt", model, projectors, config, epoch, val_metrics)
            log(f"new best | epoch={epoch} | {best_metric_name}={best_value:.6g}")
        else:
            stale_epochs += 1
            log(f"no improvement | stale_epochs={stale_epochs}/{patience}")

        if patience > 0 and stale_epochs >= patience:
            log(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break

        adjust_distillation_learning_rate(optimizer, epoch, config)

    last_metrics = history_rows[-1] if history_rows else {}
    save_checkpoint(
        checkpoints_dir / "last.pt",
        model,
        projectors,
        config,
        int(last_metrics.get("epoch", 0)),
        last_metrics,
    )
    pd.DataFrame(history_rows).to_csv(output_dir / "epoch_metrics.csv", index=False)

    if validation_only:
        final_metrics = {
            "config_path": str(config_path), "output_dir": str(output_dir), "seed": run_seed,
            "validation_only": True, "best_epoch": best_epoch,
            "best_validation_metric": {best_metric_name: best_value},
            "trained_epochs": len(history_rows), "distillation": config["distillation"],
        }
        (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        log("Validation-only run finished; no test provider was built or evaluated")
        return

    if (checkpoints_dir / "best.pt").exists():
        checkpoint = torch.load(checkpoints_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        projectors.load_state_dict(checkpoint.get("projectors_state_dict", {}))
        log(f"loaded best checkpoint from epoch {checkpoint['epoch']} for final test")

    log("final test start")
    test_rows = [forward_eval(provider, model, device, config) for provider in test_providers]
    test_metrics = aggregate_metrics(test_rows)
    pd.DataFrame(test_rows).to_csv(output_dir / "test_metrics_by_dataset.csv", index=False)
    plot_paths = write_visualizations(model, test_providers, device, config, output_dir)

    final_metrics = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "enabled_datasets": enabled_datasets,
        "seed": run_seed,
        "itr_index": config["train"]["itr_index"],
        "itr": config["train"]["itr"],
        "distillation_mode": mode,
        "distillation": config["distillation"],
        "on_policy": config.get("on_policy", {"enabled": False}),
        "hidden_distillation": {"enabled": bool(layers), "layers": layers},
        "teacher": {
            "num_encoder_blocks": num_blocks,
            "prediction_length": config["teacher"].get("prediction_length"),
            "context_length": config["teacher"].get("context_length"),
            "max_series_per_forward": config["teacher"].get("max_series_per_forward"),
            "cache": config["teacher"].get("cache", {}),
        },
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
    log(
        f"final test done | mae={test_metrics['mae']:.6g} | "
        f"mse={test_metrics['mse']:.6g}"
    )
    log(f"training finished. metrics written to {output_dir / 'metrics.json'}")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
