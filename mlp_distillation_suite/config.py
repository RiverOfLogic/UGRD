from __future__ import annotations

from argparse import Namespace
from typing import Any

from distillation_common.student_mlp_common import apply_cli_overrides, enforce_standardized_protocol, normalize_best_metric
from distillation_common.teacher_cache import cache_cli_overrides


PREDICTION_LOSSES = {
    "mse",
    "distilts",
    "distilts_mse",
    "distilts_horizon_mse",
    "distilts_horizon_weighted_mse",
    "time_distill_prediction",
    "timedistill",
    "timedistill_prediction",
    "wmse",
    "mae",
    "smooth_l1",
    "huber",
    "multi_scale_mse",
    "multiscale_mse",
    "ms_mse",
    "quantile_penalty_loss",
    "period_distribution_kl",
    "period_kl",
    "frequency_distribution_kl",
}
HIDDEN_LOSSES = {"mse", "cosine"}
UNCERTAINTY_WEIGHTED_LOSSES = {"quantile_penalty_loss"}


def _normalize_weighted_prediction_losses(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    normalized = []
    for item in value:
        if isinstance(item, str):
            normalized.append({"name": item, "weight": 1.0})
        elif isinstance(item, dict):
            entry = dict(item)
            entry.setdefault("weight", 1.0)
            if "name" not in entry:
                raise ValueError(f"Each distillation.{field_name} entry must define a name.")
            normalized.append(entry)
        else:
            raise ValueError(f"distillation.{field_name} entries must be strings or mappings.")
    if not normalized:
        raise ValueError(f"distillation.{field_name} must not be empty when provided.")
    return normalized


def apply_suite_overrides(config: dict[str, Any], args: Namespace) -> None:
    apply_cli_overrides(config, args)
    cache_cli_overrides(config, args)
    soft_loss = getattr(args, "soft_loss", None)
    if soft_loss is not None:
        config.setdefault("distillation", {})["soft_loss"] = soft_loss
    hard_loss = getattr(args, "hard_loss", None)
    if hard_loss is not None:
        config.setdefault("distillation", {})["hard_loss"] = hard_loss


def normalize_distillation_config(config: dict[str, Any]) -> None:
    normalize_best_metric(config)
    distill_cfg = config.setdefault("distillation", {})
    if "soft_loss" not in distill_cfg and "loss" in distill_cfg:
        distill_cfg["soft_loss"] = distill_cfg["loss"]
    if "soft_weight" not in distill_cfg and "weight" in distill_cfg:
        distill_cfg["soft_weight"] = distill_cfg["weight"]
    distill_cfg.setdefault("hard_weight", 1.0)
    distill_cfg.setdefault("soft_weight", 0.5)
    distill_cfg.setdefault("hidden_weight", 0.0)
    distill_cfg.setdefault("hard_loss", config.get("train", {}).get("loss", "mse"))
    distill_cfg.setdefault("soft_loss", "mse")
    distill_cfg.setdefault("hidden_loss", "mse")
    if "hard_losses" in distill_cfg:
        distill_cfg["hard_losses"] = _normalize_weighted_prediction_losses(
            distill_cfg["hard_losses"],
            field_name="hard_losses",
        )
    if "soft_losses" in distill_cfg:
        distill_cfg["soft_losses"] = _normalize_weighted_prediction_losses(
            distill_cfg["soft_losses"],
            field_name="soft_losses",
        )

    hidden_cfg = config.setdefault("hidden_distillation", {})
    hidden_cfg.setdefault("enabled", bool(hidden_cfg.get("layers", [])))
    hidden_cfg.setdefault("layers", [])


def validate_distillation_config(config: dict[str, Any]) -> None:
    distill_cfg = config["distillation"]
    hard_loss = str(distill_cfg.get("hard_loss", "mse"))
    if hard_loss not in PREDICTION_LOSSES:
        raise ValueError(f"Unsupported distillation.hard_loss: {hard_loss}")
    soft_loss = str(distill_cfg.get("soft_loss", "mse"))
    if soft_loss not in PREDICTION_LOSSES:
        raise ValueError(f"Unsupported distillation.soft_loss: {soft_loss}")
    for item in distill_cfg.get("hard_losses", []):
        name = str(item["name"])
        if name not in PREDICTION_LOSSES:
            raise ValueError(f"Unsupported distillation.hard_losses item: {name}")
        if float(item.get("weight", 1.0)) < 0.0:
            raise ValueError("distillation.hard_losses weights must be non-negative.")
    for item in distill_cfg.get("soft_losses", []):
        name = str(item["name"])
        if name not in PREDICTION_LOSSES:
            raise ValueError(f"Unsupported distillation.soft_losses item: {name}")
        if float(item.get("weight", 1.0)) < 0.0:
            raise ValueError("distillation.soft_losses weights must be non-negative.")
    hidden_loss = str(distill_cfg.get("hidden_loss", "mse"))
    if hidden_loss not in HIDDEN_LOSSES:
        raise ValueError(f"Unsupported distillation.hidden_loss: {hidden_loss}")
    from mlp_distillation_suite.schedules import hidden_weight_at_epoch
    hidden_weight_at_epoch(distill_cfg, 1)
    opd = config.get("on_policy", {})
    if opd.get("enabled", False):
        import math

        weight = float(opd.get("weight", 0.1))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("on_policy.weight must be finite and positive.")
        block = int(opd.get("block_size", 16))
        if not 1 <= block <= min(int(config["model"]["seq_len"]), int(config["model"]["pred_len"])):
            raise ValueError("on_policy.block_size must fit the student history and forecast.")
        for key, default in (("rollout_steps", 1), ("every_n_batches", 4), ("max_samples", 8), ("start_epoch", 1)):
            if int(opd.get(key, default)) < 1:
                raise ValueError(f"on_policy.{key} must be positive.")


def distillation_uses_teacher_uncertainty(config: dict[str, Any]) -> bool:
    distill_cfg = config.get("distillation", {})
    for key in ("hard_loss", "soft_loss"):
        if str(distill_cfg.get(key, "")) in UNCERTAINTY_WEIGHTED_LOSSES:
            return True
    for key in ("hard_losses", "soft_losses"):
        for item in distill_cfg.get(key, []):
            if str(item["name"]) in UNCERTAINTY_WEIGHTED_LOSSES:
                return True
    return False


def configure_teacher_cache_requirements(config: dict[str, Any], layers: list[dict[str, Any]]) -> None:
    cache_cfg = config.setdefault("teacher", {}).setdefault("cache", {})
    required_fields = list(cache_cfg.get("required_fields", ["teacher_prediction"]))
    if distillation_uses_teacher_uncertainty(config):
        required_fields = [field for field in required_fields if field != "teacher_uncertainty"]
        for field in ("teacher_q01", "teacher_q09"):
            if field not in required_fields:
                required_fields.append(field)
    cache_cfg["required_fields"] = required_fields
    if not layers:
        cache_cfg.setdefault("required_hidden_specs", {})
        return

    explicit_targets = dict(cache_cfg.get("required_hidden_targets", {}))
    existing = dict(cache_cfg.get("required_hidden_specs", {}))
    for layer in layers:
        if layer["name"] in explicit_targets:
            continue
        existing[layer["name"]] = layer
    if existing:
        cache_cfg["required_hidden_specs"] = existing


def prepare_suite_config(config: dict[str, Any], args: Namespace) -> list[dict[str, Any]]:
    from mlp_distillation_suite.hidden import validate_hidden_layers
    from distillation_common.teacher_backend import teacher_backend

    teacher_backend(config)
    apply_suite_overrides(config, args)
    normalize_distillation_config(config)
    validate_distillation_config(config)
    enforce_standardized_protocol(config)
    layers = validate_hidden_layers(config)
    configure_teacher_cache_requirements(config, layers)
    return layers
