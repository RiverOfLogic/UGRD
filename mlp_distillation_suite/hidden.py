from __future__ import annotations

from typing import Any

import torch

from distillation_common.chronos_teacher import describe_teacher_tokens, select_hidden_output
from distillation_common.student_mlp_common import student_history
from distillation_common.hidden_pca_projection import (
    build_hidden_projectors,
    project_hidden_for_layer,
    validate_fixed_pca_layer,
)
from utils.losses import masked_hidden_cosine_loss, masked_hidden_mse_loss
from utils.metrics import select_output, select_output_mask


def validate_hidden_layers(config: dict[str, Any]) -> list[dict[str, Any]]:
    hidden_cfg = config.get("hidden_distillation", {})
    if not bool(hidden_cfg.get("enabled", bool(hidden_cfg.get("layers", [])))):
        return []

    layers = hidden_cfg.get("layers", [])
    if not layers:
        return []

    names = set()
    validated = []
    for idx, layer_cfg in enumerate(layers):
        layer = dict(layer_cfg)
        layer.setdefault("name", f"hidden_{idx}")
        layer.setdefault("weight", 1.0)
        if "teacher_tokens" not in layer:
            layer.setdefault("teacher_token", "future")
        if layer["name"] in names:
            raise ValueError(f"Duplicate hidden layer mapping name: {layer['name']}")
        names.add(layer["name"])
        if "student_key" not in layer:
            raise ValueError(f"hidden_distillation.layers[{idx}] is missing student_key.")
        if "teacher_layer_index" not in layer:
            raise ValueError(f"hidden_distillation.layers[{idx}] is missing teacher_layer_index.")
        if "projection" not in layer:
            raise ValueError(f"hidden_distillation.layers[{idx}] is missing projection.")

        projection_method = layer["projection"].get("method", "linear")
        if "teacher_tokens" in layer:
            token_cfg = layer["teacher_tokens"]
            if not isinstance(token_cfg, dict) or token_cfg.get("mode") != "explicit":
                raise ValueError("teacher_tokens currently supports only mode: explicit.")
            tokens = token_cfg.get("tokens")
            if not isinstance(tokens, list) or not tokens:
                raise ValueError("teacher_tokens.tokens must be a non-empty list.")
            if projection_method != "memory_cross_attention":
                raise ValueError("teacher_tokens explicit multi-token distillation requires memory_cross_attention.")
            layer["teacher_tokens"] = {"mode": "explicit", "tokens": [str(token) for token in tokens]}
        elif layer["teacher_token"] not in {"reg", "future"}:
            raise ValueError("teacher_token must be 'reg' or 'future'.")

        validate_fixed_pca_layer(layer)
        validated.append(layer)
    return validated


def distillation_mode(layers: list[dict[str, Any]]) -> str:
    if not layers:
        return "soft_only"
    if len(layers) == 1:
        return "single_hidden"
    return "multi_hidden"


def build_projectors_for_layers(
    layers: list[dict[str, Any]],
    config: dict[str, Any],
) -> torch.nn.ModuleDict:
    if not layers:
        return torch.nn.ModuleDict()
    return build_hidden_projectors(layers, config)


def student_forward_selected_with_hiddens(
    model: torch.nn.Module,
    dataset: Any,
    seq_x: torch.Tensor,
    seq_y: torch.Tensor,
    channel_mask: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
    layers: list[dict[str, Any]],
    extra_student_keys: dict[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    seq_x = seq_x.to(device)
    seq_y = seq_y.to(device)
    channel_mask = channel_mask.to(device)
    seq_x = student_history(seq_x, config)
    prediction, hidden_states = model.forward_with_hidden(seq_x, channel_mask=channel_mask)
    label_len = int(config["data"]["label_len"])
    pred_len = int(config["model"]["pred_len"])
    target = seq_y[:, label_len : label_len + pred_len, :]
    prediction = select_output(prediction, dataset)
    target = select_output(target, dataset)
    selected_mask = select_output_mask(channel_mask, dataset)

    selected_hiddens: dict[str, torch.Tensor] = {}
    for layer in layers:
        student_key = layer["student_key"]
        if student_key not in hidden_states:
            available = ", ".join(sorted(hidden_states.keys()))
            raise ValueError(f"Student hidden key {student_key!r} not found. Available keys: {available}")
        selected_hiddens[layer["name"]] = select_hidden_output(hidden_states[student_key], dataset)
    for name, key in (extra_student_keys or {}).items():
        selected_hiddens[name] = select_hidden_output(hidden_states[key], dataset)
    return prediction, target, selected_mask, selected_hiddens


def hidden_loss_function(name: str):
    if name == "mse":
        return masked_hidden_mse_loss
    if name == "cosine":
        return masked_hidden_cosine_loss
    raise ValueError(f"hidden_loss must be 'mse' or 'cosine', got {name!r}.")


def compute_hidden_distillation_loss(
    projectors: torch.nn.ModuleDict,
    layers: list[dict[str, Any]],
    dataset_name: str,
    student_hiddens: dict[str, torch.Tensor],
    teacher_hiddens: dict[str, torch.Tensor],
    selected_mask: torch.Tensor,
    *,
    hidden_loss_name: str,
    log_fn=None,
    step: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not layers:
        device = selected_mask.device
        return torch.zeros((), device=device), {}

    loss_fn = hidden_loss_function(hidden_loss_name)
    hidden_loss = torch.zeros((), device=selected_mask.device)
    layer_loss_values: dict[str, torch.Tensor] = {}
    for layer in layers:
        name = layer["name"]
        if name not in teacher_hiddens:
            available = ", ".join(sorted(teacher_hiddens.keys()))
            raise ValueError(f"Teacher hidden {name!r} not found. Available keys: {available}")
        hidden_left, hidden_right = project_hidden_for_layer(
            projectors[name],
            layer,
            dataset_name,
            student_hiddens[name],
            teacher_hiddens[name],
        )
        layer_loss = loss_fn(hidden_left, hidden_right, selected_mask)
        hidden_loss = hidden_loss + float(layer.get("weight", 1.0)) * layer_loss
        layer_loss_values[name] = layer_loss
        if log_fn is not None and step == 1:
            log_fn(
                f"hidden {name} | student_key={layer['student_key']} | "
                f"teacher_block={layer['teacher_layer_index']} | tokens={describe_teacher_tokens(layer)} | "
                f"student={tuple(student_hiddens[name].shape)} | teacher={tuple(teacher_hiddens[name].shape)} | "
                f"aligned_left={tuple(hidden_left.shape)} | aligned_right={tuple(hidden_right.shape)}"
            )
    return hidden_loss, layer_loss_values
