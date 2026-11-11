import math
import sys
import warnings
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHRONOS_SRC = PROJECT_ROOT / "chronos-forecasting" / "src"
for import_path in (PROJECT_ROOT, CHRONOS_SRC):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from chronos import Chronos2Pipeline  # noqa: E402
from distillation_common.student_mlp_common import resolve_path  # noqa: E402
from models import HiddenStateProjector  # noqa: E402
from utils.metrics import select_output  # noqa: E402


def load_chronos2_teacher(config: dict[str, Any]) -> Chronos2Pipeline:
    teacher_cfg = config["teacher"]
    configured_path = Path(str(teacher_cfg["path"]))
    model_path = resolve_path(configured_path)
    if configured_path.is_absolute() or any(sep in str(teacher_cfg["path"]) for sep in ("\\", "/")):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Chronos2 teacher path does not exist: {model_path}. "
                "Download the teacher weights or update teacher.path in the distillation config."
            )
        pretrained_path = model_path.as_posix()
    else:
        pretrained_path = str(teacher_cfg["path"])
    device_map = teacher_cfg.get("device_map", "cuda")
    fallback_to_cpu = bool(teacher_cfg.get("fallback_to_cpu", True))
    if device_map == "cuda" and not torch.cuda.is_available():
        if not fallback_to_cpu:
            raise RuntimeError("Config requested CUDA teacher, but CUDA is unavailable.")
        device_map = "cpu"
    try:
        teacher = Chronos2Pipeline.from_pretrained(pretrained_path, device_map=device_map)
    except RuntimeError:
        if device_map == "cpu" or not fallback_to_cpu:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        teacher = Chronos2Pipeline.from_pretrained(pretrained_path, device_map="cpu")
    teacher.model.eval()
    return teacher


def select_hidden_output(hidden: torch.Tensor, dataset: Any) -> torch.Tensor:
    return hidden[:, dataset.output_slice, ...]


def single_hidden_request_from_config(config: dict[str, Any]) -> dict[str, Any]:
    teacher_cfg = config["teacher"]
    request = {
        "name": "hidden",
        "teacher_layer_index": int(teacher_cfg.get("layer_index", 11)),
        "teacher_token": teacher_cfg.get("token", "reg"),
    }
    if "teacher_tokens" in teacher_cfg:
        request["teacher_tokens"] = teacher_cfg["teacher_tokens"]
    return request


def describe_teacher_tokens(request: dict[str, Any]) -> str:
    if "teacher_tokens" in request:
        tokens = request["teacher_tokens"].get("tokens", [])
        return ",".join(str(token) for token in tokens)
    return str(request.get("teacher_token", request.get("token", "future")))


def _teacher_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _quantile_index(quantile_levels: list[float], target: float) -> int:
    for index, level in enumerate(quantile_levels):
        if abs(float(level) - float(target)) < 1e-6:
            return index
    raise ValueError(f"Chronos2 teacher quantiles must include {target}, found {quantile_levels}.")


def _sample_chunk_size(batch_size: int, num_channels: int, teacher_cfg: dict[str, Any]) -> int:
    max_series = teacher_cfg.get("max_series_per_forward")
    if max_series is None:
        micro_batch_size = teacher_cfg.get("micro_batch_size")
        if micro_batch_size is not None:
            max_series = int(micro_batch_size) * num_channels
        else:
            max_series = batch_size * num_channels
    max_series = max(1, int(max_series))
    if num_channels > max_series:
        warnings.warn(
            "teacher.max_series_per_forward is smaller than one sample's channel count; "
            "processing one full sample per Chronos2 forward to preserve group attention.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1
    return max(1, min(batch_size, max_series // num_channels))


def _normalize_teacher_request(request: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(request)
    normalized.setdefault("name", f"layer_{normalized.get('teacher_layer_index', normalized.get('layer_index', 0))}")
    if "teacher_layer_index" not in normalized:
        normalized["teacher_layer_index"] = normalized.get("layer_index", 11)
    normalized["teacher_layer_index"] = int(normalized["teacher_layer_index"])
    if "teacher_tokens" in normalized:
        token_cfg = normalized["teacher_tokens"]
        if not isinstance(token_cfg, dict) or token_cfg.get("mode") != "explicit":
            raise ValueError("teacher_tokens currently supports only mode: explicit.")
        tokens = token_cfg.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("teacher_tokens.tokens must be a non-empty list.")
        normalized["teacher_tokens"] = {"mode": "explicit", "tokens": [str(token) for token in tokens]}
    else:
        token = normalized.get("teacher_token", normalized.get("token", "future"))
        if token not in {"reg", "future"}:
            raise ValueError("teacher_token must be 'reg' or 'future' when teacher_tokens is not provided.")
        normalized["teacher_token"] = token
    return normalized


def _future_token_index(token: str, num_output_patches: int, hidden_len: int) -> int:
    try:
        future_offset = int(token.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid future token label {token!r}; expected 'fut_i'.") from exc
    if future_offset < 0 or future_offset >= num_output_patches:
        raise ValueError(
            f"Future token {token!r} is out of range for {num_output_patches} output patch tokens."
        )
    return hidden_len - num_output_patches + future_offset


def _gather_teacher_tokens(
    hidden: torch.Tensor,
    request: dict[str, Any],
    *,
    reg_index: int,
    num_output_patches: int,
) -> torch.Tensor:
    hidden_len = hidden.shape[1]
    if reg_index >= hidden_len:
        raise ValueError(f"Computed REG token index {reg_index} but hidden sequence length is {hidden_len}.")
    if num_output_patches > hidden_len:
        raise ValueError(
            f"Computed {num_output_patches} future tokens but hidden sequence length is {hidden_len}."
        )

    if "teacher_tokens" in request:
        indices = []
        for token in request["teacher_tokens"]["tokens"]:
            if token == "reg":
                indices.append(reg_index)
            elif token.startswith("fut_"):
                indices.append(_future_token_index(token, num_output_patches, hidden_len))
            elif token.startswith("hist_"):
                try:
                    history_index = int(token.split("_", 1)[1])
                except ValueError as exc:
                    raise ValueError(f"Invalid historical token {token!r}") from exc
                if not 0 <= history_index < reg_index:
                    raise ValueError(f"Historical token {token!r} outside [0,{reg_index})")
                indices.append(history_index)
            else:
                raise ValueError(f"Unsupported explicit teacher token {token!r}; use 'reg', 'fut_i', or 'hist_i'.")
        index_tensor = torch.tensor(indices, device=hidden.device, dtype=torch.long)
        return hidden.index_select(1, index_tensor)

    token = request["teacher_token"]
    if token == "reg":
        return hidden[:, reg_index, :]

    future_start = hidden_len - num_output_patches
    index_tensor = torch.arange(future_start, hidden_len, device=hidden.device)
    return hidden.index_select(1, index_tensor)


@torch.no_grad()
def teacher_predict_hiddens_and_uncertainty(
    teacher: Chronos2Pipeline,
    dataset: Any,
    seq_x: torch.Tensor,
    config: dict[str, Any],
    hidden_requests: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    teacher_cfg = config["teacher"]
    context_len = int(teacher_cfg.get("context_length") or config["model"]["seq_len"])
    if seq_x.shape[1] < context_len:
        raise ValueError(
            f"seq_x history length {seq_x.shape[1]} is shorter than teacher.context_length={context_len}."
        )
    if seq_x.shape[1] > context_len:
        seq_x = seq_x[:, -context_len:, :]
    requests = [_normalize_teacher_request(request) for request in hidden_requests]
    request_names = [request["name"] for request in requests]
    if len(request_names) != len(set(request_names)):
        raise ValueError(f"Duplicate teacher hidden request names: {request_names}")

    teacher_model = teacher.model
    teacher_model.eval()
    teacher_device = _teacher_device(teacher_model)
    blocks = teacher_model.encoder.block
    unique_layer_indices = sorted({int(request["teacher_layer_index"]) for request in requests})
    for layer_index in unique_layer_indices:
        if layer_index < 0 or layer_index >= len(blocks):
            raise ValueError(f"teacher_layer_index={layer_index} is out of range for {len(blocks)} encoder blocks.")

    batch_size, seq_len, num_channels = seq_x.shape
    pred_len = int(teacher_cfg.get("prediction_length") or config["model"]["pred_len"])
    output_patch_size = int(teacher_model.chronos_config.output_patch_size)
    input_patch_size = int(teacher_model.chronos_config.input_patch_size)
    num_output_patches = math.ceil(pred_len / output_patch_size)
    reg_index = math.ceil(seq_len / input_patch_size)
    sample_chunk_size = _sample_chunk_size(batch_size, num_channels, teacher_cfg)

    requests_by_layer: dict[int, list[dict[str, Any]]] = {layer_index: [] for layer_index in unique_layer_indices}
    for request in requests:
        requests_by_layer[int(request["teacher_layer_index"])].append(request)

    quantile_parts: list[torch.Tensor] = []
    captured_parts: dict[str, list[torch.Tensor]] = {request["name"]: [] for request in requests}

    for mb_start in range(0, batch_size, sample_chunk_size):
        mb_end = min(mb_start + sample_chunk_size, batch_size)
        mb = mb_end - mb_start
        mb_seq_x = seq_x[mb_start:mb_end]
        context = mb_seq_x.permute(0, 2, 1).reshape(mb * num_channels, seq_len)
        context = context.to(device=teacher_device, dtype=torch.float32)
        group_ids = torch.arange(mb, device=teacher_device).repeat_interleave(num_channels)

        captured: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(layer_index: int):
            def capture_hidden(_module, _inputs, output):
                hidden = output.hidden_states
                for request in requests_by_layer[layer_index]:
                    captured[request["name"]] = _gather_teacher_tokens(
                        hidden,
                        request,
                        reg_index=reg_index,
                        num_output_patches=num_output_patches,
                    ).detach()

            return capture_hidden

        for layer_index in unique_layer_indices:
            handles.append(blocks[layer_index].register_forward_hook(make_hook(layer_index)))
        try:
            output = teacher_model(
                context=context,
                group_ids=group_ids,
                num_output_patches=num_output_patches,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = [request["name"] for request in requests if request["name"] not in captured]
        if missing:
            raise RuntimeError(f"Failed to capture Chronos2 hidden states: {missing}")

        quantile_parts.append(output.quantile_preds[:, :, :pred_len])
        for request in requests:
            captured_parts[request["name"]].append(captured[request["name"]])

    quantiles = torch.cat(quantile_parts, dim=0)
    quantile_levels = [float(level) for level in teacher_model.chronos_config.quantiles]
    lower_level = float(teacher_cfg.get("quantile_lower", 0.1))
    upper_level = float(teacher_cfg.get("quantile_upper", 0.9))
    median_index = _quantile_index(quantile_levels, 0.5)
    lower_index = _quantile_index(quantile_levels, lower_level)
    upper_index = _quantile_index(quantile_levels, upper_level)
    teacher_full = quantiles[:, median_index, :].reshape(batch_size, num_channels, pred_len).transpose(1, 2)
    teacher_prediction = select_output(teacher_full.to(device=device, dtype=seq_x.dtype), dataset)
    lower_full = quantiles[:, lower_index, :].reshape(batch_size, num_channels, pred_len).transpose(1, 2)
    upper_full = quantiles[:, upper_index, :].reshape(batch_size, num_channels, pred_len).transpose(1, 2)
    teacher_q01 = select_output(lower_full.to(device=device, dtype=seq_x.dtype), dataset)
    teacher_q09 = select_output(upper_full.to(device=device, dtype=seq_x.dtype), dataset)

    teacher_hiddens = {}
    for request in requests:
        hidden = torch.cat(captured_parts[request["name"]], dim=0)
        if hidden.ndim == 2:
            hidden_full = hidden.reshape(batch_size, num_channels, hidden.shape[-1])
        else:
            hidden_full = hidden.reshape(batch_size, num_channels, hidden.shape[-2], hidden.shape[-1])
        teacher_hiddens[request["name"]] = select_hidden_output(
            hidden_full.to(device=device, dtype=seq_x.dtype),
            dataset,
        )
    teacher_quantiles = {
        "teacher_q01": teacher_q01,
        "teacher_q09": teacher_q09,
    }
    return teacher_prediction, teacher_hiddens, teacher_quantiles


@torch.no_grad()
def teacher_predict_and_hiddens(
    teacher: Chronos2Pipeline,
    dataset: Any,
    seq_x: torch.Tensor,
    config: dict[str, Any],
    hidden_requests: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_prediction, teacher_hiddens, _ = teacher_predict_hiddens_and_uncertainty(
        teacher,
        dataset,
        seq_x,
        config,
        hidden_requests,
        device,
    )
    return teacher_prediction, teacher_hiddens


# NOTE: project_hidden_for_distillation moved to distillation_common/hidden_pca_projection.py
# (it's teacher-agnostic -- operates only on projector/tensor shapes -- and keeping it here
# forced that module to import chronos_teacher, which pulls in the Chronos stack even for
# teacher-agnostic callers like the Toto distillation suite).
