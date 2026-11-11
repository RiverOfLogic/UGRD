import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distillation_common.chronos_teacher import (  # noqa: E402
    describe_teacher_tokens,
)
from distillation_common.teacher_backend import (load_teacher, teacher_structure,
    teacher_predict_hiddens_and_uncertainty, teacher_fingerprint_options)
from distillation_common.student_mlp_common import base_args, dataset_spec, resolve_path  # noqa: E402
from research_data_loader.factory import build_dataset, forecast_collate_fn  # noqa: E402


def cache_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("teacher", {}).get("cache", {}))


def cache_enabled(config: dict[str, Any]) -> bool:
    return bool(cache_config(config).get("enabled", False))


def teacher_cache_dir(config: dict[str, Any]) -> Path:
    return resolve_path(cache_config(config).get("dir", "outputs/teacher_cache/chronos2"))


def numpy_cache_dtype(config: dict[str, Any]) -> np.dtype:
    name = str(cache_config(config).get("dtype", "float32")).lower()
    if name in {"float16", "fp16", "half"}:
        return np.dtype("float16")
    if name in {"float32", "fp32"}:
        return np.dtype("float32")
    raise ValueError(f"Unsupported teacher cache dtype: {name}")


def cache_format(config: dict[str, Any]) -> str:
    fmt = str(cache_config(config).get("format", "memmap")).lower()
    if fmt == "memmap":
        return fmt
    raise ValueError(f"Unsupported teacher cache format: {fmt}. Only 'memmap' is supported.")


def _canonical_hidden_request(request: dict[str, Any]) -> dict[str, Any]:
    canonical = {
        "teacher_layer_index": int(request.get("teacher_layer_index", request.get("layer_index", 11))),
    }
    if "teacher_tokens" in request:
        token_cfg = request["teacher_tokens"]
        canonical["teacher_tokens"] = {
            "mode": token_cfg.get("mode", "explicit"),
            "tokens": [str(token) for token in token_cfg.get("tokens", [])],
        }
    else:
        canonical["teacher_token"] = request.get("teacher_token", request.get("token", "future"))
    return canonical


def hidden_request_signature(request: dict[str, Any]) -> str:
    encoded = json.dumps(_canonical_hidden_request(request), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def teacher_base_fingerprint(config: dict[str, Any], dataset_name: str) -> str:
    spec = dataset_spec(config, dataset_name)
    payload = {
        "model": {
            "seq_len": config["model"]["seq_len"],
            "pred_len": config["model"]["pred_len"],
        },
        "data": {
            "scale": config["data"].get("scale", True),
            "embed": config["data"].get("embed", "timeF"),
            "label_len": config["data"].get("label_len", 0),
        },
        "dataset": {
            "name": dataset_name,
            "data": spec.get("data"),
            "data_path": spec.get("data_path"),
            "features": spec.get("features"),
            "target": spec.get("target"),
            "freq": spec.get("freq"),
            "split": spec.get("split"),
            "root_path": spec.get("root_path"),
        },
        "teacher": {
            "path": config["teacher"].get("path"),
            "prediction_length": config["teacher"].get("prediction_length"),
            "context_length": config["teacher"].get("context_length"),
            "prediction_mode": config["teacher"].get("prediction_mode", "multivariate"),
        },
    }
    payload["teacher"].update(teacher_fingerprint_options(config["teacher"]))
    q_lower = float(config["teacher"].get("quantile_lower", 0.1))
    q_upper = float(config["teacher"].get("quantile_upper", 0.9))
    if (q_lower, q_upper) != (0.1, 0.9):
        payload["teacher"]["quantile_lower"] = q_lower
        payload["teacher"]["quantile_upper"] = q_upper
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def teacher_request_fingerprint(config: dict[str, Any], hidden_requests: list[dict[str, Any]]) -> str:
    payload = {
        "base": {
            "model": {
                "seq_len": config["model"]["seq_len"],
                "pred_len": config["model"]["pred_len"],
            },
            "data": {
                "scale": config["data"].get("scale", True),
                "embed": config["data"].get("embed", "timeF"),
                "label_len": config["data"].get("label_len", 0),
            },
            "teacher": {
                "path": config["teacher"].get("path"),
                "prediction_length": config["teacher"].get("prediction_length"),
                "context_length": config["teacher"].get("context_length"),
                "prediction_mode": config["teacher"].get("prediction_mode", "multivariate"),
            },
        },
        "hidden_requests": [_canonical_hidden_request(request) for request in hidden_requests],
    }
    payload["base"]["teacher"].update(teacher_fingerprint_options(config["teacher"]))
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_train_dataset_and_loader(
    config: dict[str, Any],
    dataset_name: str,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[Any, DataLoader]:
    args = base_args(config)
    spec = dataset_spec(config, dataset_name)
    for key, value in spec.items():
        setattr(args, key, value)
    if batch_size is not None:
        args.batch_size = int(batch_size)
    if num_workers is not None:
        args.num_workers = int(num_workers)
    dataset = build_dataset(spec, args, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=forecast_collate_fn,
    )
    return dataset, loader


def _safe_field_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)


def _to_numpy(tensor: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if dtype is not None and np.issubdtype(array.dtype, np.floating):
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _remove_cache_payload_files(output_dir: Path) -> None:
    for pattern in ("shard_*.pt", "*.npy"):
        for old_file in output_dir.glob(pattern):
            old_file.unlink()


def _write_memmap_cache_for_dataset(
    config: dict[str, Any],
    dataset_name: str,
    hidden_requests: list[dict[str, Any]],
    *,
    max_batches: int | None,
    shard_size: int,
    output_dir: Path,
    base_fingerprint: str,
    request_fingerprint: str,
    log_fn=print,
) -> Path:
    cache_cfg = cache_config(config)
    request_cfg = config.get("requests", {})
    save_inputs = bool(request_cfg.get("save_inputs", True))
    save_teacher_prediction = bool(request_cfg.get("save_teacher_prediction", True))
    save_teacher_quantiles = bool(
        request_cfg.get(
            "save_teacher_quantiles",
            request_cfg.get("save_teacher_uncertainty", save_teacher_prediction),
        )
    )
    dtype = numpy_cache_dtype(config)

    dataset, loader = deterministic_train_dataset_and_loader(config, dataset_name)
    teacher = load_teacher(config)
    input_patch_size, output_patch_size, num_blocks = teacher_structure(teacher, config)
    device = torch.device("cpu")

    total_samples = len(dataset)
    if max_batches is not None:
        total_samples = min(total_samples, int(max_batches) * int(loader.batch_size or 1))

    memmaps: dict[str, np.memmap] = {}
    quantile_files = {
        "teacher_q01": "teacher_q01.npy",
        "teacher_q09": "teacher_q09.npy",
    }
    hidden_files = {request["name"]: f"teacher_hidden_{_safe_field_name(request['name'])}.npy" for request in hidden_requests}
    hidden_shapes: dict[str, list[int]] = {}
    sample_count = 0
    started = time.time()

    def ensure_memmaps(seq_x: torch.Tensor, seq_y: torch.Tensor, channel_mask: torch.Tensor,
                       teacher_prediction: torch.Tensor, teacher_hiddens: dict[str, torch.Tensor],
                       teacher_quantiles: dict[str, torch.Tensor]) -> None:
        if memmaps:
            return
        memmaps["sample_start_indices"] = np.lib.format.open_memmap(
            output_dir / "sample_start_indices.npy",
            mode="w+",
            dtype=np.int64,
            shape=(total_samples,),
        )
        memmaps["channel_mask"] = np.lib.format.open_memmap(
            output_dir / "channel_mask.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_samples, int(channel_mask.shape[1])),
        )
        if save_inputs:
            memmaps["seq_x"] = np.lib.format.open_memmap(
                output_dir / "seq_x.npy",
                mode="w+",
                dtype=dtype,
                shape=(total_samples, *tuple(seq_x.shape[1:])),
            )
            memmaps["seq_y"] = np.lib.format.open_memmap(
                output_dir / "seq_y.npy",
                mode="w+",
                dtype=dtype,
                shape=(total_samples, *tuple(seq_y.shape[1:])),
            )
        if save_teacher_prediction:
            memmaps["teacher_prediction"] = np.lib.format.open_memmap(
                output_dir / "teacher_prediction.npy",
                mode="w+",
                dtype=dtype,
                shape=(total_samples, *tuple(teacher_prediction.shape[1:])),
            )
        if save_teacher_quantiles:
            for quantile_name, tensor in teacher_quantiles.items():
                memmaps[quantile_name] = np.lib.format.open_memmap(
                    output_dir / quantile_files[quantile_name],
                    mode="w+",
                    dtype=dtype,
                    shape=(total_samples, *tuple(tensor.shape[1:])),
                )
        for name, value in teacher_hiddens.items():
            hidden_shapes[name] = [total_samples, *list(value.shape[1:])]
            memmaps[f"teacher_hidden:{name}"] = np.lib.format.open_memmap(
                output_dir / hidden_files[name],
                mode="w+",
                dtype=dtype,
                shape=tuple(hidden_shapes[name]),
            )

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > int(max_batches):
            break
        seq_x, seq_y, _, _, channel_mask = batch
        teacher_prediction, teacher_hiddens, teacher_quantiles = teacher_predict_hiddens_and_uncertainty(
            teacher,
            dataset,
            seq_x,
            config,
            hidden_requests,
            device,
        )
        ensure_memmaps(seq_x, seq_y, channel_mask, teacher_prediction, teacher_hiddens, teacher_quantiles)
        batch_size = int(seq_x.shape[0])
        end = min(sample_count + batch_size, total_samples)
        count = end - sample_count
        if count <= 0:
            break
        source_slice = slice(0, count)
        target_slice = slice(sample_count, end)
        memmaps["sample_start_indices"][target_slice] = np.asarray(
            dataset.sample_indices[sample_count:end],
            dtype=np.int64,
        )
        memmaps["channel_mask"][target_slice] = _to_numpy(channel_mask[source_slice], np.dtype("float32"))
        if save_inputs:
            memmaps["seq_x"][target_slice] = _to_numpy(seq_x[source_slice], dtype)
            memmaps["seq_y"][target_slice] = _to_numpy(seq_y[source_slice], dtype)
        if save_teacher_prediction:
            memmaps["teacher_prediction"][target_slice] = _to_numpy(teacher_prediction[source_slice], dtype)
        if save_teacher_quantiles:
            for quantile_name, value in teacher_quantiles.items():
                memmaps[quantile_name][target_slice] = _to_numpy(value[source_slice], dtype)
        for name, value in teacher_hiddens.items():
            memmaps[f"teacher_hidden:{name}"][target_slice] = _to_numpy(value[source_slice], dtype)

        sample_count = end
        if sample_count % max(1, shard_size) == 0 or batch_index == 1:
            log_fn(f"cache {dataset_name} | wrote memmap rows={sample_count}/{total_samples}")

    for value in memmaps.values():
        value.flush()

    available_fields = ["sample_start_indices", "channel_mask"]
    if save_inputs:
        available_fields.extend(["seq_x", "seq_y"])
    if save_teacher_prediction:
        available_fields.append("teacher_prediction")
    if save_teacher_quantiles:
        available_fields.extend(["teacher_q01", "teacher_q09"])
    if hidden_requests:
        available_fields.append("teacher_hiddens")

    metadata = {
        "dataset": dataset_name,
        "split": "train",
        "format": "memmap",
        "available_fields": available_fields,
        "features": getattr(dataset, "features", None),
        "target": getattr(dataset, "target", None),
        "seq_len": int(getattr(dataset, "seq_len", config["model"]["seq_len"])),
        "student_seq_len": int(config["model"]["seq_len"]),
        "data_context_length": int(getattr(dataset, "seq_len", config["model"]["seq_len"])),
        "pred_len": int(config["model"]["pred_len"]),
        "label_len": int(config["data"]["label_len"]),
        "teacher_path": config["teacher"].get("path"),
        "teacher_backend": config["teacher"].get("backend", "chronos2"),
        "teacher_prediction_mode": config["teacher"].get("prediction_mode", "multivariate"),
        "teacher_context_length": config["teacher"].get("context_length"),
        "teacher_prediction_length": config["teacher"].get("prediction_length"),
        "input_patch_size": input_patch_size,
        "output_patch_size": output_patch_size,
        "num_encoder_blocks": num_blocks,
        "hidden_requests": hidden_requests,
        "hidden_targets": {
            request["name"]: {
                "teacher_layer_index": int(request.get("teacher_layer_index", request.get("layer_index", 11))),
                "teacher_token": request.get("teacher_token"),
                "teacher_tokens": request.get("teacher_tokens"),
            }
            for request in hidden_requests
        },
        "hidden_request_signatures": {
            request["name"]: hidden_request_signature(request)
            for request in hidden_requests
        },
        "hidden_request_summary": {
            request["name"]: describe_teacher_tokens(request)
            for request in hidden_requests
        },
        "files": {
            "sample_start_indices": "sample_start_indices.npy",
            "channel_mask": "channel_mask.npy",
            **({"seq_x": "seq_x.npy", "seq_y": "seq_y.npy"} if save_inputs else {}),
            **({"teacher_prediction": "teacher_prediction.npy"} if save_teacher_prediction else {}),
            **({name: filename for name, filename in quantile_files.items()} if save_teacher_quantiles else {}),
        },
        "hidden_files": hidden_files,
        "hidden_shapes": hidden_shapes,
        "dtype": str(dtype),
        "shard_size": shard_size,
        "sample_count": sample_count,
        "channel_names": list(getattr(dataset, "feature_names", [])),
        "output_slice": _slice_to_json(dataset.output_slice),
        "base_config_fingerprint": base_fingerprint,
        "config_fingerprint": request_fingerprint,
        "cache_config": {
            "save_inputs": save_inputs,
            "save_teacher_prediction": save_teacher_prediction,
            "save_teacher_quantiles": save_teacher_quantiles,
            "format": "memmap",
            "dtype": str(dtype),
        },
        "dataset_spec": dataset_spec(config, dataset_name),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - started, 4),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def build_teacher_cache_for_dataset(
    config: dict[str, Any],
    dataset_name: str,
    hidden_requests: list[dict[str, Any]],
    *,
    max_batches: int | None = None,
    shard_size: int | None = None,
    output_root: Path | None = None,
    log_fn=print,
) -> Path:
    cache_cfg = cache_config(config)
    cache_format(config)
    shard_size = int(shard_size or cache_cfg.get("shard_size", 256))
    output_root = output_root or teacher_cache_dir(config)
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    base_fingerprint = teacher_base_fingerprint(config, dataset_name)
    request_fingerprint = teacher_request_fingerprint(config, hidden_requests)
    required_signatures = {hidden_request_signature(request) for request in hidden_requests}
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not bool(cache_cfg.get("rebuild", False)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_format = metadata.get("format")
        if existing_format != "memmap":
            raise ValueError(
                f"Existing teacher cache at {output_dir} uses format={existing_format}, "
                "but only format=memmap is supported. Use teacher.cache.rebuild=true "
                "or a different cache path."
            )
        requested_fields = {"channel_mask", "sample_start_indices"}
        request_cfg = config.get("requests", {})
        if bool(request_cfg.get("save_inputs", True)):
            requested_fields.update({"seq_x", "seq_y"})
        if bool(request_cfg.get("save_teacher_prediction", True)):
            requested_fields.add("teacher_prediction")
        if bool(
            request_cfg.get(
                "save_teacher_quantiles",
                request_cfg.get("save_teacher_uncertainty", request_cfg.get("save_teacher_prediction", True)),
            )
        ):
            requested_fields.update({"teacher_q01", "teacher_q09"})
        available_fields = set(metadata.get("available_fields", []))
        missing_fields = sorted(requested_fields - available_fields)
        if missing_fields:
            raise ValueError(
                f"Existing teacher cache at {output_dir} is missing requested fields {missing_fields}. "
                "Use teacher.cache.rebuild=true or a different cache path."
            )
        actual_base = metadata.get("base_config_fingerprint") or metadata.get("config_fingerprint")
        existing_signatures = set((metadata.get("hidden_request_signatures") or {}).values())
        if actual_base != base_fingerprint:
            raise ValueError(
                f"Existing teacher cache at {output_dir} has a different base config fingerprint. "
                "Use teacher.cache.rebuild=true or a different teacher.cache.dir."
            )
        missing = sorted(required_signatures - existing_signatures)
        if missing:
            raise ValueError(
                f"Existing teacher cache at {output_dir} is missing required hidden targets. "
                "Use --rebuild with a cache profile that includes all required hidden requests, "
                "or use a different teacher.cache.dir."
            )
        log_fn(f"cache {dataset_name} | existing cache is valid; skip rebuild")
        return output_dir
    if bool(cache_cfg.get("rebuild", False)):
        _remove_cache_payload_files(output_dir)
    return _write_memmap_cache_for_dataset(
        config,
        dataset_name,
        hidden_requests,
        max_batches=max_batches,
        shard_size=shard_size,
        output_dir=output_dir,
        base_fingerprint=base_fingerprint,
        request_fingerprint=request_fingerprint,
        log_fn=log_fn,
    )


def _slice_to_json(value: slice) -> dict[str, int | None]:
    return {"start": value.start, "stop": value.stop, "step": value.step}


def cache_cli_overrides(config: dict[str, Any], args: Any) -> None:
    teacher_cfg = config.setdefault("teacher", {})
    cache_cfg = teacher_cfg.setdefault("cache", {})
    if getattr(args, "teacher_cache_enabled", False):
        cache_cfg["enabled"] = True
    if getattr(args, "teacher_cache_path", None) is not None:
        cache_cfg["path"] = str(args.teacher_cache_path)
    if getattr(args, "teacher_cache_dir", None) is not None:
        cache_cfg["dir"] = str(args.teacher_cache_dir)
    if getattr(args, "teacher_cache_name", None) is not None:
        cache_cfg["name"] = str(args.teacher_cache_name)
    if getattr(args, "teacher_cache_dtype", None) is not None:
        cache_cfg["dtype"] = args.teacher_cache_dtype
    if getattr(args, "teacher_cache_format", None) is not None:
        cache_cfg["format"] = args.teacher_cache_format
    if getattr(args, "shard_size", None) is not None:
        cache_cfg["shard_size"] = int(args.shard_size)
    if getattr(args, "rebuild", False):
        cache_cfg["rebuild"] = True
