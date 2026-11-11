"""Formal train-scaled shared caches, frozen REG means and GT increment scales."""
import math
import shutil
import time

import numpy as np
import torch
from scipy.fft import dct, idct

from .common import cache_root, file_hash, read_json, resolve, save_json, signature
from .data import raw_dataset


def prepare_cache(study, cell, device):
    from distillation_common.chronos_teacher import load_chronos2_teacher, teacher_predict_hiddens_and_uncertainty

    root = cache_root(study, cell)
    root.mkdir(parents=True, exist_ok=True)
    raw = raw_dataset(study, cell)
    multiple = math.lcm(*study["grid"]["batch_sizes"])
    count = min(study["train_windows"], len(raw))//multiple*multiple
    if count < multiple:
        raise ValueError("Training windows are insufficient for the configured batch grid")
    starts = raw.sample_indices[np.linspace(0, len(raw)-1, count).round().astype(int)]
    # Optional paired-context study: keep forecast origins identical to the longest history.
    origin_context = study.get("origin_context", study["teacher_context"])
    if origin_context < study["teacher_context"]:
        raise ValueError("origin_context must cover teacher_context")
    starts = starts + origin_context - study["teacher_context"]
    horizon, channels = cell["horizon"], raw.output_dim
    contract = {"version": 1, "dataset_sha256": file_hash(resolve(study["data_root"])/cell["data_path"]),
                "teacher_sha256": file_hash(resolve(study["teacher_path"])/"model.safetensors"),
                "teacher_config_sha256": file_hash(resolve(study["teacher_path"])/"config.json"),
                "teacher_context": study["teacher_context"], "horizon": horizon, "cell": cell,
                "windows": count, "starts_sha256": signature(starts.tolist()), "dtype": "float32",
                "normalization": "full original training split", "reg_layer": study.get('reg_layer', 11)}
    exports = study.get('quantile_exports', [])
    if "origin_context" in study:
        contract['origin_context'] = origin_context
    if study.get('keep_raw_reg', False):
        contract['keep_raw_reg'] = True
    if exports:
        contract['quantile_exports'] = exports
    identity = signature(contract)
    metadata_path = root/"metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata["signature"] != identity:
            raise ValueError(f"Cache protocol changed; select another cache_dir: {root}")
        if not all((root/key).exists() for key in metadata["hashes"]):
            raise FileNotFoundError(f"Completed cache missing files: {root}")
        return metadata
    progress_path = root/"teacher_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else None
    if progress and progress["signature"] != identity:
        raise ValueError("Incomplete cache belongs to another protocol")
    estimated = count*channels*(768+channels+(3+len(exports))*horizon)*4
    if not progress and shutil.disk_usage(root).free < estimated+study["min_free_disk_gib"]*2**30:
        raise RuntimeError(f"Cache needs about {estimated/2**30:.2f} GiB plus disk reserve at {root}")
    shapes = {"median": (count, horizon, channels), "lower": (count, horizon, channels),
              "upper": (count, horizon, channels), "reg_raw": (count, channels, 768)}
    arrays = {key: np.lib.format.open_memmap(root/f"{key}.npy", mode="r+" if progress else "w+",
                                           dtype="float32", shape=shape) for key, shape in shapes.items()}
    cfg = {"model": {"seq_len": study["model"]["seq_len"], "pred_len": horizon},
           "teacher": {"path": study["teacher_path"], "device_map": device, "fallback_to_cpu": False,
                       "context_length": study["teacher_context"], "prediction_length": horizon,
                       "prediction_mode": "multivariate", "max_series_per_forward": channels*study["teacher_micro_batch"],
                       "quantile_lower": cell["quantile_lower"], "quantile_upper": cell["quantile_upper"]}}
    torch.use_deterministic_algorithms(False)  # Frozen teacher preparation; students use strict mode.
    teacher = load_chronos2_teacher(cfg)
    teacher.model.float().eval().requires_grad_(False)
    export_arrays = {}
    export_indices = {}
    for level in exports:
        levels = list(teacher.model.chronos_config.quantiles)
        export_indices[level] = levels.index(level)
        key = 'quantile_' + str(level).replace('.', 'p')
        export_arrays[level] = np.lib.format.open_memmap(
            root/f'{key}.npy', mode='r+' if progress else 'w+',
            dtype='float32', shape=(count, horizon, channels))
    start_at = progress["completed"] if progress else 0
    elapsed = progress["seconds"] if progress else 0.
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    mb = study["teacher_micro_batch"]
    for start in range(start_at, count, mb):
        end = min(start+mb, count)
        x = torch.from_numpy(np.stack([raw.full_data[s:s+study["teacher_context"]] for s in starts[start:end]]))
        captured = []
        handle = None
        if exports:
            handle = teacher.model.register_forward_hook(
                lambda _model, _args, result: captured.append(result.quantile_preds.detach()))
        try:
            median, hidden, bounds = teacher_predict_hiddens_and_uncertainty(
                teacher, raw, x, cfg, [{"name": "reg", "teacher_layer_index": study.get('reg_layer', 11), "teacher_token": "reg"}], torch.device(device))
        finally:
            if handle is not None:
                handle.remove()
        if exports:
            quantiles = torch.cat(captured, dim=0)
            for level, array in export_arrays.items():
                values = quantiles[:, export_indices[level], :horizon]
                array[start:end] = values.reshape(end-start, channels, horizon).transpose(1, 2).cpu().numpy()
        a, b = bounds["teacher_q01"].cpu().numpy(), bounds["teacher_q09"].cpu().numpy()
        arrays["median"][start:end] = median.cpu().numpy()
        arrays["lower"][start:end], arrays["upper"][start:end] = np.minimum(a, b), np.maximum(a, b)
        arrays["reg_raw"][start:end] = hidden["reg"].cpu().numpy()
        if end%32 == 0 or end == count:
            for key in arrays:
                arrays[key].flush()
            for array in export_arrays.values():
                array.flush()
            save_json(progress_path, {"signature": identity, "completed": end, "seconds": elapsed+time.perf_counter()-started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(device)), "teacher_micro_batch": mb})
            print(f"CACHE {cell['dataset']} H{horizon} {end}/{count}", flush=True)
    del teacher
    torch.cuda.empty_cache()
    np.save(root/"starts.npy", starts)
    active = np.asarray(raw.scaler.var_) > 1e-16
    if active.sum()<2:
        raise ValueError("REG requires at least two nonconstant training channels")
    np.save(root/"active.npy", active)
    np.save(root/"scaler_mean.npy", raw.scaler.mean_)
    np.save(root/"scaler_scale.npy", raw.scaler.scale_)
    mean = np.zeros((channels, 768), dtype="float64")
    for start in range(0, count, 64):
        mean += arrays["reg_raw"][start:start+64].sum(0, dtype="float64")
    mean /= count
    np.save(root/"reg_mean.npy", mean.astype("float32"))
    grams = np.lib.format.open_memmap(root/"reg_gram.npy", mode="w+", dtype="float32", shape=(count, channels, channels))
    scale_sum = np.zeros(channels, dtype="float64")
    first, lag, keep = int(cell["first_position"]), int(cell["lag"]), int(cell["K"])
    for start in range(0, count, 32):
        end = min(start+32, count)
        z = np.asarray(arrays["reg_raw"][start:end], dtype="float64")-mean
        z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
        grams[start:end] = z @ z.transpose(0, 2, 1)
        y = np.stack([raw.full_data[s+study["teacher_context"]:s+study["teacher_context"]+horizon] for s in starts[start:end]])
        coefficients = dct(y.astype("float64"), norm="ortho", axis=1)
        coefficients[:, keep:] = 0
        smooth = idct(coefficients, norm="ortho", axis=1)
        scale_sum += np.square(smooth[:, first:]-smooth[:, first-lag:horizon-lag]).sum((0, 1))
    grams.flush()
    np.save(root/"increment_scales.npy", np.maximum(np.sqrt(scale_sum/(count*(horizon-first))), 1e-3).astype("float32"))
    edges = np.linspace(0, horizon, 5).round().astype(int)
    weights = np.repeat(cell["qbd_weights"], np.diff(edges)).astype("float32")
    weights /= weights.mean()
    np.save(root/"qbd_weights.npy", weights)
    files = ["starts", "active", "scaler_mean", "scaler_scale", "reg_mean", "reg_gram", "median", "lower", "upper", "increment_scales", "qbd_weights"]
    files += ['quantile_' + str(level).replace('.', 'p') for level in exports]
    if study.get('keep_raw_reg', False):
        files.append('reg_raw')
    metadata = {"signature": identity, "contract": contract, "channels": raw.feature_names, "windows": count,
                "full_train_rows": int(raw.sample_indices[-1])+study["teacher_context"]+horizon,
                "student_history": study["model"]["seq_len"], "teacher_history": study["teacher_context"],
                "hashes": {key+".npy": file_hash(root/(key+".npy")) for key in files}}
    save_json(metadata_path, metadata)
    # This is a newly generated temporary array, never a historical cache dependency.
    del arrays["reg_raw"]
    if not study["keep_raw_reg"]:
        (root/"reg_raw.npy").unlink()
    return metadata
