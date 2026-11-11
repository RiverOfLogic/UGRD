"""Teacher dispatch shared by online distillation and offline cache building."""
from typing import Any


def teacher_backend(config: dict[str, Any]) -> str:
    name = str(config.get("teacher", {}).get("backend", "chronos2")).lower()
    if name not in {"chronos2", "timestral"}:
        raise ValueError(f"Unsupported teacher.backend: {name!r}")
    return name


def load_teacher(config):
    if teacher_backend(config) == "timestral":
        from extensions.timestral.teacher import load_timestral_teacher
        return load_timestral_teacher(config)
    from distillation_common.chronos_teacher import load_chronos2_teacher
    return load_chronos2_teacher(config)


def teacher_predict_hiddens_and_uncertainty(teacher, dataset, seq_x, config, requests, device):
    if teacher_backend(config) == "timestral":
        from extensions.timestral.teacher import teacher_predict_hiddens_and_uncertainty as predict
    else:
        from distillation_common.chronos_teacher import teacher_predict_hiddens_and_uncertainty as predict
    return predict(teacher, dataset, seq_x, config, requests, device)


def teacher_structure(teacher, config):
    if teacher_backend(config) == "timestral":
        patch = int(teacher.config.patch_size)
        return patch, patch, len(teacher.model.transformer.layers)
    model = teacher.model
    return (int(model.chronos_config.input_patch_size),
            int(model.chronos_config.output_patch_size), len(model.encoder.block))


def teacher_fingerprint_options(teacher_config):
    """Preserve old Chronos fingerprints while isolating Timestral decode settings."""
    backend = teacher_backend({"teacher": teacher_config})
    if backend == "chronos2":
        return {}
    return {
        "backend": backend,
        "revision": teacher_config.get("revision"),
        "symmetric_averaging": bool(teacher_config.get("symmetric_averaging", True)),
        "clip_nonnegative": bool(teacher_config.get("clip_nonnegative", True)),
        "per_shot_horizon_patches": int(teacher_config.get("per_shot_horizon_patches", 32)),
        "quantile_real_cap_k": float(teacher_config.get("quantile_real_cap_k", 1e4)),
    }
