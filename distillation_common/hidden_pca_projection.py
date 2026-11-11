import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distillation_common.student_mlp_common import resolve_path  # noqa: E402
from models import HiddenProjectionConfig, HiddenStateProjector  # noqa: E402


def project_hidden_for_distillation(
    projector: HiddenStateProjector,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if projector.config.method == "memory_cross_attention":
        if teacher_hidden.ndim != 4:
            raise ValueError(
                "memory_cross_attention requires teacher hidden shape "
                f"[batch, channels, tokens, dim], got {tuple(teacher_hidden.shape)}."
            )
        projected_student_hidden = projector(student_hidden, num_future_tokens=teacher_hidden.shape[-2])
        return projected_student_hidden, teacher_hidden

    if teacher_hidden.ndim != 3:
        raise ValueError(
            f"Projection method {projector.config.method!r} requires single-token teacher hidden shape "
            f"[batch, channels, dim], got {tuple(teacher_hidden.shape)}."
        )
    projected_teacher_hidden = projector(teacher_hidden)
    return student_hidden, projected_teacher_hidden


def safe_module_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def fixed_pca_hidden_name(config: dict[str, Any], layer: dict[str, Any]) -> str:
    projection_cfg = layer["projection"]
    pca_cfg = dict(projection_cfg.get("pca", {}))
    if pca_cfg.get("hidden_name"):
        return str(pca_cfg["hidden_name"])
    mapped_targets = config.get("teacher", {}).get("cache", {}).get("required_hidden_targets", {})
    return str(mapped_targets.get(layer["name"], layer["name"]))


def fixed_pca_root(config: dict[str, Any], projection_cfg: dict[str, Any]) -> Path:
    pca_cfg = dict(projection_cfg.get("pca", {}))
    if pca_cfg.get("root"):
        return resolve_path(pca_cfg["root"])
    cache_name = config.get("teacher", {}).get("cache", {}).get("name") or "custom"
    return resolve_path(Path("outputs") / "hidden_svd" / cache_name)


def fixed_pca_path_for_dataset(
    config: dict[str, Any],
    layer: dict[str, Any],
    dataset_name: str,
) -> Path:
    projection_cfg = layer["projection"]
    pca_cfg = dict(projection_cfg.get("pca", {}))
    root = fixed_pca_root(config, projection_cfg)
    scope = str(pca_cfg.get("scope", "dataset"))
    hidden_name = fixed_pca_hidden_name(config, layer)
    if scope == "dataset":
        path = root / "dataset" / dataset_name / hidden_name / "pca.npz"
        fallback = root / "dataset" / dataset_name / layer["name"] / "pca.npz"
        return fallback if not path.exists() and fallback.exists() else path
    if scope == "global":
        path = root / "global" / "all" / hidden_name / "pca.npz"
        fallback = root / "global" / "all" / layer["name"] / "pca.npz"
        return fallback if not path.exists() and fallback.exists() else path
    raise ValueError(f"Unsupported fixed_pca scope {scope!r}; expected 'dataset' or 'global'.")


def validate_fixed_pca_layer(layer: dict[str, Any]) -> None:
    if layer["projection"].get("method") != "fixed_pca":
        return
    if "teacher_tokens" in layer:
        raise ValueError("projection.method='fixed_pca' does not support multi-token teacher_tokens.")
    if layer.get("teacher_token") == "future":
        raise ValueError(
            "projection.method='fixed_pca' requires a single-vector teacher token (e.g. 'reg' or "
            "'fut_i'); 'future' yields a token sequence and requires memory_cross_attention instead."
        )


class FixedPCATeacherProjector(torch.nn.Module):
    """Project teacher hidden states through fixed PCA, then a trainable linear map."""

    def __init__(
        self,
        *,
        layer: dict[str, Any],
        config: dict[str, Any],
        dataset_names: list[str],
    ) -> None:
        super().__init__()
        projection_cfg = layer["projection"]
        pca_cfg = dict(projection_cfg.get("pca", {}))
        self.rank = int(pca_cfg.get("rank") or projection_cfg.get("student_dim"))
        self.student_dim = int(projection_cfg["student_dim"])
        self.teacher_dim = int(projection_cfg["teacher_dim"])
        self.apply_layer_norm = bool(pca_cfg.get("layer_norm", True))
        self.scope = str(pca_cfg.get("scope", "dataset"))
        if self.rank <= 0:
            raise ValueError("fixed_pca rank must be positive.")
        self.pca_to_student = torch.nn.Linear(self.rank, self.student_dim)
        if self.rank == self.student_dim:
            torch.nn.init.eye_(self.pca_to_student.weight)
            torch.nn.init.zeros_(self.pca_to_student.bias)

        keys = ["all"] if self.scope == "global" else dataset_names
        self.dataset_to_key = {dataset: ("all" if self.scope == "global" else dataset) for dataset in dataset_names}
        self.key_to_buffer_name: dict[str, str] = {}
        for key in keys:
            dataset_for_path = dataset_names[0] if key == "all" else key
            pca_path = fixed_pca_path_for_dataset(config, layer, dataset_for_path)
            if not pca_path.exists():
                raise FileNotFoundError(f"fixed_pca file not found: {pca_path}")
            pca = np.load(pca_path)
            mean = np.asarray(pca["mean"], dtype=np.float32)
            components = np.asarray(pca["components"], dtype=np.float32)
            if mean.ndim != 1 or mean.shape[0] != self.teacher_dim:
                raise ValueError(
                    f"fixed_pca mean shape mismatch for {pca_path}: "
                    f"expected [{self.teacher_dim}], got {mean.shape}."
                )
            if components.ndim != 2 or components.shape[0] != self.teacher_dim:
                raise ValueError(
                    f"fixed_pca components shape mismatch for {pca_path}: "
                    f"expected [{self.teacher_dim}, >=rank], got {components.shape}."
                )
            if components.shape[1] < self.rank:
                raise ValueError(
                    f"fixed_pca rank={self.rank} exceeds stored components={components.shape[1]} at {pca_path}."
                )
            buffer_key = safe_module_key(key)
            self.key_to_buffer_name[key] = buffer_key
            self.register_buffer(f"{buffer_key}_mean", torch.from_numpy(mean), persistent=True)
            self.register_buffer(
                f"{buffer_key}_components",
                torch.from_numpy(components[:, : self.rank].copy()),
                persistent=True,
            )

    def forward(self, teacher_hidden: torch.Tensor, *, dataset_name: str) -> torch.Tensor:
        if teacher_hidden.ndim != 3:
            raise ValueError(
                "fixed_pca requires single-token teacher hidden shape [batch, channels, dim], "
                f"got {tuple(teacher_hidden.shape)}."
            )
        if teacher_hidden.shape[-1] != self.teacher_dim:
            raise ValueError(f"Expected teacher_dim={self.teacher_dim}, got {teacher_hidden.shape[-1]}.")
        key = self.dataset_to_key.get(dataset_name)
        if key is None:
            raise ValueError(f"No fixed_pca projection loaded for dataset {dataset_name!r}.")
        buffer_key = self.key_to_buffer_name[key]
        mean = getattr(self, f"{buffer_key}_mean").to(device=teacher_hidden.device, dtype=teacher_hidden.dtype)
        components = getattr(self, f"{buffer_key}_components").to(
            device=teacher_hidden.device,
            dtype=teacher_hidden.dtype,
        )
        pca_hidden = (teacher_hidden - mean) @ components
        return self.pca_to_student(pca_hidden)


def build_hidden_projectors(layers: list[dict[str, Any]], config: dict[str, Any]) -> torch.nn.ModuleDict:
    projectors = torch.nn.ModuleDict()
    dataset_names = [str(name) for name in config["data"]["enabled_datasets"]]
    for layer in layers:
        projection = dict(layer["projection"])
        method = projection.get("method", "linear")
        if method == "fixed_pca":
            projectors[layer["name"]] = FixedPCATeacherProjector(
                layer=layer,
                config=config,
                dataset_names=dataset_names,
            )
        else:
            projectors[layer["name"]] = HiddenStateProjector(HiddenProjectionConfig(**projection))
    return projectors


def project_hidden_for_layer(
    projector: torch.nn.Module,
    layer: dict[str, Any],
    dataset_name: str,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer["projection"].get("method", "linear") != "fixed_pca":
        return project_hidden_for_distillation(projector, student_hidden, teacher_hidden)
    if not isinstance(projector, FixedPCATeacherProjector):
        raise TypeError(f"Expected FixedPCATeacherProjector for layer {layer['name']}.")
    projected_teacher = projector(teacher_hidden, dataset_name=dataset_name)
    if student_hidden.ndim != 3:
        raise ValueError(
            "fixed_pca aligns teacher PCA coordinates to student hidden shape [batch, channels, dim], "
            f"got student shape {tuple(student_hidden.shape)}."
        )
    if student_hidden.shape[-1] != projected_teacher.shape[-1]:
        raise ValueError(
            f"Student hidden dim {student_hidden.shape[-1]} must match fixed_pca linear output dim "
            f"{projected_teacher.shape[-1]}."
        )
    if projector.apply_layer_norm:
        student_hidden = F.layer_norm(student_hidden, (student_hidden.shape[-1],))
        projected_teacher = F.layer_norm(projected_teacher, (projected_teacher.shape[-1],))
    return student_hidden, projected_teacher
