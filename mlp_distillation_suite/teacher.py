from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHRONOS_SRC = PROJECT_ROOT / "chronos-forecasting" / "src"
for import_path in (PROJECT_ROOT, CHRONOS_SRC):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from distillation_common.teacher_backend import load_teacher as load_backend_teacher, teacher_predict_hiddens_and_uncertainty


def load_teacher(config: dict[str, Any]) -> Any:
    return load_backend_teacher(config)


@torch.no_grad()
def teacher_predict_for_suite(
    teacher: Any,
    dataset: Any,
    seq_x: torch.Tensor,
    config: dict[str, Any],
    layers: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return teacher_predict_hiddens_and_uncertainty(teacher, dataset, seq_x, config, layers, device)
