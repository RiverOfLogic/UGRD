"""The sole final loss: MSE + weighted QBD + latent REG + band increment Huber."""
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import read_json, resolve, save_json


def band_projector(horizon, keep, device):
    t = torch.arange(horizon, dtype=torch.float64)
    k = torch.arange(keep, dtype=torch.float64)[:, None]
    basis = torch.cos(math.pi/horizon*(t+.5)*k)*math.sqrt(2/horizon)
    basis[0] /= math.sqrt(2)
    basis = basis.to(device=device, dtype=torch.float32)
    return basis.T @ basis


def gram(hidden):
    unit = F.normalize(hidden, dim=-1, eps=1e-8)
    return unit @ unit.transpose(-1, -2)


def components(prediction, hidden, truth, median, lower, upper, teacher_gram,
               mask, step_weights, projector, scales, first, lag):
    active = mask.bool()
    valid = active[:, None, :].expand_as(prediction)
    mse = (prediction-truth).square()[valid].mean()
    lo, hi = torch.minimum(lower, upper), torch.maximum(lower, upper)
    q = (prediction-median).abs()+.5*((lo-prediction).clamp_min(0).square()+(prediction-hi).clamp_min(0).square())
    qbd = (q*step_weights[None, :, None])[valid].mean()
    relation = gram(hidden)
    edge = active[:, :, None]&active[:, None, :]
    edge &= ~torch.eye(active.shape[-1], dtype=torch.bool, device=prediction.device)
    reg = (relation-teacher_gram).square()[edge].mean()
    error = torch.einsum("btc,st->bsc", prediction-median, projector)
    delta = (error[:, first:]-error[:, first-lag:error.shape[1]-lag])/scales
    values = F.huber_loss(delta, torch.zeros_like(delta), reduction="none", delta=1.)
    inc = values[active[:, None, :].expand_as(values)].mean()
    return {"mse": mse, "qbd": qbd, "reg": reg, "increment": inc}


class FinalObjective(nn.Module):
    student_keys = {"last": "last"}
    replaces_base_loss = True

    def __init__(self, config, device):
        super().__init__()
        cfg = config["final_method"]
        root = resolve(cfg["cache_dir"])
        metadata = read_json(root/"metadata.json")
        if metadata["signature"] != cfg["cache_signature"]:
            raise ValueError("Cache identity changed after resolving training config")
        cell = cfg["cell"]
        self.first, self.lag = int(cell["first_position"]), int(cell["lag"])
        self.weights = cfg["weights"]
        self.register_buffer("step_weights", torch.tensor(np.load(root/"qbd_weights.npy"), device=device))
        self.register_buffer("scales", torch.tensor(np.load(root/"increment_scales.npy"), device=device))
        self.register_buffer("projector", band_projector(config["model"]["pred_len"], int(cell["K"]), device))
        self.heartbeat_path = cfg.get("heartbeat_path")
        self.heartbeat_seconds = cfg.get("heartbeat_seconds", 2.)
        self.last_heartbeat = 0.

    def heartbeat(self):
        if self.heartbeat_path and time.monotonic()-self.last_heartbeat >= self.heartbeat_seconds:
            device = self.scales.device
            save_json(self.heartbeat_path, {"reserved_bytes": torch.cuda.memory_reserved(device) if device.type=="cuda" else 0,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
                      "time": time.time()})
            self.last_heartbeat = time.monotonic()

    def on_model_ready(self, model):
        self.heartbeat()

    def forward(self, prediction, mask, batch, median, bounds, *, student_hiddens):
        self.heartbeat()
        pieces = components(prediction, student_hiddens["last"], batch["seq_y"].to(prediction.device),
                            median, bounds["teacher_q01"], bounds["teacher_q09"],
                            batch["final_reg_gram"].to(prediction.device), mask,
                            self.step_weights, self.projector, self.scales, self.first, self.lag)
        loss = pieces["mse"]+sum(self.weights[key]*pieces[key] for key in ("qbd", "reg", "increment"))
        return loss, {"final_"+key: value.detach() for key, value in pieces.items()}
