"""Shared calibration math; formal training recomputes coefficients for each batch."""
import numpy as np
import torch

from distillation_common.student_mlp_common import build_model, set_seed
from .common import ROOT, calibration_path, config_for, file_hash, read_json, save_json, signature
from .data import CachedTraining, collate_final_batch
from .objective import FinalObjective, components


def flat_gradient(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gradients, parameters)]).detach().cpu()


def measure(model, batches, cell, step_weights, projector, scales, goals, device):
    parameters = tuple(model.parameters())
    stored = []
    model.eval()
    for batch in batches:
        x = batch["seq_x"].to(device)
        mask = batch["channel_mask"].to(device)
        prediction, hidden = model.forward_with_hidden(x, channel_mask=mask)
        pieces = components(prediction, hidden["last"], batch["seq_y"].to(device),
                            batch["teacher_prediction"].to(device), batch["teacher_q01"].to(device),
                            batch["teacher_q09"].to(device), batch["final_reg_gram"].to(device),
                            mask, step_weights, projector, scales, int(cell["first_position"]), int(cell["lag"]))
        stored.append([flat_gradient(pieces[key], parameters) for key in ("mse", "qbd", "reg", "increment")])
    norms = np.array([[float(g.double().norm()) for g in group] for group in stored])
    if not np.isfinite(norms).all() or (norms<=0).any():
        raise ValueError("Nonfinite or zero calibration gradient")
    m = np.median(norms, axis=0)
    q = goals["qbd_to_mse"]*m[0]/m[1]
    r = goals["reg_to_mse"]*m[0]/m[2]
    bases = [gm+q*gq+r*gr for gm, gq, gr, gi in stored]
    base = float(np.median([float(g.double().norm()) for g in bases]))
    inc = goals["increment_to_base"]*base/m[3]
    return {"weights": {"qbd": float(q), "reg": float(r), "increment": float(inc) if cell["increment_enabled"] else 0.},
            "increment_calibrated_before_screen": float(inc),
            "median_gradient_norms": dict(zip(("mse", "qbd", "reg", "increment"), map(float, m))),
            "base_gradient_median": base, "gradient_norms_per_batch": norms.tolist(),
            "target_ratios": goals, "initial_eval_only": True}


def calibrate(study, cell, batch, device):
    destination = calibration_path(study, cell, batch)
    cfg = config_for(study, cell, device, batch, study["grid"]["learning_rates"][0],
                     study["calibration"]["seed"], destination.parent/"calibration_runtime",
                     weights={"qbd": 1., "reg": .1, "increment": .1 if cell["increment_enabled"] else 0.})
    contract = {"model": cfg["model"], "cache_signature": cfg["final_method"]["cache_signature"],
                "batch": batch, "policy": study["calibration"], "objective_version": 1,
                "source_sha256": {name: file_hash(ROOT/"extensions/final_method"/name)
                                  for name in ("calibration.py", "objective.py")}}
    identity = signature(contract)
    if destination.exists():
        result = read_json(destination)
        if result["signature"] != identity:
            raise ValueError("Calibration recipe changed; select another cache directory")
        return result
    dataset = CachedTraining(cfg)
    count = min(len(dataset), study["calibration"]["windows"])
    if count<batch:
        raise ValueError("Calibration window count must be at least the batch size")
    indices = np.linspace(0, len(dataset)-1, count).round().astype(int)
    batches = (collate_final_batch([dataset[int(i)] for i in indices[start:start+batch]])
               for start in range(0, count-count%batch, batch))
    set_seed(study["calibration"]["seed"])
    model = build_model(cfg).to(device)
    objective = FinalObjective(cfg, torch.device(device))
    objective.heartbeat_path = None
    goals = {key: study["calibration"][key] for key in ("qbd_to_mse", "reg_to_mse", "increment_to_base")}
    result = measure(model, batches, cell, objective.step_weights, objective.projector, objective.scales, goals, torch.device(device))
    result.update(signature=identity, contract=contract, windows=count-count%batch, batch=batch,
                  normalization="full original training split", seed=study["calibration"]["seed"])
    save_json(destination, result)
    return result
