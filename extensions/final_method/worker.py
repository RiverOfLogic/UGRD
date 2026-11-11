"""One isolated cache/profile/calibration/training/evaluation process."""
import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch
import yaml

from distillation_common.student_mlp_common import (
    aggregate_metrics, build_eval_providers, build_model, set_seed,
)
from mlp_distillation_suite.trainer import forward_eval, run as shared_run, train_one_epoch
from .calibration import calibrate
from .common import (
    config_for, file_hash, load_study, read_json, save_json, setup, signature,
)
from .data import train_provider
from .objective import FinalObjective
from .prepare import prepare_cache


def objective_class(study):
    if "comparison" in study:
        from extensions.paper_experiments.objectives import ComparisonObjective
        return ComparisonObjective
    return FinalObjective


def profile(study, job, device):
    cell, batch = job["cell"], job["batch"]
    cfg = config_for(study, cell, str(device), batch, study["grid"]["learning_rates"][0],
                     study["grid"]["search_seed"], job["output"],
                     weights={"qbd": 1., "reg": .1, "increment": .1 if cell["increment_enabled"] else 0.})
    set_seed(cfg["train"]["seed"])
    model = build_model(cfg).to(device)
    objective = objective_class(study)(cfg, device)
    provider = train_provider(cfg, [cell["dataset"]])
    projectors = torch.nn.ModuleDict().to(device)
    if hasattr(objective, 'configure_projectors'):
        objective.configure_projectors(projectors)
    optimizer = torch.optim.AdamW(list(model.parameters())+list(projectors.parameters()),
                                  lr=cfg["train"]["learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"], foreach=False)
    train_one_epoch(model, projectors, None, provider, optimizer, device, cfg, [],
                    max_batches=2, epoch=1, extra_objective=objective)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = train_one_epoch(model, projectors, None, provider, optimizer, device, cfg, [],
                              max_batches=4, epoch=2, extra_objective=objective)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter()-started
    for value in build_eval_providers(cfg, [cell["dataset"]], "val"):
        cfg["train"]["max_eval_batches"] = 1
        forward_eval(value, model, device, cfg)
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "batch": batch,
            "warmup_batches": 2, "measured_batches": metrics["train_batches"],
            "windows_per_second": metrics["train_batches"]*batch/seconds,
            "seconds": seconds, "precision": "float32", "effective_global_batch": batch,
            "world_size": 1, "mode": "independent single-process jobs"}


def train(study, job, device):
    directory = Path(job["output"])
    if (directory/"checkpoints/best.pt").exists():
        raise FileExistsError(f"Incomplete job kept intact: {directory}; use a new run name or archive it explicitly")
    cfg = config_for(study, job["cell"], str(device), job["batch"], job["learning_rate"],
                     job["seed"], directory)
    path = directory/"config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    args = argparse.Namespace(epochs=None, max_train_batches=None, max_eval_batches=None,
                              datasets=None, run_seed=job["seed"], seed=job["seed"], itr=1, itr_index=0)
    shared_run(path, args, objective_factory=objective_class(study), train_provider_factory=train_provider)
    checkpoint = directory/"checkpoints/best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    frame = pd.read_csv(directory/"epoch_metrics.csv", float_precision="round_trip")
    if abs(frame.val_mse.min()-saved["metrics"]["mse"])>1e-12:
        raise ValueError("Best checkpoint and validation log disagree")
    return {"validation": saved["metrics"], "best_epoch": saved["epoch"], "trained_epochs": len(frame),
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
            "resolved_config": str(directory/"resolved_config.yaml"),
            "config_sha256": file_hash(directory/"resolved_config.yaml"),
            "cache_signature": cfg["final_method"]["cache_signature"],
            "weights": cfg["final_method"]["weights"], "batch": job["batch"], "learning_rate": job["learning_rate"],
            "seed": job["seed"], "precision": "float32", "world_size": 1}


def test(study, job, device):
    selected = job["selected"]
    if file_hash(selected["checkpoint"]) != selected["checkpoint_sha256"] or file_hash(selected["resolved_config"]) != selected["config_sha256"]:
        raise ValueError("Validation-locked checkpoint/config changed")
    cfg = yaml.safe_load(Path(selected["resolved_config"]).read_text(encoding="utf-8"))
    cfg["train"]["device"] = str(device)
    cfg["evaluation"] = study.get("evaluation", {})
    model = build_model(cfg).to(device)
    saved = torch.load(selected["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    provider, = build_eval_providers(cfg, [job["cell"]["dataset"]], "test")
    if cfg["evaluation"].get("mode") == "overall":
        with torch.no_grad():
            metrics = forward_eval(provider, model, device, cfg)
    else:
        from .evaluation import evaluate_four

        metrics, step_metrics = evaluate_four(provider, model, device, cfg)
        pd.DataFrame(step_metrics).to_csv(Path(job['output'])/'metrics_by_step.csv', index=False)
    # Export only the student and global training scaler; no teacher/objective is deployed.
    cache = Path(cfg["final_method"]["cache_dir"])
    import numpy as np
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "model_config": cfg["model"], "scaler_mean": torch.tensor(np.load(cache/"scaler_mean.npy")),
                "scaler_scale": torch.tensor(np.load(cache/"scaler_scale.npy")),
                "dataset": job["cell"]["dataset"], "horizon": job["cell"]["horizon"],
                "seed": selected["seed"]}, Path(job["output"])/"student.pt")
    return {"test": metrics, "selected": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    study = load_study(job["study_file"])
    directory = Path(job["output"])
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/"result.json"
    if result_path.exists():
        if read_json(result_path)["job_signature"] != job["job_signature"]:
            raise ValueError("Job output belongs to a different recipe")
        return
    device = setup(args.device, study["train"]["cpu_threads"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    save_json(directory/"heartbeat.json", {"reserved_bytes": 0, "time": time.time()})
    action = job["action"]
    if action == "prepare":
        result = {"cache": prepare_cache(study, job["cell"], str(device))["signature"]}
    elif action == "profile":
        result = profile(study, job, device)
    elif action == "calibrate":
        result = calibrate(study, job["cell"], job["batch"], str(device))
    elif action == "test_intervals":
        from .evaluation import prepare_test_mask

        result = {'fbo_masks': prepare_test_mask(study, job['cell'])}
    elif action == "train":
        result = train(study, job, device)
    elif action == "test":
        result = test(study, job, device)
    else:
        raise ValueError(action)
    torch.cuda.synchronize(device)
    result.update(job_signature=job["job_signature"], seconds_total=time.perf_counter()-started,
                  device=str(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    save_json(result_path, result)
    print("DONE", job["id"], flush=True)


if __name__ == "__main__":
    main()
