"""Portable preparation, memory-profiled grid search, confirmation and locked test."""
import argparse
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import yaml

from .common import (ROOT, case_key, cells, file_hash, load_study, read_json, run_root,
                     save_json, signature)
from .scheduler import schedule


def batch_key(cell, batch, window):
    return f"{case_key(cell)}_w{window}_b{batch}"


def learning_tag(rate):
    return format(rate, ".8g").replace(".", "p").replace("-", "m")


def stamp(study, *, extra_sources=()):
    root = run_root(study)
    root.mkdir(parents=True, exist_ok=True)
    recipe = {key: value for key, value in study.items() if key not in ("resources", "run_name", "output_dir")}
    dependency_names = ["models/student_lmpnet.py", "models/dsd_lmpnet.py", "mlp_distillation_suite/trainer.py",
                        "distillation_common/student_mlp_common.py", "distillation_common/chronos_teacher.py",
                        "research_data_loader/datasets.py", "research_data_loader/factory.py", "utils/tools.py"]
    files = (list((ROOT/"extensions/final_method").glob("*.py"))
             + [ROOT/p for p in dependency_names] + list(extra_sources))
    payload = {"recipe": recipe, "parameter_table_sha256": file_hash(ROOT/study["parameter_table"]),
               "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in files}}
    path = root/"protocol.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("Run recipe/source changed; use another run_name")
    save_json(path, payload)
    (root/"study.yaml").write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")
    for file in files:
        target = root/"source_snapshot"/file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    return signature(payload)


def create_job(study, recipe, action, identifier, cell, output, **extra):
    payload = {"id": identifier, "action": action, "cell": cell, "output": str(output),
               "study_file": str(run_root(study)/"study.yaml"), **extra}
    identity = {"recipe": recipe, "action": action, "id": identifier,
                "selected_checkpoint": extra.get("selected", {}).get("checkpoint_sha256")}
    if action in ("test", "test_intervals"):
        evaluation = study.get("evaluation", {})
        source = "worker.py" if evaluation.get("mode") == "overall" else "evaluation.py"
        identity.update(evaluation=evaluation,
                        evaluation_source=file_hash(ROOT/"extensions/final_method"/source))
    payload["job_signature"] = signature(identity)
    job_file = run_root(study)/"job_specs"/(identifier+".json")
    save_json(job_file, payload)
    resources = study["resources"]
    budget = resources["profile_budget_gib"]
    if action in ("prepare", "test_intervals"):
        budget = resources["prepare_budget_gib"]
    elif action == "calibrate":
        budget = resources["calibration_budget_gib"]
    elif action in ("train", "test"):
        batch = extra.get("batch", extra.get("selected", {}).get("batch"))
        path = run_root(study)/"profiles"/batch_key(cell, batch, study["model"]["moving_avg"])/"result.json"
        if path.exists():
            budget = read_json(path)["peak_reserved_bytes"]/2**30*resources["memory_safety_factor"]+resources["context_allowance_gib"]
    return dict(payload, job_file=job_file, output_path=output, budget_bytes=int(budget*2**30))


def stage_jobs(study, recipe, stage):
    root = run_root(study)
    result = []
    for cell in cells(study):
        key = case_key(cell)
        if stage == "prepare":
            result.append(create_job(study, recipe, "prepare", "prepare_"+key, cell, root/"preparation"/key))
        elif stage in ("profile", "calibrate"):
            for batch in study["grid"]["batch_sizes"]:
                ident = batch_key(cell, batch, study["model"]["moving_avg"])
                directory = "profiles" if stage == "profile" else "calibrations"
                result.append(create_job(study, recipe, stage, stage+"_"+ident, cell, root/directory/ident, batch=batch))
        elif stage == "grid":
            for batch, rate in itertools.product(study["grid"]["batch_sizes"], study["grid"]["learning_rates"]):
                ident = f"{key}_b{batch}_lr{learning_tag(rate)}_s{study['grid']['search_seed']}"
                result.append(create_job(study, recipe, "train", "grid_"+ident, cell, root/"grid"/ident,
                                         batch=batch, learning_rate=rate, seed=study["grid"]["search_seed"]))
    return result


def select_grid(study, recipe):
    chosen = []
    all_jobs = stage_jobs(study, recipe, "grid")
    for cell in cells(study):
        jobs = [j for j in all_jobs if case_key(j["cell"])==case_key(cell)]
        rows = [read_json(j["output_path"]/"result.json") for j in jobs]
        best = min(rows, key=lambda r: (r["validation"]["mse"], r["validation"]["mae"], r["learning_rate"], r["batch"]))
        chosen.append({"cell": cell, "selected": best})
    path = run_root(study)/"grid_selection.json"
    if path.exists() and read_json(path) != chosen:
        raise ValueError("Validation selection changed after being frozen")
    save_json(path, chosen)
    return chosen


def confirmation_jobs(study, recipe):
    chosen = read_json(run_root(study)/"grid_selection.json")
    jobs = []
    for item in chosen:
        cell, selected = item["cell"], item["selected"]
        for seed in study["grid"]["confirmation_seeds"]:
            if seed == study["grid"]["search_seed"]:
                continue  # Reuse exactly the selected grid model for seed2026.
            identifier = f"confirm_{case_key(cell)}_s{seed}"
            jobs.append(create_job(study, recipe, "train", identifier, cell, run_root(study)/"confirmation"/identifier,
                                   batch=selected["batch"], learning_rate=selected["learning_rate"], seed=seed))
    return jobs


def lock_models(study, recipe):
    confirmations = {j["id"]: j for j in confirmation_jobs(study, recipe)}
    rows = []
    for item in read_json(run_root(study)/"grid_selection.json"):
        cell, selected = item["cell"], item["selected"]
        for seed in study["grid"]["confirmation_seeds"]:
            row = selected if seed == study["grid"]["search_seed"] else read_json(
                confirmations[f"confirm_{case_key(cell)}_s{seed}"]["output_path"]/"result.json")
            if file_hash(row["checkpoint"]) != row["checkpoint_sha256"] or file_hash(row["resolved_config"]) != row["config_sha256"]:
                raise ValueError("Model/config changed before test lock")
            cfg = yaml.safe_load(Path(row["resolved_config"]).read_text())
            frame = pd.read_csv(Path(row["resolved_config"]).parent/"epoch_metrics.csv", float_precision="round_trip")
            best, best_epoch, stale, stop = float("inf"), None, 0, None
            for epoch, mse in enumerate(frame.val_mse, 1):
                if mse < best:
                    best, best_epoch, stale = mse, epoch, 0
                else:
                    stale += 1
                if stale >= cfg["train"]["early_stopping_patience"]:
                    stop = epoch;break
            if len(frame)!=(stop or cfg["train"]["epochs"]) or best_epoch!=row["best_epoch"] or not cfg["train"]["validation_only"]:
                raise ValueError("Early stopping / validation-only audit failed")
            rows.append({"cell": cell, "selected": row})
    path = run_root(study)/"validation_lock.json"
    if path.exists() and read_json(path)!=rows:
        raise ValueError("Frozen validation lock changed")
    save_json(path, rows)
    pd.DataFrame([dict(dataset=r['cell']['dataset'], horizon=r['cell']['horizon'],
                       seed=r['selected']['seed'], batch=r['selected']['batch'],
                       learning_rate=r['selected']['learning_rate'],
                       best_epoch=r['selected']['best_epoch'],
                       checkpoint=r['selected']['checkpoint'],
                       checkpoint_sha256=r['selected']['checkpoint_sha256'],
                       resolved_config=r['selected']['resolved_config']) for r in rows]).to_csv(
                           run_root(study)/'final_models.csv', index=False)
    return rows


def test_jobs(study, recipe):
    locked = lock_models(study, recipe)
    return [create_job(study, recipe, "test", f"test_fbo_{case_key(r['cell'])}_s{r['selected']['seed']}", r["cell"],
                       run_root(study)/"test_fbo"/f"{case_key(r['cell'])}_s{r['selected']['seed']}", selected=r["selected"])
            for r in locked]


def report(study, jobs):
    rows = []
    for job in jobs:
        result = read_json(job["output_path"]/"result.json")
        selected = result["selected"]
        rows.append({"dataset": job["cell"]["dataset"], "horizon": job["cell"]["horizon"],
                     "seed": selected["seed"], "learning_rate": selected["learning_rate"], "batch": selected["batch"],
                     "best_epoch": selected["best_epoch"], **result["test"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(run_root(study)/"test_results_fbo.csv", index=False)
    metrics = ["mae", "mse", "fbo_mae", "fbo_mse"]
    summary = frame.groupby(["dataset", "horizon"])[metrics].agg(["mean", "std", "count"])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(run_root(study)/"summary_fbo.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="extensions/final_method/configs/train.yaml")
    parser.add_argument("--stage", choices=["plan", "prepare", "profile", "calibrate", "grid", "select", "confirm", "test", "all"], default="plan")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", help="Visible CUDA indices, e.g. 0,1; respects CUDA_VISIBLE_DEVICES")
    parser.add_argument("--run-name")
    parser.add_argument("--cache-dir")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--moving-avg", type=int, choices=[25, 65])
    parser.add_argument("--fbo-reference", help="Validation-locked no-QBD run directory")
    args = parser.parse_args()
    study = load_study(args.config)
    if args.fbo_reference:
        study['evaluation'] = {'reference_run': args.fbo_reference}
    if args.stage in ('test', 'all') and not args.dry_run:
        from .evaluation import reference_root
        baseline_root = reference_root(study)
        if not (baseline_root/'validation_lock.json').is_file():
            raise ValueError('Lock the no-QBD reference run before requesting test/all')
    for field in ("run_name", "cache_dir", "datasets", "horizons"):
        if getattr(args, field) is not None:
            study[field] = getattr(args, field)
    if args.gpus:
        study["resources"]["gpus"] = list(map(int, args.gpus.split(",")))
    if args.moving_avg:
        study["model"]["moving_avg"] = args.moving_avg
    if study["calibration"]["windows"] < max(study["grid"]["batch_sizes"]):
        raise ValueError("Calibration windows must cover at least one full largest batch")
    if (run_root(study)/"coordinator.lock").exists():
        raise RuntimeError("Coordinator lock exists; inspect the active coordinator before continuing")
    recipe = stamp(study)
    if args.stage == "plan" or args.dry_run:
        jobs = [j for stage in ("prepare", "profile", "calibrate", "grid") for j in stage_jobs(study, recipe, stage)]
        save_json(run_root(study)/"plan.json", [{k: v for k, v in j.items() if k not in ("output_path", "job_file")} for j in jobs])
        print(f"PLAN {len(cells(study))} cells; grid={len(stage_jobs(study,recipe,'grid'))}; confirmation adds seeds excluding reused search seed; no training launched")
        return
    # One coordinator per run; Ctrl-C cleanup releases the lock, hard crashes leave it for explicit inspection.
    lock = run_root(study)/"coordinator.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            import os
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(f"Coordinator lock exists: {lock}; inspect running processes before removing stale lock")
    try:
        for stage in ("prepare", "profile", "calibrate", "grid", "select", "confirm", "test"):
            if args.stage not in (stage, "all"):
                continue
            if stage == "select":
                select_grid(study, recipe);continue
            if stage in ("grid", "confirm", "test"):
                profiles = stage_jobs(study, recipe, "profile")
                if not all((j["output_path"]/"result.json").exists() for j in profiles):
                    raise RuntimeError("Run profile stage first; training requires measured GPU budgets")
            jobs = confirmation_jobs(study, recipe) if stage=="confirm" else test_jobs(study, recipe) if stage=="test" else stage_jobs(study, recipe, stage)
            if stage == 'test':
                interval_jobs = [create_job(study, recipe, 'test_intervals', 'fbo_masks_'+case_key(c),
                                            c, run_root(study)/'fbo_mask_jobs'/case_key(c)) for c in cells(study)]
                schedule(interval_jobs, study['resources'], stage='test_intervals')
            schedule(jobs, study["resources"], stage=stage)
            if stage == "test":
                report(study, jobs)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
