"""Run the full Chronos2 → LMP-Net method with five fixed seeds."""

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from extensions.final_method.common import case_key, cells, load_study, read_json, run_root, save_json
from extensions.final_method.run import (
    confirmation_jobs, create_job, lock_models, select_grid, stage_jobs, stamp,
)
from extensions.final_method.scheduler import schedule


def cell_studies(study, root):
    for cell in cells(study):
        key = case_key(cell)
        params = study["fixed_cells"][key]
        single = deepcopy(study)
        single.pop("fixed_cells")
        single.update(
            datasets=[cell["dataset"]], horizons=[cell["horizon"]],
            run_name=key, output_dir=str(root),
            fixed_weights=params["weights"],
        )
        single["model"]["moving_avg"] = params["moving_avg"]
        single["grid"].update(batch_sizes=[params["batch"]], learning_rates=[params["learning_rate"]])
        yield cell, single


def execute(study, studies, root):
    recipes = [
        (cell, single, stamp(single, extra_sources=[Path(__file__).resolve()]))
        for cell, single in studies
    ]
    resources = study["resources"]
    for stage in ("prepare", "profile", "grid"):
        jobs = [job for _, single, recipe in recipes for job in stage_jobs(single, recipe, stage)]
        schedule(jobs, resources, stage=stage)

    confirmations = []
    for _, single, recipe in recipes:
        select_grid(single, recipe)
        confirmations.extend(confirmation_jobs(single, recipe))
    schedule(confirmations, resources, stage="confirm")

    # Audit every validation checkpoint before any worker opens the test split.
    locked = [(single, recipe, lock_models(single, recipe)) for _, single, recipe in recipes]
    save_json(root/"validation_lock.json", [row for _, _, rows in locked for row in rows])
    tests = []
    for single, recipe, rows in locked:
        for row in rows:
            cell, selected = row["cell"], row["selected"]
            key = f"{case_key(cell)}_s{selected['seed']}"
            tests.append(create_job(
                single, recipe, "test", f"test_{key}", cell, run_root(single)/"test"/key,
                selected=selected,
            ))
    schedule(tests, resources, stage="test")

    results = []
    for job in tests:
        result = read_json(job["output_path"]/"result.json")
        cell, selected = job["cell"], result["selected"]
        results.append(dict(
            dataset=cell["dataset"], horizon=cell["horizon"], seed=selected["seed"],
            best_epoch=selected["best_epoch"], mae=result["test"]["mae"], mse=result["test"]["mse"],
        ))
    frame = pd.DataFrame(results)
    frame.to_csv(root/"test_results.csv", index=False)
    summary = frame.groupby(["dataset", "horizon"])[["mae", "mse"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(root/"summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="extensions/final_method_standard/configs/study.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpus", help="Comma-separated visible GPU indices")
    parser.add_argument("--run-name")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--horizons", nargs="+", type=int)
    args = parser.parse_args()
    study = load_study(args.config)
    for field in ("run_name", "datasets", "horizons"):
        value = getattr(args, field)
        if value is not None:
            study[field] = value
    if args.gpus:
        study["resources"]["gpus"] = [int(index) for index in args.gpus.split(",")]
    seeds = study["grid"]["confirmation_seeds"]
    if len(set(seeds)) != 5 or len(seeds) != 5 or seeds[0] != study["grid"]["search_seed"]:
        parser.error("Configure five distinct seeds, starting with search_seed.")
    if study["evaluation"] != {"mode": "overall"}:
        parser.error("This workflow requires overall evaluation.")
    root = run_root(study)
    studies = list(cell_studies(study, root))
    print(f"{len(studies)} cells × {len(seeds)} seeds = {len(studies)*len(seeds)} models")
    for cell, single in studies:
        print(case_key(cell), single["model"]["moving_avg"], single["grid"]["batch_sizes"][0],
              single["grid"]["learning_rates"][0], single["fixed_weights"])
    if not args.execute:
        return

    root.mkdir(parents=True, exist_ok=True)
    lock = root/"coordinator.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        protocol = {key: value for key, value in study.items() if key != "resources"}
        protocol_path = root/"protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError("Run protocol changed; use a new --run-name.")
        save_json(protocol_path, protocol)
        execute(study, studies, root)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
