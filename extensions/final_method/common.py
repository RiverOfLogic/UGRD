"""Paths, immutable cache identities and portable resolved configurations."""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT/p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_study(path):
    study = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if study["model"]["seq_len"] != 96:
        raise ValueError("The frozen parameter study applies to student history 96")
    return study


def cells(study):
    frame = pd.read_csv(resolve(study["parameter_table"]))
    result = []
    for row in frame.to_dict("records"):
        if row["dataset"] not in study["datasets"] or row["horizon"] not in study["horizons"]:
            continue
        row["horizon"] = int(row["horizon"])
        row["qbd_weights"] = json.loads(row["qbd_weights"])
        row["increment_enabled"] = str(row["increment_enabled"]).lower() == "true"
        result.append(row)
    expected = {(d, h) for d in study["datasets"] for h in study["horizons"]}
    if {(c["dataset"], c["horizon"]) for c in result} != expected or len(result) != len(expected):
        raise ValueError("Parameter table has missing or duplicate dataset/horizon cells")
    return result


def case_key(cell):
    return f"{cell['dataset']}_h{cell['horizon']}"


def run_root(study):
    return resolve(study["output_dir"])/study["run_name"]


def cache_root(study, cell):
    return resolve(study["cache_dir"])/case_key(cell)


def calibration_path(study, cell, batch):
    return cache_root(study, cell)/f"calibration_w{study['model']['moving_avg']}_b{batch}.json"


def setup(device, threads):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.set_device(target)
    return target


def config_for(study, cell, device, batch, learning_rate, seed, directory, *, weights=None):
    metadata = read_json(cache_root(study, cell)/"metadata.json")
    model = dict(study["model"], pred_len=cell["horizon"], enc_in=len(metadata["channels"]))
    train = dict(study["train"], device=device, batch_size=batch, learning_rate=learning_rate,
                 seed=seed, validation_only=True, best_metric="mse")
    if weights is None:
        if 'fixed_weights' in study:
            weights = study['fixed_weights']
            if set(weights) != {'qbd', 'reg', 'increment'} or not all(
                    isinstance(value, (int, float)) and 0 <= value < float('inf')
                    for value in weights.values()):
                raise ValueError('fixed_weights must contain three finite nonnegative coefficients')
        else:
            weights = read_json(calibration_path(study, cell, batch))["weights"]
        multipliers = study.get('loss_multipliers', {})
        weights = {key: value*multipliers.get(key, 1.) for key,value in weights.items()}
    return {"model": model, "train": train,
            "data": {"root_path": study["data_root"], "enabled_datasets": [cell["dataset"]],
                     "scale": True, "embed": "timeF", "label_len": 0,
                     "datasets": {cell["dataset"]: {"data": cell["loader"], "data_path": cell["data_path"],
                                                  "features": "M", "target": "OT", "freq": cell["freq"]}}},
            "teacher": {"backend": "chronos2", "path": study["teacher_path"], "context_length": study["teacher_context"],
                        "prediction_length": cell["horizon"], "cache": {"enabled": True}},
            "distillation": {"hard_weight": 0., "soft_weight": 0., "hidden_weight": 0., "hard_loss": "mse", "soft_loss": "mse"},
            "hidden_distillation": {"enabled": False, "layers": []},
            "visualization": {"enabled": False}, "output": {"dir": str(directory)},
            **({"comparison": study["comparison"]} if "comparison" in study else {}),
            "final_method": {"cell": cell, "cache_dir": str(cache_root(study, cell)), "weights": weights,
                             "cache_signature": metadata["signature"],
                             "heartbeat_path": str(Path(directory)/"heartbeat.json"),
                             "heartbeat_seconds": study["resources"]["heartbeat_seconds"]}}
