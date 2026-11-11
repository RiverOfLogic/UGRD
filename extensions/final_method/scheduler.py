"""Visible-GPU job scheduling with measured budgets and startup reservations."""
import subprocess
import sys
import time

import torch

from .common import ROOT, read_json, save_json


def effective_free(free_bytes, running, reserve_bytes):
    """Free already includes resident allocations; reserve only unobserved headroom."""
    headroom = sum(max(item["budget_bytes"]-item.get("reserved_bytes", 0), 0) for item in running)
    return free_bytes-reserve_bytes-headroom


def gpu_ids(resources):
    count = torch.cuda.device_count()
    ids = list(range(count)) if resources["gpus"] == "auto" else list(map(int, resources["gpus"]))
    if not ids or any(i<0 or i>=count for i in ids):
        raise ValueError("Requested GPU index is not visible under CUDA_VISIBLE_DEVICES")
    return ids


def schedule(jobs, resources, *, stage):
    running = []
    try:
        _schedule(jobs, resources, stage=stage, running=running)
    finally:
        for item in running:
            if item["process"].poll() is None:
                item["process"].terminate()
        for item in running:
            try:
                item["process"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait()
            item["log"].close()


def _schedule(jobs, resources, *, stage, running):
    ids = gpu_ids(resources)
    pending = []
    for job in jobs:
        path = job["output_path"]
        if (path/"result.json").exists():
            if read_json(path/"result.json")["job_signature"] != job["job_signature"]:
                raise ValueError(f"Completed output has another recipe: {path}")
            continue
        pending.append(job)
    failures = []
    limit_per_gpu = resources["max_per_gpu"] if stage in ("grid", "confirm", "test") else 1
    max_global = resources["max_parallel"] or len(ids)*limit_per_gpu
    while pending or running:
        alive = []
        for item in running:
            code = item["process"].poll()
            if code is None:
                hb = item["job"]["output_path"]/"heartbeat.json"
                if hb.exists():
                    try:
                        item["reserved_bytes"] = read_json(hb).get("reserved_bytes", 0)
                    except (FileNotFoundError, PermissionError):
                        # Windows atomic replacement can briefly deny shared reads.
                        # Zero resident credit conservatively retains the full job budget.
                        item["reserved_bytes"] = 0
                alive.append(item)
            else:
                item["log"].close()
                if code != 0:
                    failures.append({"job": item["job"]["id"], "exit_code": code,
                                     "log": str(item["job"]["output_path"]/"process.log")})
        running[:] = alive
        if failures and not running:
            save_json(jobs[0]["output_path"].parents[1]/f"{stage}_failures.json", failures)
            raise RuntimeError(f"{stage} failed; new launches stopped. See logs: {failures}")
        launched = False
        while pending and not failures and len(running)<max_global:
            options = []
            for gpu in ids:
                assigned = [r for r in running if r["gpu"]==gpu]
                if len(assigned)>=limit_per_gpu:
                    continue
                free, _ = torch.cuda.mem_get_info(gpu)
                capacity = effective_free(free, assigned, resources["reserve_gib"]*2**30)
                options.append((capacity, gpu))
            if not options:
                break
            capacity, gpu = max(options)
            candidate = next((j for j in pending if j["budget_bytes"]<=capacity), None)
            if candidate is None:
                break
            pending.remove(candidate)
            directory = candidate["output_path"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory/"heartbeat.json").unlink(missing_ok=True)
            log = (directory/"process.log").open("a", encoding="utf-8")
            process = subprocess.Popen([candidate.get("python_executable", sys.executable), "-u", "-m", candidate.get("worker_module", "extensions.final_method.worker"),
                                         "--job", str(candidate["job_file"]), "--device", f"cuda:{gpu}"],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append({"process": process, "job": candidate, "gpu": gpu, "log": log,
                            "budget_bytes": candidate["budget_bytes"], "reserved_bytes": 0})
            print(f"LAUNCH {candidate['id']} cuda:{gpu} budget={candidate['budget_bytes']/2**30:.2f}GiB", flush=True)
            launched = True
        if pending and not running:
            available = max(torch.cuda.mem_get_info(i)[0] for i in ids)-resources["reserve_gib"]*2**30
            needed = min(j["budget_bytes"] for j in pending)
            raise RuntimeError(f"No job fits current free GPU memory: minimum budget {needed/2**30:.2f}GiB, free after reserve {available/2**30:.2f}GiB; no batch/precision fallback")
        if running:
            time.sleep(resources["poll_seconds"])
