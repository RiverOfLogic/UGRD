"""Thin training cache: raw train-scaled input, compact teacher responses shared by jobs."""
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from research_data_loader.cache_dataset import teacher_cache_collate_fn
from research_data_loader.factory import build_dataset
from .common import file_hash, read_json, resolve


def raw_dataset(study, cell, flag='train'):
    args = SimpleNamespace(seq_len=study.get("origin_context", study["teacher_context"]), pred_len=cell["horizon"],
                           label_len=0, embed="timeF", scale=True, freq=cell["freq"])
    spec = {"data": cell["loader"], "data_path": cell["data_path"], "features": "M", "target": "OT",
            "freq": cell["freq"], "root_path": str(resolve(study["data_root"])), "scale": True}
    return build_dataset(spec, args, flag)


class EpochSampler(Sampler):
    def __init__(self, size, seed):
        self.size, self.seed, self.epoch = size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed*100003+self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())


class CachedTraining(Dataset):
    def __init__(self, config):
        settings = config["final_method"]
        self.root = resolve(settings["cache_dir"])
        self.metadata = read_json(self.root/"metadata.json")
        if self.metadata["signature"] != settings["cache_signature"]:
            raise ValueError("Training cache differs from resolved config")
        cell = settings["cell"]
        study = {"teacher_context": config["teacher"]["context_length"], "data_root": config["data"]["root_path"]}
        self.raw = raw_dataset(study, cell)
        if file_hash(resolve(study["data_root"])/cell["data_path"]) != self.metadata["contract"]["dataset_sha256"]:
            raise ValueError("Dataset changed after teacher cache preparation")
        self.starts = np.load(self.root/"starts.npy")
        self.mask = torch.tensor(np.load(self.root/"active.npy"), dtype=torch.float32)
        self.arrays = {}
        self.dataset_name = cell["dataset"]
        self.features, self.output_slice = "M", slice(None)
        self.context = config["teacher"]["context_length"]
        self.student = config["model"]["seq_len"]
        self.horizon = config["model"]["pred_len"]

    def __len__(self):
        return len(self.starts)

    def array(self, key):
        if key not in self.arrays:
            self.arrays[key] = np.load(self.root/f"{key}.npy", mmap_mode="r")
        return self.arrays[key]

    def __getitem__(self, index):
        origin = int(self.starts[index])+self.context
        def tensor(key):
            return torch.from_numpy(np.array(self.array(key)[index]))
        result = {"seq_x": torch.tensor(self.raw.full_data[origin-self.student:origin]),
                "seq_y": torch.tensor(self.raw.full_data[origin:origin+self.horizon]),
                "channel_mask": self.mask, "sample_start_index": torch.tensor(int(self.starts[index])),
                "teacher_prediction": tensor("median"), "teacher_q01": tensor("lower"), "teacher_q09": tensor("upper"),
                "teacher_hiddens": {}, "final_reg_gram": tensor("reg_gram")}
        if 'reg_raw.npy' in self.metadata['hashes']:
            result['final_reg_raw'] = tensor('reg_raw')
        return result


def collate_final_batch(items):
    batch = teacher_cache_collate_fn(items)
    # The shared trainer eagerly copies teacher_hiddens. Keep this objective-only
    # [B,C,C] target separate so ECL does not make a second unused GPU copy.
    batch["final_reg_gram"] = torch.stack([item["final_reg_gram"] for item in items])
    if 'final_reg_raw' in items[0]:
        batch['final_reg_raw'] = torch.stack([item['final_reg_raw'] for item in items])
    return batch


def train_provider(config, names):
    dataset = CachedTraining(config)
    workers = config["train"]["num_workers"]
    extra = {"persistent_workers": True, "prefetch_factor": 2} if workers else {}
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"],
                        sampler=EpochSampler(len(dataset), config["train"]["seed"]), drop_last=True,
                        num_workers=workers, pin_memory=True, collate_fn=collate_final_batch,
                        generator=torch.Generator().manual_seed(config["train"]["seed"]+41), **extra)
    return dataset, loader
