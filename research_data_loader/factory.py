from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from .cache_dataset import ChunkShuffleSampler, build_teacher_cache_dataset, teacher_cache_collate_fn
from .datasets import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_PEMS,
    Dataset_Pred,
    Dataset_Solar,
)

DATASET_REGISTRY = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "ECL": Dataset_Custom,
    "WTH": Dataset_Custom,
    "Weather": Dataset_Custom,
    "Exchange": Dataset_Custom,
    "Traffic": Dataset_Custom,
    "Illness": Dataset_Custom,
    "Solar": Dataset_Solar,
    "PEMS": Dataset_PEMS,
    "custom": Dataset_Custom,
}


def forecast_collate_fn(batch):
    batch_size = len(batch)
    seq_len = batch[0][0].shape[0]
    y_len = max(item[1].shape[0] for item in batch)
    x_mark_dim = max(item[2].shape[-1] for item in batch)
    y_mark_len = max(item[3].shape[0] for item in batch)
    y_mark_dim = max(item[3].shape[-1] for item in batch)
    max_channels = max(item[0].shape[-1] for item in batch)

    seq_x = torch.zeros(batch_size, seq_len, max_channels, dtype=torch.float32)
    seq_y = torch.zeros(batch_size, y_len, max_channels, dtype=torch.float32)
    seq_x_mark = torch.zeros(batch_size, seq_len, x_mark_dim, dtype=torch.float32)
    seq_y_mark = torch.zeros(batch_size, y_mark_len, y_mark_dim, dtype=torch.float32)
    channel_mask = torch.zeros(batch_size, max_channels, dtype=torch.float32)

    for idx, (x, y, x_mark, y_mark) in enumerate(batch):
        channels = x.shape[-1]
        seq_x[idx, :, :channels] = torch.as_tensor(x, dtype=torch.float32)
        seq_y[idx, : y.shape[0], :channels] = torch.as_tensor(y, dtype=torch.float32)
        seq_x_mark[idx, :, : x_mark.shape[-1]] = torch.as_tensor(x_mark, dtype=torch.float32)
        seq_y_mark[idx, : y_mark.shape[0], : y_mark.shape[-1]] = torch.as_tensor(y_mark, dtype=torch.float32)
        channel_mask[idx, :channels] = 1.0

    return seq_x, seq_y, seq_x_mark, seq_y_mark, channel_mask


class MultiSourceDataLoader:
    def __init__(self, sources, strategy="round_robin"):
        self.sources = sources
        self.strategy = strategy

    def __len__(self):
        return sum(len(source["loader"]) for source in self.sources)

    def __iter__(self):
        if self.strategy == "sequential":
            for source in self.sources:
                for batch in source["loader"]:
                    yield {"dataset": source["dataset"], "batch": batch, "name": source["name"]}
            return

        iterators = {idx: iter(source["loader"]) for idx, source in enumerate(self.sources)}
        active = list(iterators.keys())
        while active:
            next_active = []
            for idx in active:
                try:
                    batch = next(iterators[idx])
                    source = self.sources[idx]
                    yield {"dataset": source["dataset"], "batch": batch, "name": source["name"]}
                    next_active.append(idx)
                except StopIteration:
                    continue
            active = next_active


def _base_dataset_config(args):
    return {
        "data": getattr(args, "data", None),
        "root_path": getattr(args, "root_path", None),
        "data_path": getattr(args, "data_path", None),
        "features": getattr(args, "features", "S"),
        "target": getattr(args, "target", None),
        "scale": getattr(args, "scale", True),
        "freq": getattr(args, "freq", "h"),
        "embed": getattr(args, "embed", "timeF"),
        "inverse": getattr(args, "inverse", False),
        "cols": getattr(args, "cols", None),
    }


def _merge_dataset_config(base_config, override_config):
    merged = deepcopy(base_config)
    for key, value in override_config.items():
        merged[key] = value
    return merged


def build_dataset(spec, args, flag):
    normalized_flag = flag.lower()
    required_keys = ("data", "root_path", "data_path", "features", "target")
    missing = [key for key in required_keys if spec.get(key) is None]
    if missing:
        raise ValueError(f"Dataset spec for flag '{flag}' is missing required fields: {missing}")

    if normalized_flag == "pred":
        dataset_class = Dataset_Pred
    else:
        if spec["data"] not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset type: {spec['data']}.")
        dataset_class = DATASET_REGISTRY[spec["data"]]

    timeenc = 0 if spec.get("embed", args.embed) != "timeF" else 1
    kwargs = dict(
        root_path=spec["root_path"],
        data_path=spec["data_path"],
        flag=normalized_flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=spec["features"],
        target=spec["target"],
        scale=spec.get("scale", args.scale),
        timeenc=timeenc,
        freq=spec.get("freq", args.freq),
    )
    if normalized_flag == "pred":
        kwargs["inverse"] = spec.get("inverse", getattr(args, "inverse", False))
        kwargs["cols"] = spec.get("cols", getattr(args, "cols", None))

    dataset = dataset_class(**kwargs)
    dataset.dataset_name = spec.get("name", spec["data"])
    return dataset


def build_loader(dataset, args, flag):
    normalized_flag = flag.lower()
    shuffle_flag = normalized_flag == "train"
    drop_last = normalized_flag == "train"
    collate_fn = teacher_cache_collate_fn if getattr(dataset, "is_teacher_cache", False) else forecast_collate_fn
    sampler = None
    teacher_cache = getattr(args, "teacher_cache", {}) or {}
    if getattr(dataset, "is_teacher_cache", False) and normalized_flag == "train":
        shuffle_mode = str(teacher_cache.get("shuffle_mode", "chunk")).lower()
        if shuffle_mode == "chunk":
            sampler = ChunkShuffleSampler(dataset, chunk_size=int(teacher_cache.get("chunk_size", 2048)))
            shuffle_flag = False
        elif shuffle_mode == "global":
            shuffle_flag = True
        elif shuffle_mode in {"none", "sequential"}:
            shuffle_flag = False
        else:
            raise ValueError(f"Unsupported teacher cache shuffle_mode: {shuffle_mode}")
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=bool(teacher_cache.get("pin_memory", False)) if getattr(dataset, "is_teacher_cache", False) else False,
    )
    if args.num_workers > 0 and getattr(dataset, "is_teacher_cache", False):
        dataloader_kwargs["persistent_workers"] = bool(teacher_cache.get("persistent_workers", False))
        if teacher_cache.get("prefetch_factor") is not None:
            dataloader_kwargs["prefetch_factor"] = int(teacher_cache["prefetch_factor"])
    return DataLoader(**dataloader_kwargs)


def _teacher_cache_enabled(args, flag):
    teacher_cache = getattr(args, "teacher_cache", None)
    return flag.lower() == "train" and isinstance(teacher_cache, dict) and bool(teacher_cache.get("enabled", False))


def build_cached_dataset(spec, args):
    dataset = build_teacher_cache_dataset(spec, args)
    if getattr(dataset, "needs_raw_dataset", False):
        dataset.set_raw_dataset(build_dataset(spec, args, "train"))
    dataset.dataset_name = spec.get("name", spec["data"])
    dataset.is_teacher_cache = True
    return dataset


def _resolve_source_specs(args, flag):
    singular_key = f"{flag.lower()}_dataset"
    plural_key = f"{flag.lower()}_datasets"
    if hasattr(args, plural_key) and getattr(args, plural_key) is not None:
        specs = getattr(args, plural_key)
        if not isinstance(specs, list):
            raise ValueError(f"{plural_key} must be a list of dataset specs.")
        return specs
    if hasattr(args, singular_key) and getattr(args, singular_key) is not None:
        return [getattr(args, singular_key)]
    return [_base_dataset_config(args)]


def data_provider(args, flag):
    source_specs = [_merge_dataset_config(_base_dataset_config(args), spec) for spec in _resolve_source_specs(args, flag)]
    built_sources = []
    for spec in source_specs:
        source_args = deepcopy(args)
        if "teacher_cache" in spec:
            source_args.teacher_cache = spec["teacher_cache"]
        dataset = build_cached_dataset(spec, source_args) if _teacher_cache_enabled(source_args, flag) else build_dataset(spec, source_args, flag)
        built_sources.append(
            {
                "name": spec.get("name", spec["data"]),
                "dataset": dataset,
                "loader": build_loader(dataset, source_args, flag),
            }
        )

    if flag.lower() in {"test", "pred"} and len(built_sources) > 1:
        raise ValueError(f"{flag} currently supports a single dataset source for clear evaluation artifacts.")

    if len(built_sources) == 1:
        return built_sources[0]["dataset"], built_sources[0]["loader"]

    multi_strategy = getattr(args, "multi_dataset_strategy", "round_robin")
    return [source["dataset"] for source in built_sources], MultiSourceDataLoader(built_sources, strategy=multi_strategy)
