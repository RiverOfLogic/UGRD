import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from distillation_common.teacher_backend import teacher_fingerprint_options


def _resolve_path(value: str | Path, project_root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = project_root or Path(__file__).resolve().parents[1]
    return root / path


def _canonical_hidden_target(target: dict[str, Any]) -> dict[str, Any]:
    canonical = {
        "teacher_layer_index": int(target.get("teacher_layer_index", target.get("layer_index", 11))),
    }
    if "teacher_tokens" in target:
        token_cfg = target["teacher_tokens"]
        canonical["teacher_tokens"] = {
            "mode": token_cfg.get("mode", "explicit"),
            "tokens": [str(token) for token in token_cfg.get("tokens", [])],
        }
    else:
        canonical["teacher_token"] = target.get("teacher_token", target.get("token", "future"))
    return canonical


def hidden_target_signature(target: dict[str, Any]) -> str:
    encoded = json.dumps(_canonical_hidden_target(target), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def base_cache_fingerprint(
    *,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    dataset_spec: dict[str, Any],
    teacher_config: dict[str, Any],
    dataset_name: str,
) -> str:
    payload = {
        "model": {
            "seq_len": model_config["seq_len"],
            "pred_len": model_config["pred_len"],
        },
        "data": {
            "scale": data_config.get("scale", True),
            "embed": data_config.get("embed", "timeF"),
            "label_len": data_config.get("label_len", 0),
        },
        "dataset": {
            "name": dataset_name,
            "data": dataset_spec.get("data"),
            "data_path": dataset_spec.get("data_path"),
            "features": dataset_spec.get("features"),
            "target": dataset_spec.get("target"),
            "freq": dataset_spec.get("freq"),
            "split": dataset_spec.get("split"),
            "root_path": dataset_spec.get("root_path"),
        },
        "teacher": {
            "path": teacher_config.get("path"),
            "prediction_length": teacher_config.get("prediction_length"),
            "context_length": teacher_config.get("context_length"),
            "prediction_mode": teacher_config.get("prediction_mode", "multivariate"),
        },
    }
    payload["teacher"].update(teacher_fingerprint_options(teacher_config))
    # Match teacher_base_fingerprint; preserve legacy fingerprints for default bounds.
    q_lower = float(teacher_config.get("quantile_lower", 0.1))
    q_upper = float(teacher_config.get("quantile_upper", 0.9))
    if (q_lower, q_upper) != (0.1, 0.9):
        payload["teacher"]["quantile_lower"] = q_lower
        payload["teacher"]["quantile_upper"] = q_upper
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slice_from_json(value: dict[str, int | None]) -> slice:
    return slice(value.get("start"), value.get("stop"), value.get("step"))


def _tensor_from_array(value: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value).copy()).float()


class ChunkShuffleSampler(Sampler[int]):
    """Shuffle chunk order and sample order within chunks to reduce random cache IO."""

    def __init__(self, data_source: Dataset, chunk_size: int = 2048) -> None:
        self.data_source = data_source
        self.chunk_size = max(1, int(chunk_size))

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        chunks = [(start, min(start + self.chunk_size, n)) for start in range(0, n, self.chunk_size)]
        if not chunks:
            return iter(())
        chunk_order = torch.randperm(len(chunks)).tolist()
        indices: list[int] = []
        for chunk_idx in chunk_order:
            start, end = chunks[chunk_idx]
            local = torch.randperm(end - start).add(start).tolist()
            indices.extend(local)
        return iter(indices)


class TeacherCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        required_fields: list[str] | None = None,
        required_hidden_targets: dict[str, str] | None = None,
        required_hidden_specs: dict[str, dict[str, Any]] | None = None,
        validate_config: bool = True,
        expected_base_fingerprint: str | None = None,
        expected_metadata: dict[str, Any] | None = None,
        requested_format: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.metadata_path = self.cache_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Teacher cache metadata not found: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.cache_format = self.metadata.get("format")
        if requested_format and requested_format != self.cache_format:
            raise ValueError(
                f"Teacher cache format mismatch for {self.cache_dir}: "
                f"expected {requested_format}, found {self.cache_format}."
            )
        if self.cache_format != "memmap":
            raise ValueError(
                f"Unsupported teacher cache format for {self.cache_dir}: {self.cache_format}. "
                "Only v2 memmap cache is supported; rebuild this cache with cache.format=memmap."
            )

        self.dataset_name = self.metadata["dataset"]
        self._validate_expected_metadata(expected_metadata or {})
        self.features = self.metadata.get("features", "M")
        self.target = self.metadata.get("target")
        self.feature_names = list(self.metadata.get("channel_names", []))
        self.output_slice = _slice_from_json(self.metadata.get("output_slice", {}))
        self.required_fields = list(required_fields or ["teacher_prediction"])
        self.hidden_name_map = self._resolve_hidden_name_map(
            required_hidden_targets or {},
            required_hidden_specs or {},
        )

        if validate_config and expected_base_fingerprint is not None:
            actual_base = self.metadata.get("base_config_fingerprint") or self.metadata.get("config_fingerprint")
            if actual_base != expected_base_fingerprint:
                raise ValueError(
                    f"Teacher cache base fingerprint mismatch for {self.cache_dir}: "
                    f"expected {expected_base_fingerprint}, found {actual_base}."
                )

        available_fields = set(self.metadata.get("available_fields", []))
        if not available_fields:
            available_fields = {"seq_x", "seq_y", "channel_mask", "teacher_prediction", "teacher_hiddens"}
        self.available_fields = available_fields
        missing_fields = [field for field in self.required_fields if field not in available_fields]
        if missing_fields:
            raise ValueError(f"Teacher cache is missing required fields: {missing_fields}")

        self.raw_dataset = None
        self.needs_raw_dataset = not {"seq_x", "seq_y"}.issubset(available_fields)
        self._arrays: dict[str, np.ndarray] = {}
        self.sample_count = int(self.metadata["sample_count"])
        self.files = dict(self.metadata.get("files", {}))
        self.hidden_files = dict(self.metadata.get("hidden_files", {}))

    def _validate_expected_metadata(self, expected_metadata: dict[str, Any]) -> None:
        for key, expected_value in expected_metadata.items():
            if expected_value is None or key not in self.metadata:
                continue
            actual_value = self.metadata[key]
            if isinstance(expected_value, int):
                matches = int(actual_value) == int(expected_value)
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(
                    f"Teacher cache metadata mismatch for {self.cache_dir}: "
                    f"{key} expected {expected_value!r}, found {actual_value!r}. "
                    "Rebuild the cache or use a cache name that matches this distillation config."
                )

    @property
    def output_dim(self) -> int:
        if self.features == "MS":
            return 1
        return len(self.feature_names)

    def __len__(self) -> int:
        return self.sample_count

    def set_raw_dataset(self, dataset: Dataset) -> None:
        if len(dataset) < self.sample_count:
            raise ValueError(
                f"Raw dataset for thin teacher cache is shorter than cache: "
                f"raw={len(dataset)}, cache={self.sample_count}."
            )
        self.raw_dataset = dataset

    def _resolve_hidden_name_map(
        self,
        required_hidden_targets: dict[str, str],
        required_hidden_specs: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        hidden_targets = self.metadata.get("hidden_targets", {})
        hidden_signatures = self.metadata.get("hidden_request_signatures", {})
        signature_to_cache_name = {signature: name for name, signature in hidden_signatures.items()}
        result = {}
        missing = []
        for requested_name, cache_name in required_hidden_targets.items():
            if cache_name in hidden_targets or cache_name in hidden_signatures:
                result[requested_name] = cache_name
            elif requested_name in hidden_targets or requested_name in hidden_signatures:
                result[requested_name] = requested_name
            else:
                missing.append(f"{requested_name}->{cache_name}")
        for requested_name, spec in required_hidden_specs.items():
            signature = hidden_target_signature(spec)
            cache_name = signature_to_cache_name.get(signature)
            if cache_name is None:
                missing.append(requested_name)
            else:
                result[requested_name] = cache_name
        if missing:
            raise ValueError(f"Teacher cache is missing required hidden targets: {missing}")
        return result

    def _array(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            if key.startswith("teacher_hidden:"):
                cache_name = key.split(":", 1)[1]
                filename = self.hidden_files[cache_name]
            else:
                filename = self.files[key]
            self._arrays[key] = np.load(self.cache_dir / filename, mmap_mode="r")
        return self._arrays[key]

    def _raw_inputs(self, index: int) -> tuple[Any, Any]:
        if self.raw_dataset is None:
            raise RuntimeError(
                f"Teacher cache at {self.cache_dir} does not store seq_x/seq_y. "
                "Attach the matching raw train dataset before reading it."
            )
        seq_x, seq_y, _, _ = self.raw_dataset[index]
        return seq_x, seq_y

    def _memmap_item(self, index: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "channel_mask": _tensor_from_array(self._array("channel_mask")[index]),
            "sample_start_index": torch.as_tensor(self._array("sample_start_indices")[index]).long(),
        }
        if "seq_x" in self.available_fields and "seq_y" in self.available_fields:
            result["seq_x"] = _tensor_from_array(self._array("seq_x")[index])
            result["seq_y"] = _tensor_from_array(self._array("seq_y")[index])
        else:
            seq_x, seq_y = self._raw_inputs(index)
            result["seq_x"] = torch.as_tensor(seq_x).float()
            result["seq_y"] = torch.as_tensor(seq_y).float()
        if "teacher_prediction" in self.required_fields:
            result["teacher_prediction"] = _tensor_from_array(self._array("teacher_prediction")[index])
        if "teacher_q01" in self.required_fields:
            result["teacher_q01"] = _tensor_from_array(self._array("teacher_q01")[index])
        if "teacher_q09" in self.required_fields:
            result["teacher_q09"] = _tensor_from_array(self._array("teacher_q09")[index])
        if "teacher_uncertainty" in self.required_fields:
            result["teacher_uncertainty"] = _tensor_from_array(self._array("teacher_uncertainty")[index])
        teacher_hiddens = {}
        for requested_name, cache_name in self.hidden_name_map.items():
            teacher_hiddens[requested_name] = _tensor_from_array(self._array(f"teacher_hidden:{cache_name}")[index])
        result["teacher_hiddens"] = teacher_hiddens
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._memmap_item(int(index))


def teacher_cache_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "seq_x": torch.stack([item["seq_x"] for item in batch], dim=0),
        "seq_y": torch.stack([item["seq_y"] for item in batch], dim=0),
        "channel_mask": torch.stack([item["channel_mask"] for item in batch], dim=0),
        "sample_start_indices": torch.stack([item["sample_start_index"] for item in batch], dim=0),
    }
    if "teacher_prediction" in batch[0]:
        result["teacher_prediction"] = torch.stack([item["teacher_prediction"] for item in batch], dim=0)
    if "teacher_q01" in batch[0]:
        result["teacher_q01"] = torch.stack([item["teacher_q01"] for item in batch], dim=0)
    if "teacher_q09" in batch[0]:
        result["teacher_q09"] = torch.stack([item["teacher_q09"] for item in batch], dim=0)
    if "teacher_uncertainty" in batch[0]:
        result["teacher_uncertainty"] = torch.stack([item["teacher_uncertainty"] for item in batch], dim=0)
    hidden_names = list(batch[0].get("teacher_hiddens", {}).keys())
    result["teacher_hiddens"] = {
        name: torch.stack([item["teacher_hiddens"][name] for item in batch], dim=0)
        for name in hidden_names
    }
    return result


def build_teacher_cache_dataset(spec: dict[str, Any], args: Any) -> TeacherCacheDataset:
    teacher_cache = getattr(args, "teacher_cache", {}) or {}
    dataset_name = spec.get("name", spec["data"])
    if teacher_cache.get("path"):
        cache_root = _resolve_path(teacher_cache["path"])
        cache_dir = cache_root / dataset_name
    else:
        cache_root = _resolve_path(teacher_cache["dir"])
        cache_name = teacher_cache.get("name")
        cache_dir = cache_root / cache_name / dataset_name if cache_name else cache_root / dataset_name
    expected_base = teacher_cache.get("base_fingerprint")
    return TeacherCacheDataset(
        cache_dir,
        required_fields=list(teacher_cache.get("required_fields", ["teacher_prediction"])),
        required_hidden_targets=dict(teacher_cache.get("required_hidden_targets", {})),
        required_hidden_specs=dict(teacher_cache.get("required_hidden_specs", {})),
        validate_config=bool(teacher_cache.get("validate_config", True)),
        expected_base_fingerprint=expected_base,
        expected_metadata={
            "seq_len": teacher_cache.get("expected_seq_len"),
            "student_seq_len": teacher_cache.get("expected_student_seq_len"),
            "data_context_length": teacher_cache.get("expected_data_context_length"),
            "pred_len": teacher_cache.get("expected_pred_len"),
            "label_len": teacher_cache.get("expected_label_len"),
            "teacher_context_length": teacher_cache.get("expected_teacher_context_length"),
            "teacher_prediction_length": teacher_cache.get("expected_teacher_prediction_length"),
        },
        requested_format=teacher_cache.get("format"),
    )
