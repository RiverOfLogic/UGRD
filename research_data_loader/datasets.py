import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .time_features import time_features


class BaseForecastDataset(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        if flag not in {"train", "val", "test", "pred"}:
            raise ValueError(f"Unsupported flag: {flag}")

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.scaler: Optional[StandardScaler] = None
        self.target_idx = 0
        self.feature_names = []
        self.full_data = None
        self.full_stamp = None
        self.sample_indices = None
        self.__read_data__()

    @property
    def output_slice(self):
        if self.features == "MS":
            return slice(self.target_idx, self.target_idx + 1)
        return slice(None)

    @property
    def output_dim(self):
        if self.features == "MS":
            return 1
        return len(self.feature_names)

    def _read_csv(self):
        return pd.read_csv(os.path.join(self.root_path, self.data_path))

    def _select_features(self, df_raw):
        if self.features in {"M", "MS"}:
            cols_data = list(df_raw.columns[1:])
            df_data = df_raw.loc[:, cols_data]
        elif self.features == "S":
            df_data = df_raw.loc[:, [self.target]]
        else:
            raise ValueError(f"Unsupported features mode: {self.features}")

        self.feature_names = list(df_data.columns)
        self.target_idx = self.feature_names.index(self.target) if self.target in self.feature_names else 0
        return df_data

    def _build_time_features(self, date_series, total_length):
        if date_series is None:
            return np.zeros((total_length, 1), dtype=np.float32)

        df_stamp = pd.DataFrame({"date": pd.to_datetime(date_series)})
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].dt.month
            df_stamp["day"] = df_stamp["date"].dt.day
            df_stamp["weekday"] = df_stamp["date"].dt.weekday
            df_stamp["hour"] = df_stamp["date"].dt.hour
            if self.freq.lower() in {"t", "min", "15min"}:
                df_stamp["minute"] = df_stamp["date"].dt.minute // 15
            return df_stamp.drop(columns=["date"]).to_numpy(dtype=np.float32)

        encoded = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
        return encoded.transpose(1, 0).astype(np.float32)

    def _assign_data(self, values, borders, date_series=None):
        """Hardcoded TSL-style three-way split.

        borders = [train_end, val_end, total_length]

        border1s = [0,              train_end - seq_len,  val_end - seq_len]
        border2s = [train_end,      val_end,              total_length    ]

        Non-train split start points are shifted back by seq_len so the first
        sample's history window can reach into the previous split.
        """
        train_end, val_end, total_length = int(borders[0]), int(borders[1]), int(borders[2])

        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, total_length]

        # Only keep data within [0, total_length); discard anything beyond.
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(values[border1s[0]:border2s[0]])
            data = self.scaler.transform(values[:total_length]).astype(np.float32)
        else:
            data = values[:total_length].astype(np.float32)

        self.full_data = data
        self.full_stamp = self._build_time_features(
            date_series[:total_length] if date_series is not None else None,
            total_length,
        )

        type_map = {"train": 0, "val": 1, "test": 2}
        set_type = type_map[self.flag]
        border1 = border1s[set_type]
        border2 = border2s[set_type]

        min_start = border1
        max_start = border2 - self.pred_len - self.seq_len
        if max_start < min_start:
            raise ValueError(
                f"Split '{self.flag}' is too short for seq_len={self.seq_len} and pred_len={self.pred_len}."
            )
        self.sample_indices = np.arange(min_start, max_start + 1, dtype=np.int64)

    def inverse_transform(self, data, target_only=False):
        if self.scaler is None:
            return data

        is_tensor = isinstance(data, torch.Tensor)
        array = data.detach().cpu().numpy() if is_tensor else np.asarray(data)
        original_shape = array.shape
        flat = array.reshape(-1, original_shape[-1])

        full_dim = len(self.scaler.mean_)
        if flat.shape[-1] == full_dim and not target_only:
            restored = self.scaler.inverse_transform(flat)
        elif flat.shape[-1] == 1:
            index = self.target_idx if (target_only or self.features == "MS") else 0
            restored = flat * self.scaler.scale_[index] + self.scaler.mean_[index]
        else:
            raise ValueError(
                f"Cannot inverse transform shape {original_shape}; expected last dim 1 or {full_dim}."
            )

        restored = restored.reshape(original_shape)
        if is_tensor:
            return torch.as_tensor(restored, dtype=data.dtype, device=data.device)
        return restored

    def __getitem__(self, index):
        s_begin = int(self.sample_indices[index])
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.full_data[s_begin:s_end]
        seq_y = self.full_data[r_begin:r_end]
        seq_x_mark = self.full_stamp[s_begin:s_end]
        seq_y_mark = self.full_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_indices)


class Dataset_ETT_hour(BaseForecastDataset):
    """ETT hour: 12 months train, 4 months val, 4 months test."""

    def __read_data__(self):
        df_raw = self._read_csv()
        df_data = self._select_features(df_raw)
        borders = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        self._assign_data(df_data.to_numpy(dtype=np.float32), borders, date_series=df_raw["date"])


class Dataset_ETT_minute(BaseForecastDataset):
    """ETT minute: 12 months train, 4 months val, 4 months test."""

    def __read_data__(self):
        df_raw = self._read_csv()
        df_data = self._select_features(df_raw)
        borders = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        self._assign_data(df_data.to_numpy(dtype=np.float32), borders, date_series=df_raw["date"])


class Dataset_Custom(BaseForecastDataset):
    """Custom: 70% train, 10% val (remainder), 20% test."""

    def __read_data__(self):
        df_raw = self._read_csv()
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        df_data = self._select_features(df_raw)

        N = len(df_raw)
        num_train = int(N * 0.7)
        num_test = int(N * 0.2)
        num_vali = N - num_train - num_test
        borders = [num_train, num_train + num_vali, N]
        self._assign_data(df_data.to_numpy(dtype=np.float32), borders, date_series=df_raw["date"])


class Dataset_PEMS(BaseForecastDataset):
    """PEMS: 70% train, 10% val, 20% test (same as Custom)."""

    def __read_data__(self):
        data_file = os.path.join(self.root_path, self.data_path)
        raw = np.load(data_file, allow_pickle=True)["data"][:, :, 0].astype(np.float32)
        frame = pd.DataFrame(raw).ffill().bfill()
        self.feature_names = [f"sensor_{idx}" for idx in range(frame.shape[1])]
        self.target_idx = 0

        N = len(frame)
        num_train = int(N * 0.7)
        num_test = int(N * 0.2)
        num_vali = N - num_train - num_test
        borders = [num_train, num_train + num_vali, N]
        self._assign_data(frame.to_numpy(dtype=np.float32), borders, date_series=None)


class Dataset_Solar(BaseForecastDataset):
    """Solar: 70% train, 10% val, 20% test (same as Custom)."""

    def __read_data__(self):
        rows = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding="utf-8") as handle:
            for line in handle:
                rows.append([float(item) for item in line.strip().split(",")])

        raw = np.asarray(rows, dtype=np.float32)
        self.feature_names = [f"value_{idx}" for idx in range(raw.shape[1])]
        self.target_idx = 0

        N = len(raw)
        num_train = int(N * 0.7)
        num_test = int(N * 0.2)
        num_vali = N - num_train - num_test
        borders = [num_train, num_train + num_vali, N]
        self._assign_data(raw, borders, date_series=None)


class Dataset_Pred(BaseForecastDataset):
    def __init__(
        self,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="15min",
        cols=None,
    ):
        self.inverse = inverse
        self.cols = cols
        super().__init__(
            root_path=root_path,
            flag=flag,
            size=size,
            features=features,
            data_path=data_path,
            target=target,
            scale=scale,
            timeenc=timeenc,
            freq=freq,
        )

    def __read_data__(self):
        df_raw = self._read_csv()
        if self.cols:
            cols = list(self.cols)
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)
        df_data = self._select_features(df_raw)
        values = df_data.to_numpy(dtype=np.float32)

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(values)
            data = self.scaler.transform(values).astype(np.float32)
        else:
            data = values.astype(np.float32)

        tail_dates = pd.to_datetime(df_raw.loc[border1 : border2 - 1, "date"])
        pred_dates = pd.date_range(tail_dates.iloc[-1], periods=self.pred_len + 1, freq=self.freq)
        full_dates = pd.Series(list(tail_dates.values) + list(pred_dates[1:]))
        data_stamp = self._build_time_features(full_dates, len(full_dates))

        self.full_data = data.astype(np.float32)
        self.full_stamp = np.concatenate(
            [
                self._build_time_features(df_raw["date"], len(df_raw)),
                data_stamp[self.seq_len :],
            ],
            axis=0,
        )
        self.sample_indices = np.array([border1], dtype=np.int64)

    def __getitem__(self, index):
        s_begin = int(self.sample_indices[index])
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.full_data[s_begin:s_end]
        seq_y = self.full_data[r_begin : r_begin + self.label_len]
        seq_x_mark = self.full_stamp[s_begin:s_end]
        seq_y_mark = self.full_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
