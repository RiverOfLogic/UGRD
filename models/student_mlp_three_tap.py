from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .student_mlp import _activation


@dataclass
class ThreeTapChannelIndependentMLPConfig:
    seq_len: int
    pred_len: int
    hidden_dim: int = 128
    num_layers: int = 3
    head_hidden_dim: int | None = None
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu", "tanh"] = "gelu"
    channel_weight_mode: Literal["shared", "separate"] = "shared"
    num_channels: int | None = None


class _ThreeTapTemporalMLP(nn.Module):
    """Three temporal hidden transforms with fixed distillation tap names."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden_dim: int,
        head_hidden_dim: int | None,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        head_hidden_dim = int(head_hidden_dim or hidden_dim)
        self.input_proj = nn.Linear(seq_len, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.activation1 = _activation(activation)
        self.activation2 = _activation(activation)
        self.activation3 = _activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, pred_len),
        )

    def encode_with_hidden_states(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.activation1(self.input_proj(x))
        hidden = self.dropout(hidden)
        middle_states = {"middle_1": hidden}

        hidden = self.activation2(self.linear2(self.norm2(hidden)))
        hidden = self.dropout(hidden)
        middle_states["middle_5"] = hidden

        hidden = self.activation3(self.linear3(self.norm3(hidden)))
        hidden = self.dropout(hidden)
        hidden = self.final_norm(hidden)
        middle_states["middle_6"] = hidden
        middle_states["last"] = hidden
        return hidden, middle_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode_with_hidden_states(x)
        return self.output_head(hidden)

    def forward_with_hidden_states(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden, hidden_states = self.encode_with_hidden_states(x)
        return self.output_head(hidden), hidden_states


class ThreeTapChannelIndependentMLP(nn.Module):
    """
    Channel-independent shallow MLP student.

    The internal temporal stack has three hidden transforms, but it exposes the
    same middle_1, middle_5, and middle_6 keys used by the full distillation
    setup so teacher alignment configs can stay unchanged.
    """

    def __init__(self, config: ThreeTapChannelIndependentMLPConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.channel_weight_mode = config.channel_weight_mode
        if int(config.num_layers) != 3:
            raise ValueError("ThreeTapChannelIndependentMLP requires num_layers=3.")
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive.")

        if config.channel_weight_mode == "shared":
            self.temporal_mlp = _ThreeTapTemporalMLP(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=config.hidden_dim,
                head_hidden_dim=config.head_hidden_dim,
                dropout=config.dropout,
                activation=config.activation,
            )
            self.channel_mlps = None
        elif config.channel_weight_mode == "separate":
            if config.num_channels is None or config.num_channels <= 0:
                raise ValueError("num_channels must be provided when channel_weight_mode='separate'.")
            self.temporal_mlp = None
            self.channel_mlps = nn.ModuleList(
                [
                    _ThreeTapTemporalMLP(
                        seq_len=self.seq_len,
                        pred_len=self.pred_len,
                        hidden_dim=config.hidden_dim,
                        head_hidden_dim=config.head_hidden_dim,
                        dropout=config.dropout,
                        activation=config.activation,
                    )
                    for _ in range(config.num_channels)
                ]
            )
        else:
            raise ValueError(f"Unsupported channel_weight_mode: {config.channel_weight_mode}")

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ThreeTapChannelIndependentMLP":
        return cls(ThreeTapChannelIndependentMLPConfig(**kwargs))

    def _validate_input(self, x: torch.Tensor) -> tuple[int, int]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, seq_len, channels], got {tuple(x.shape)}.")
        batch_size, seq_len, num_channels = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")
        return batch_size, num_channels

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_hidden(x, x_mark=x_mark, channel_mask=channel_mask)
        return output

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del x_mark
        batch_size, num_channels = self._validate_input(x)
        x_by_channel = x.transpose(1, 2).contiguous()

        if self.channel_weight_mode == "shared":
            assert self.temporal_mlp is not None
            flat_output, flat_hidden_states = self.temporal_mlp.forward_with_hidden_states(
                x_by_channel.reshape(batch_size * num_channels, self.seq_len)
            )
            forecast = flat_output.reshape(batch_size, num_channels, self.pred_len)
            hidden_states = {
                key: value.reshape(batch_size, num_channels, -1)
                for key, value in flat_hidden_states.items()
            }
        else:
            assert self.channel_mlps is not None
            if num_channels != len(self.channel_mlps):
                raise ValueError(f"Expected num_channels={len(self.channel_mlps)}, got {num_channels}.")
            channel_outputs = []
            hidden_state_lists: dict[str, list[torch.Tensor]] = {}
            for channel_idx, mlp in enumerate(self.channel_mlps):
                output, channel_hidden_states = mlp.forward_with_hidden_states(x_by_channel[:, channel_idx, :])
                channel_outputs.append(output.unsqueeze(1))
                for key, value in channel_hidden_states.items():
                    hidden_state_lists.setdefault(key, []).append(value.unsqueeze(1))
            forecast = torch.cat(channel_outputs, dim=1)
            hidden_states = {
                key: torch.cat(values, dim=1)
                for key, values in hidden_state_lists.items()
            }

        if channel_mask is not None:
            if channel_mask.shape != (batch_size, num_channels):
                raise ValueError(
                    f"channel_mask must have shape {(batch_size, num_channels)}, got {tuple(channel_mask.shape)}."
                )
            forecast = forecast * channel_mask[:, :, None].to(dtype=forecast.dtype, device=forecast.device)

        return forecast.transpose(1, 2).contiguous(), hidden_states

    def extra_repr(self) -> str:
        return (
            f"seq_len={self.seq_len}, pred_len={self.pred_len}, "
            f"num_layers=3, hidden_dim={self.config.hidden_dim}, mode={self.channel_weight_mode}"
        )


class Model(nn.Module):
    """TSL-compatible wrapper around ThreeTapChannelIndependentMLP."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        if self.task_name in ("classification", "anomaly_detection", "imputation"):
            self.pred_len = self.seq_len
        else:
            self.pred_len = int(configs.pred_len)

        self.mlp = ThreeTapChannelIndependentMLP(
            ThreeTapChannelIndependentMLPConfig(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=int(getattr(configs, "d_model", 128)),
                num_layers=3,
                head_hidden_dim=int(getattr(configs, "head_hidden_dim", getattr(configs, "d_model", 128))),
                dropout=float(getattr(configs, "dropout", 0.1)),
                activation=getattr(configs, "activation", "gelu"),
                channel_weight_mode="shared",
                num_channels=int(getattr(configs, "enc_in", 1)),
            )
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.mlp(x_enc)
