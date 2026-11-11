from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class ChannelIndependentMLPConfig:
    seq_len: int
    pred_len: int
    hidden_dim: int = 512
    num_layers: int = 6
    head_hidden_dim: int | None = None
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu", "tanh"] = "gelu"
    channel_weight_mode: Literal["shared", "separate"] = "shared"
    num_channels: int | None = None


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class _ResidualGroup(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation1 = _activation(activation)
        self.activation2 = _activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.forward_with_middle(x, start_index=1)
        return hidden

    def forward_with_middle(self, x: torch.Tensor, start_index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = x
        hidden = self.norm1(x)
        hidden = self.linear1(hidden)
        hidden = self.activation1(hidden)
        middle_states = {f"middle_{start_index}": hidden}
        hidden = self.dropout(hidden)
        hidden = self.norm2(hidden)
        hidden = self.linear2(hidden)
        hidden = self.activation2(hidden)
        middle_states[f"middle_{start_index + 1}"] = hidden
        hidden = self.dropout(hidden)
        return residual + hidden, middle_states

    def forward_with_middle_and_injection(
        self,
        x: torch.Tensor,
        *,
        start_index: int,
        batch_size: int,
        num_channels: int,
        relation_matrix: torch.Tensor,
        injector: nn.Module,
        injection_layer: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        def maybe_inject(hidden: torch.Tensor, key: str) -> torch.Tensor:
            if key != injection_layer:
                return hidden
            shaped = hidden.reshape(batch_size, num_channels, -1)
            injected = injector(shaped, relation_matrix)
            return injected.reshape(batch_size * num_channels, -1)

        residual = x
        hidden = self.norm1(x)
        hidden = self.linear1(hidden)
        hidden = self.activation1(hidden)
        key = f"middle_{start_index}"
        hidden = maybe_inject(hidden, key)
        middle_states = {key: hidden}
        hidden = self.dropout(hidden)
        hidden = self.norm2(hidden)
        hidden = self.linear2(hidden)
        hidden = self.activation2(hidden)
        key = f"middle_{start_index + 1}"
        hidden = maybe_inject(hidden, key)
        middle_states[key] = hidden
        hidden = self.dropout(hidden)
        return residual + hidden, middle_states


class _ResidualTemporalMLP(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden_dim: int,
        num_layers: int,
        head_hidden_dim: int | None,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or num_layers % 2 != 0:
            raise ValueError("num_layers must be a positive even number; it represents hidden layers.")
        head_hidden_dim = int(head_hidden_dim or hidden_dim)
        if head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive.")

        self.input_proj = nn.Linear(seq_len, hidden_dim)
        self.residual_groups = nn.ModuleList(
            [_ResidualGroup(hidden_dim, dropout, activation) for _ in range(num_layers // 2)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, pred_len),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode_with_hidden_states(x)
        return hidden

    def encode_with_hidden_states(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.input_proj(x)
        middle_states: dict[str, torch.Tensor] = {}
        for group_idx, group in enumerate(self.residual_groups):
            hidden, group_states = group.forward_with_middle(hidden, start_index=group_idx * 2 + 1)
            middle_states.update(group_states)
        hidden = self.final_norm(hidden)
        middle_states["last"] = hidden
        return hidden, middle_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(x)
        return self.output_head(hidden)

    def forward_with_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.encode_with_hidden_states(x)
        output = self.output_head(hidden)
        return output, hidden

    def forward_with_hidden_states(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden, hidden_states = self.encode_with_hidden_states(x)
        output = self.output_head(hidden)
        return output, hidden_states

    def forward_with_relation_injection(
        self,
        x: torch.Tensor,
        *,
        batch_size: int,
        num_channels: int,
        relation_matrix: torch.Tensor,
        injector: nn.Module,
        injection_layer: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.input_proj(x)
        middle_states: dict[str, torch.Tensor] = {}
        for group_idx, group in enumerate(self.residual_groups):
            hidden, group_states = group.forward_with_middle_and_injection(
                hidden,
                start_index=group_idx * 2 + 1,
                batch_size=batch_size,
                num_channels=num_channels,
                relation_matrix=relation_matrix,
                injector=injector,
                injection_layer=injection_layer,
            )
            middle_states.update(group_states)
        hidden = self.final_norm(hidden)
        if injection_layer == "last":
            shaped = hidden.reshape(batch_size, num_channels, -1)
            hidden = injector(shaped, relation_matrix).reshape(batch_size * num_channels, -1)
        middle_states["last"] = hidden
        output = self.output_head(hidden)
        return output, middle_states


class ChannelIndependentMLP(nn.Module):
    """
    A channel-independent MLP student for forecasting.

    Input shape:
        x: [batch, seq_len, channels]

    Output shape:
        forecast: [batch, pred_len, channels]

    The model never mixes information across channels. In `shared` mode, the
    same 6-layer temporal MLP is applied to every channel. In `separate` mode,
    each channel gets its own 6-layer temporal MLP, so `num_channels` must be
    provided and fixed.
    """

    def __init__(self, config: ChannelIndependentMLPConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.channel_weight_mode = config.channel_weight_mode

        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive.")

        if config.channel_weight_mode == "shared":
            self.temporal_mlp = _ResidualTemporalMLP(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
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
                    _ResidualTemporalMLP(
                        seq_len=self.seq_len,
                        pred_len=self.pred_len,
                        hidden_dim=config.hidden_dim,
                        num_layers=config.num_layers,
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
    def from_kwargs(cls, **kwargs) -> "ChannelIndependentMLP":
        return cls(ChannelIndependentMLPConfig(**kwargs))

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, seq_len, channels], got {tuple(x.shape)}.")
        batch_size, seq_len, num_channels = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")

        # [B, L, C] -> [B, C, L], so each channel is a separate temporal sample.
        x_by_channel = x.transpose(1, 2).contiguous()

        if self.channel_weight_mode == "shared":
            assert self.temporal_mlp is not None
            forecast = self.temporal_mlp(x_by_channel.reshape(batch_size * num_channels, self.seq_len))
            forecast = forecast.reshape(batch_size, num_channels, self.pred_len)
        else:
            assert self.channel_mlps is not None
            if num_channels != len(self.channel_mlps):
                raise ValueError(f"Expected num_channels={len(self.channel_mlps)}, got {num_channels}.")
            channel_outputs = [
                mlp(x_by_channel[:, channel_idx, :]).unsqueeze(1)
                for channel_idx, mlp in enumerate(self.channel_mlps)
            ]
            forecast = torch.cat(channel_outputs, dim=1)

        if channel_mask is not None:
            if channel_mask.shape != (batch_size, num_channels):
                raise ValueError(
                    f"channel_mask must have shape {(batch_size, num_channels)}, got {tuple(channel_mask.shape)}."
                )
            forecast = forecast * channel_mask[:, :, None].to(dtype=forecast.dtype, device=forecast.device)

        return forecast.transpose(1, 2).contiguous()

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del x_mark
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, seq_len, channels], got {tuple(x.shape)}.")
        batch_size, seq_len, num_channels = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")

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

    def forward_with_relation_injection(
        self,
        x: torch.Tensor,
        relation_matrix: torch.Tensor,
        injector: nn.Module,
        *,
        injection_layer: str = "middle_6",
        x_mark: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del x_mark
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, seq_len, channels], got {tuple(x.shape)}.")
        batch_size, seq_len, num_channels = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")
        if relation_matrix.shape != (batch_size, num_channels, num_channels):
            raise ValueError(
                "relation_matrix must have shape [batch, channels, channels], got "
                f"{tuple(relation_matrix.shape)} for batch={batch_size}, channels={num_channels}."
            )
        if self.channel_weight_mode != "shared":
            raise ValueError("Relation injection currently requires channel_weight_mode='shared'.")
        assert self.temporal_mlp is not None

        x_by_channel = x.transpose(1, 2).contiguous()
        flat_output, flat_hidden_states = self.temporal_mlp.forward_with_relation_injection(
            x_by_channel.reshape(batch_size * num_channels, self.seq_len),
            batch_size=batch_size,
            num_channels=num_channels,
            relation_matrix=relation_matrix,
            injector=injector,
            injection_layer=injection_layer,
        )
        forecast = flat_output.reshape(batch_size, num_channels, self.pred_len)
        hidden_states = {
            key: value.reshape(batch_size, num_channels, -1)
            for key, value in flat_hidden_states.items()
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
            f"num_layers={self.config.num_layers}, hidden_dim={self.config.hidden_dim}, "
            f"mode={self.channel_weight_mode}"
        )


class Model(nn.Module):
    """TSL-compatible wrapper around the DTS channel-independent MLP."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        if self.task_name in ("classification", "anomaly_detection", "imputation"):
            self.pred_len = self.seq_len
        else:
            self.pred_len = int(configs.pred_len)

        hidden_dim = int(getattr(configs, "d_model", 512))
        num_layers = int(getattr(configs, "e_layers", 2)) * 3
        if num_layers % 2 != 0:
            num_layers += 1
        dropout = float(getattr(configs, "dropout", 0.1))
        activation = getattr(configs, "activation", "gelu")

        self.mlp = ChannelIndependentMLP(
            ChannelIndependentMLPConfig(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
                channel_weight_mode="shared",
                num_channels=int(getattr(configs, "enc_in", 1)),
            )
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.mlp(x_enc)
