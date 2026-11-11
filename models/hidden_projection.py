from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class HiddenProjectionConfig:
    teacher_dim: int
    student_dim: int
    method: Literal["linear", "mlp", "identity", "memory_cross_attention"] = "linear"
    hidden_dim: int | None = None
    activation: Literal["gelu", "relu", "silu", "tanh"] = "gelu"
    dropout: float = 0.0
    memory_tokens: int = 4
    num_heads: int = 8
    max_future_tokens: int = 128


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


class HiddenStateProjector(nn.Module):
    """Project hidden states for teacher-student hidden distillation.

    The classic methods map teacher hidden states [B, C, teacher_dim] to
    student hidden states [B, C, student_dim]. The memory_cross_attention method
    maps student hidden states [B, C, student_dim] to teacher future-token space
    [B, C, num_future_tokens, teacher_dim].
    """

    def __init__(self, config: HiddenProjectionConfig) -> None:
        super().__init__()
        self.config = config
        teacher_dim = int(config.teacher_dim)
        student_dim = int(config.student_dim)
        if teacher_dim <= 0 or student_dim <= 0:
            raise ValueError("teacher_dim and student_dim must be positive.")

        self.net: nn.Module | None
        if config.method == "identity":
            if teacher_dim != student_dim:
                raise ValueError("identity projection requires teacher_dim == student_dim.")
            self.net = nn.Identity()
        elif config.method == "linear":
            self.net = nn.Linear(teacher_dim, student_dim)
        elif config.method == "mlp":
            hidden_dim = int(config.hidden_dim or max(teacher_dim, student_dim))
            self.net = nn.Sequential(
                nn.Linear(teacher_dim, hidden_dim),
                _activation(config.activation),
                nn.Dropout(float(config.dropout)),
                nn.Linear(hidden_dim, student_dim),
            )
        elif config.method == "memory_cross_attention":
            memory_tokens = int(config.memory_tokens)
            num_heads = int(config.num_heads)
            max_future_tokens = int(config.max_future_tokens)
            if memory_tokens <= 0:
                raise ValueError("memory_tokens must be positive.")
            if num_heads <= 0:
                raise ValueError("num_heads must be positive.")
            if teacher_dim % num_heads != 0:
                raise ValueError("teacher_dim must be divisible by num_heads for memory_cross_attention.")
            if max_future_tokens <= 0:
                raise ValueError("max_future_tokens must be positive.")
            self.net = None
            self.memory_tokens = memory_tokens
            self.max_future_tokens = max_future_tokens
            self.memory_proj = nn.Linear(student_dim, memory_tokens * teacher_dim)
            self.query_tokens = nn.Parameter(torch.empty(1, max_future_tokens, teacher_dim))
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=teacher_dim,
                num_heads=num_heads,
                dropout=float(config.dropout),
                batch_first=True,
            )
            self.output_norm = nn.LayerNorm(teacher_dim)
            nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unsupported projection method: {config.method}")

    def forward(self, hidden: torch.Tensor, *, num_future_tokens: int | None = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"hidden must have shape [batch, channels, dim], got {tuple(hidden.shape)}.")
        if self.config.method == "memory_cross_attention":
            return self._forward_memory_cross_attention(hidden, num_future_tokens=num_future_tokens)

        if hidden.shape[-1] != int(self.config.teacher_dim):
            raise ValueError(f"Expected teacher_dim={self.config.teacher_dim}, got {hidden.shape[-1]}.")
        assert self.net is not None
        return self.net(hidden)

    def _forward_memory_cross_attention(
        self,
        student_hidden: torch.Tensor,
        *,
        num_future_tokens: int | None,
    ) -> torch.Tensor:
        if num_future_tokens is None:
            raise ValueError("num_future_tokens is required for memory_cross_attention.")
        num_future_tokens = int(num_future_tokens)
        if num_future_tokens <= 0:
            raise ValueError("num_future_tokens must be positive.")
        if num_future_tokens > self.max_future_tokens:
            raise ValueError(
                f"num_future_tokens={num_future_tokens} exceeds max_future_tokens={self.max_future_tokens}."
            )
        if student_hidden.shape[-1] != int(self.config.student_dim):
            raise ValueError(f"Expected student_dim={self.config.student_dim}, got {student_hidden.shape[-1]}.")

        batch_size, num_channels, _ = student_hidden.shape
        teacher_dim = int(self.config.teacher_dim)
        memory = self.memory_proj(student_hidden)
        memory = memory.reshape(batch_size * num_channels, self.memory_tokens, teacher_dim)
        query = self.query_tokens[:, :num_future_tokens, :].expand(batch_size * num_channels, -1, -1)
        attended, _ = self.cross_attention(query=query, key=memory, value=memory, need_weights=False)
        attended = self.output_norm(attended)
        return attended.reshape(batch_size, num_channels, num_future_tokens, teacher_dim)
