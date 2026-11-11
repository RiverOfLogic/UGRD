"""DSD LMP-Net adapter; training-only local responses share the original encoder.

Reference: jyh0526/DSD commit 050c40045a834c5b523d6aa76e42ff86f5df4709.
The unmodified reference implementation is in models/dsd_lmpnet.py.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from models.dsd_lmpnet import Model


@dataclass
class StudentLMPNetConfig:
    seq_len: int = 96
    pred_len: int = 96
    enc_in: int = 7
    d_model: int = 512
    e_layers: int = 1
    moving_avg: int = 25
    rev_in: int = 1
    dropout: float = 0.1
    patch_length: int = 16
    cross_channel_rank: int = 0
    cross_channel_gate: float = 0.1


class StudentLMPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        if min(config.seq_len, config.pred_len, config.d_model, config.e_layers, config.enc_in) < 1:
            raise ValueError("Lengths, widths, channels and depth must be positive")
        if config.moving_avg < 1:
            raise ValueError("moving_avg must be a positive kernel length")
        if config.seq_len < 2 or config.seq_len % config.patch_length:
            raise ValueError("History must contain complete non-overlapping patches and at least two steps")
        self.config = config
        self.seq_len, self.pred_len = config.seq_len, config.pred_len
        self.backbone = Model(SimpleNamespace(**vars(config)))
        self.cross_rank = int(getattr(config, "cross_channel_rank", 0))
        if self.cross_rank > 0:
            self.cross_down = nn.Linear(config.d_model, self.cross_rank, bias=False)
            self.cross_up = nn.Linear(self.cross_rank, config.d_model, bias=False)
            self.cross_mix = nn.Parameter(torch.zeros(config.enc_in, config.enc_in))
            with torch.no_grad():
                self.cross_mix.normal_(0.0, 0.01)
                self.cross_mix.fill_diagonal_(0.0)
            self.cross_gate = nn.Parameter(torch.tensor(float(config.cross_channel_gate)))

    def _cross_residual(self, latent):
        mix = self.cross_mix.masked_fill(
            torch.eye(self.config.enc_in, device=latent.device, dtype=torch.bool), 0
        )
        down = self.cross_down(latent)
        mixed = torch.einsum("bcr,cd->bdr", down, mix)
        return self.cross_gate * self.cross_up(mixed)

    def _input(self, x, channel_mask):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.config.enc_in):
            raise ValueError(f"Expected [B,{self.seq_len},{self.config.enc_in}]")
        valid = torch.ones(x.shape[0], x.shape[2], device=x.device, dtype=torch.bool)
        if channel_mask is not None:
            if channel_mask.shape != valid.shape:
                raise ValueError("channel_mask must be [B,C]")
            valid = channel_mask.to(device=x.device, dtype=torch.bool)
        if not valid.any(-1).all():
            raise ValueError("Every sample needs a valid channel")
        return x.masked_fill(~valid[:, None, :], 0), valid

    def forward_with_hidden(self, x, x_mark=None, channel_mask=None):
        if self.cross_rank > 0:
            components = self.forward_with_components(x, x_mark=x_mark, channel_mask=channel_mask)
            return components["prediction"], {"last": components["latent"]}
        x, valid = self._input(x, channel_mask)
        prediction, hidden = self.backbone(x, None, None, None, return_feature=True)
        return prediction * valid[:, None, :], {"last": hidden * valid[:, :, None]}

    def forward_with_components(self, x, x_mark=None, channel_mask=None):
        """One official-order forward exposing the actual prediction-path tensors."""
        x, valid = self._input(x, channel_mask)
        b = self.backbone
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True) + 1e-5
        if b.rev_in:
            x = (x - mean) / torch.sqrt(var)
            x = x * b.affine_weight + b.affine_bias
        seasonal, trend = b.decomposition(x)
        seasonal_input = seasonal.permute(0, 2, 1)
        trend_output = b.linear_trend(trend.permute(0, 2, 1))
        latent = b.manifold_expander(seasonal.permute(0, 2, 1))
        latent = b.act(latent)
        latent = b.dropout(latent)
        if b.latent_dynamics is not None:
            for layer in b.latent_dynamics:
                latent = layer(latent)
                latent = b.act(latent)
                latent = b.dropout(latent)
        if self.cross_rank > 0:
            latent = latent + self._cross_residual(latent)
        latent = latent * valid[:, :, None]
        seasonal_output = b.manifold_compressor(latent)
        internal = (seasonal_output + trend_output).permute(0, 2, 1)
        prediction = self.decode_internal(internal, mean, var)
        return {'prediction': prediction * valid[:, None, :], 'latent': latent,
                'seasonal': seasonal_output, 'seasonal_input': seasonal_input * valid[:, :, None],
                'trend': trend_output,
                'mean': mean, 'var': var, 'channel_mask': valid}

    def decode_internal(self, output, mean, var):
        b = self.backbone
        if b.rev_in:
            output = (output - b.affine_bias) / b.affine_weight
            output = output * torch.sqrt(var) + mean
        return output

    def local_hidden(self, x, channel_mask=None):
        """Normalize/decompose once, then encode masked seasonal inputs without dropout."""
        x, valid = self._input(x, channel_mask)
        if self.backbone.rev_in:
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True) + 1e-5  # official uses unbiased variance
            x = (x - mean) / var.sqrt()
            x = x * self.backbone.affine_weight + self.backbone.affine_bias
        seasonal, _ = self.backbone.decomposition(x)
        seasonal = seasonal.transpose(1, 2)
        p = self.seq_len // self.config.patch_length
        masks = torch.eye(p, device=x.device, dtype=x.dtype).repeat_interleave(self.config.patch_length, -1)
        local = seasonal[:, None, :, :] * masks[None, :, None, :]
        hidden = self.backbone.act(self.backbone.manifold_expander(local))
        if self.backbone.latent_dynamics is not None:
            for layer in self.backbone.latent_dynamics:
                hidden = self.backbone.act(layer(hidden))
        return hidden * valid[:, None, :, None]

    def forward_with_local_hidden(self, x, x_mark=None, channel_mask=None):
        prediction, states = self.forward_with_hidden(x, x_mark, channel_mask)
        return prediction, states, self.local_hidden(x, channel_mask)

    def forward(self, x, x_mark=None, channel_mask=None):
        return self.forward_with_hidden(x, x_mark, channel_mask)[0]
