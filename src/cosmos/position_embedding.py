# Positional embeddings for Cosmos DiT.
# Extracted from cosmos_predict1/diffusion/module/position_embedding.py
# and cosmos_predict1/diffusion/training/module/position_embedding.py

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from .attention import normalize


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_B_T_H_W_C, fps=None):
        return self.generate_embeddings(x_B_T_H_W_C.shape, fps=fps)

    def generate_embeddings(self, B_T_H_W_C, fps=None):
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,
        head_dim,
        len_h,
        len_w,
        len_t,
        base_fps=24,
        h_extrapolation_ratio=1.0,
        w_extrapolation_ratio=1.0,
        t_extrapolation_ratio=1.0,
        **kwargs,
    ):
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def generate_embeddings(self, B_T_H_W_C, fps=None):
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C

        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if fps is None:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ] * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class SinCosPosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,
        interpolation,
        model_channels,
        len_h,
        len_w,
        len_t,
        h_extrapolation_ratio=1.0,
        w_extrapolation_ratio=1.0,
        t_extrapolation_ratio=1.0,
        **kwargs,
    ):
        super().__init__()
        self.interpolation = interpolation

        dim = model_channels
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h

        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos=np.arange(len_h) * 1.0 / h_extrapolation_ratio)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos=np.arange(len_w) * 1.0 / w_extrapolation_ratio)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, pos=np.arange(len_t) * 1.0 / t_extrapolation_ratio)

        self.register_buffer("pos_emb_h", torch.from_numpy(emb_h).float(), persistent=False)
        self.register_buffer("pos_emb_w", torch.from_numpy(emb_w).float(), persistent=False)
        self.register_buffer("pos_emb_t", torch.from_numpy(emb_t).float(), persistent=False)

    def generate_embeddings(self, B_T_H_W_C, fps=None):
        B, T, H, W, C = B_T_H_W_C
        emb_h_H = self.pos_emb_h[:H]
        emb_w_W = self.pos_emb_w[:W]
        emb_t_T = self.pos_emb_t[:T]
        emb = torch.cat(
            [
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W),
                repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W),
                repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H),
            ],
            dim=-1,
        )
        return emb
