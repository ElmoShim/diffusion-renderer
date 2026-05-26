# Transformer building blocks for Cosmos DiT.
# Extracted from cosmos_predict1/diffusion/module/blocks.py

import math
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from .attention import Attention, GPT2FeedForward


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        in_dtype = timesteps.dtype
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)
        return emb.to(in_dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features, use_adaln_lora=False):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        if self.use_adaln_lora:
            adaln_lora_B_3D = emb
            emb_B_D = sample
        else:
            emb_B_D = emb
            adaln_lora_B_3D = None
        return emb_B_D, adaln_lora_B_3D


class FourierFeatures(nn.Module):
    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1

    def forward(self, x, gain=1.0):
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, temporal_patch_size, in_channels=3, out_channels=768, bias=True):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=bias
            ),
        )
        self.out = nn.Identity()

    def forward(self, x):
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels,
                 use_adaln_lora=False, adaln_lora_dim=256):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

    def forward(self, x_BT_HW_D, emb_B_D, adaln_lora_B_3D=None):
        if self.use_adaln_lora:
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T)
        scale_BT_D = repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)
        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D


class VideoAttn(nn.Module):
    def __init__(self, x_dim, context_dim, num_heads, bias=False, qkv_norm_mode="per_head",
                 x_format="BTHWD", n_views=1):
        super().__init__()
        self.x_format = x_format
        self.n_views = n_views
        self.attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads,
            qkv_bias=bias, qkv_norm="RRI", out_bias=bias,
            qkv_norm_mode=qkv_norm_mode, qkv_format="sbhd",
        )

    def forward(self, x, context=None, crossattn_mask=None, rope_emb_L_1_1_D=None):
        x_T_H_W_B_D = x
        T, H, W, B, D = x_T_H_W_B_D.shape
        x_THW_B_D = rearrange(x_T_H_W_B_D, "t h w b d -> (t h w) b d")
        x_THW_B_D = self.attn(x_THW_B_D, context, crossattn_mask, rope_emb=rope_emb_L_1_1_D)
        x_T_H_W_B_D = rearrange(x_THW_B_D, "(t h w) b d -> t h w b d", h=H, w=W)
        return x_T_H_W_B_D


def adaln_norm_state(norm_state, x, scale, shift):
    return norm_state(x) * (1 + scale) + shift


class DITBuildingBlock(nn.Module):
    def __init__(self, block_type, x_dim, context_dim, num_heads, mlp_ratio=4.0, bias=False,
                 mlp_dropout=0.0, qkv_norm_mode="per_head", x_format="BTHWD",
                 use_adaln_lora=False, adaln_lora_dim=256, n_views=1):
        block_type = block_type.lower()
        super().__init__()
        self.x_format = x_format
        if block_type in ["cross_attn", "ca"]:
            self.block = VideoAttn(x_dim, context_dim, num_heads, bias=bias,
                                   qkv_norm_mode=qkv_norm_mode, x_format=self.x_format, n_views=n_views)
        elif block_type in ["full_attn", "fa"]:
            self.block = VideoAttn(x_dim, None, num_heads, bias=bias,
                                   qkv_norm_mode=qkv_norm_mode, x_format=self.x_format)
        elif block_type in ["mlp", "ff"]:
            self.block = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), dropout=mlp_dropout, bias=bias)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.block_type = block_type
        self.use_adaln_lora = use_adaln_lora
        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = 3
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, self.n_adaln_chunks * x_dim, bias=False)
            )

    def forward(self, x, emb_B_D, crossattn_emb, crossattn_mask=None, rope_emb_L_1_1_D=None,
                adaln_lora_B_3D=None):
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift = shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scale = scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        gate = gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if self.block_type in ["mlp", "ff"]:
            x = x + gate * self.block(adaln_norm_state(self.norm_state, x, scale, shift))
        elif self.block_type in ["full_attn", "fa"]:
            x = x + gate * self.block(
                adaln_norm_state(self.norm_state, x, scale, shift),
                context=None, rope_emb_L_1_1_D=rope_emb_L_1_1_D)
        elif self.block_type in ["cross_attn", "ca"]:
            x = x + gate * self.block(
                adaln_norm_state(self.norm_state, x, scale, shift),
                context=crossattn_emb, crossattn_mask=crossattn_mask, rope_emb_L_1_1_D=rope_emb_L_1_1_D)
        return x


class GeneralDITTransformerBlock(nn.Module):
    def __init__(self, x_dim, context_dim, num_heads, block_config, mlp_ratio=4.0,
                 x_format="BTHWD", use_adaln_lora=False, adaln_lora_dim=256, n_views=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.x_format = x_format
        for block_type in block_config.split("-"):
            self.blocks.append(
                DITBuildingBlock(
                    block_type, x_dim, context_dim, num_heads, mlp_ratio,
                    x_format=self.x_format, use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim, n_views=n_views,
                )
            )

    def forward(self, x, emb_B_D, crossattn_emb, crossattn_mask=None, rope_emb_L_1_1_D=None,
                adaln_lora_B_3D=None, extra_per_block_pos_emb=None):
        if extra_per_block_pos_emb is not None:
            x = x + extra_per_block_pos_emb.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, emb_B_D, crossattn_emb, crossattn_mask,
                      rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_3D=adaln_lora_B_3D)
        return x
