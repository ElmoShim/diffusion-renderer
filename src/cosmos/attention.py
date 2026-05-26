# Attention and feed-forward modules for Cosmos DiT.
# Extracted from cosmos_predict1/diffusion/module/attention.py
# Replaces transformer_engine dependency with native PyTorch SDPA.

from typing import List, Optional

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation=nn.ReLU(), is_gated=False, bias=False):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        return self.layer2(x)


class GPT2FeedForward(FeedForward):
    def __init__(self, d_model, d_ff, dropout=0.1, bias=False):
        super().__init__(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=nn.GELU(), is_gated=False, bias=bias)

    def forward(self, x):
        x = self.layer1(x)

        def activation_layer2_forward(x):
            x = self.activation(x)
            x = self.layer2(x)
            return x

        x = checkpoint(activation_layer2_forward, x, use_reentrant=False)
        return x


def normalize(x, dim=None, eps=0):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def get_normalization(name, channels):
    if name == "I":
        return nn.Identity()
    elif name == "R":
        return nn.RMSNorm(channels, eps=1e-6)
    else:
        raise ValueError(f"Normalization {name} not found")


def apply_rotary_pos_emb_native(t, freqs):
    """Apply rotary position embeddings using native PyTorch.

    Args:
        t: tensor of shape (..., seq_len, num_heads, head_dim)
        freqs: tensor of shape (seq_len, 1, 1, head_dim) where head_dim contains
               [cos_part, sin_part] concatenated (each half_dim)
    """
    rot_dim = freqs.shape[-1]
    half_dim = rot_dim // 2
    cos_freqs = torch.cos(freqs[..., :half_dim])
    sin_freqs = torch.sin(freqs[..., :half_dim])

    t_rot = t[..., :half_dim]
    t_pass = t[..., half_dim:]

    # Rotary embedding: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    t_rot_out = t_rot * cos_freqs + torch.cat([-t_rot[..., half_dim // 2:], t_rot[..., :half_dim // 2]], dim=-1) * sin_freqs

    # Actually the Cosmos format stores freqs as [half_emb_t, half_emb_h, half_emb_w] * 2
    # The first half are the positions, second half is duplicate for cos/sin
    # Let's use a simpler approach matching the TE convention
    return torch.cat([t_rot_out, t_pass], dim=-1)


def apply_rotary_pos_emb_cosmos(x, rope_emb):
    """Apply RoPE to input tensor, matching TransformerEngine `apply_rotary_pos_emb`.

    x: (seq_len, batch, num_heads, head_dim) format "sbhd"
    rope_emb: (seq_len, 1, 1, head_dim) where the second half of the last dim is
        a duplicate of the first (rope_emb = cat([freqs, freqs])).
    """
    orig_dtype = x.dtype
    cos_emb = torch.cos(rope_emb)  # (sq, 1, 1, head_dim)
    sin_emb = torch.sin(rope_emb)
    d = x.shape[-1]
    half = d // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos_emb + x_rot * sin_emb).to(orig_dtype)


class Attention(nn.Module):
    """Attention module using native PyTorch SDPA instead of TransformerEngine."""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_op=None,
        qkv_bias=False,
        out_bias=False,
        qkv_norm="SSI",
        qkv_norm_mode="per_head",
        backend="torch",
        qkv_format="sbhd",
    ):
        super().__init__()
        self.is_selfattn = context_dim is None
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format

        norm_dim = dim_head

        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[0], norm_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[1], norm_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[2], norm_dim),
        )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

    def cal_qkv(self, x, context=None, mask=None, rope_emb=None, **kwargs):
        q = self.to_q[0](x)
        context = x if context is None else context
        k = self.to_k[0](context)
        v = self.to_v[0](context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads, c=self.dim_head),
            (q, k, v),
        )
        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)

        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb_cosmos(q, rope_emb)
            k = apply_rotary_pos_emb_cosmos(k, rope_emb)
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        # Input format is "sbhd": (seq_len, batch, heads, dim_head)
        # Convert to (batch, heads, seq_len, dim_head) for SDPA
        q = rearrange(q, "s b h d -> b h s d")
        k = rearrange(k, "s b h d -> b h s d")
        v = rearrange(v, "s b h d -> b h s d")
        v = v.to(dtype=q.dtype)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.to_out(rearrange(out, "b h s d -> s b (h d)"))

    def forward(self, x, context=None, mask=None, rope_emb=None, **kwargs):
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        return self.cal_attn(q, k, v, mask)
