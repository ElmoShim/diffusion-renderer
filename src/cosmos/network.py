# FADiT v2 Transformer network for Cosmos Diffusion Renderer.
# Extracted from cosmos_predict1/diffusion/networks/general_dit.py
# and cosmos_predict1/diffusion/networks/general_dit_diffusion_renderer.py

from typing import List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torchvision import transforms

from .attention import get_normalization
from .blocks import (
    FinalLayer,
    GeneralDITTransformerBlock,
    PatchEmbed,
    TimestepEmbedding,
    Timesteps,
)
from .position_embedding import VideoRopePosition3DEmb, SinCosPosEmbAxis


class GeneralDIT(nn.Module):
    def __init__(
        self,
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        block_config="FA-CA-MLP",
        model_channels=4096,
        num_blocks=28,
        num_heads=32,
        mlp_ratio=4.0,
        block_x_format="BTHWD",
        crossattn_emb_channels=1024,
        use_cross_attn_mask=False,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        pos_emb_interpolation="crop",
        affline_emb_norm=False,
        use_adaln_lora=False,
        adaln_lora_dim=256,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False,
        extra_per_block_abs_pos_emb_type="sincos",
        extra_h_extrapolation_ratio=1.0,
        extra_w_extrapolation_ratio=1.0,
        extra_t_extrapolation_ratio=1.0,
    ):
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.use_cross_attn_mask = use_cross_attn_mask
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.affline_emb_norm = affline_emb_norm
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_per_block_abs_pos_emb_type = extra_per_block_abs_pos_emb_type.lower()
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio

        self.build_patch_embed()
        self.build_pos_embed()
        self.cp_group = None
        self.block_x_format = block_x_format
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        self.blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            self.blocks[f"block{idx}"] = GeneralDITTransformerBlock(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                block_config=block_config,
                mlp_ratio=mlp_ratio,
                x_format=self.block_x_format,
                use_adaln_lora=use_adaln_lora,
                adaln_lora_dim=adaln_lora_dim,
            )

        self.build_decode_head()
        if self.affline_emb_norm:
            self.affline_norm = get_normalization("R", model_channels)
        else:
            self.affline_norm = nn.Identity()

    @property
    def is_context_parallel_enabled(self):
        return False

    def build_decode_head(self):
        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

    def build_patch_embed(self):
        in_channels = self.in_channels
        in_channels = in_channels + 1 if self.concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
            bias=False,
        )

    def build_pos_embed(self):
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
        )
        self.pos_embedder = VideoRopePosition3DEmb(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio

    def prepare_embedded_sequence(self, x_B_C_T_H_W, fps=None, padding_mask=None,
                                  latent_condition=None, latent_condition_sigma=None):
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb and hasattr(self, 'extra_pos_embedder'):
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x,
        timesteps,
        crossattn_emb,
        crossattn_mask=None,
        fps=None,
        image_size=None,
        padding_mask=None,
        scalar_feature=None,
        data_type=None,
        latent_condition=None,
        latent_condition_sigma=None,
        condition_video_augment_sigma=None,
        **kwargs,
    ):
        B = x.shape[0]
        timesteps_B = timesteps
        if timesteps_B.dim() == 0:
            timesteps_B = timesteps_B.unsqueeze(0).expand(B)

        x_B_T_H_W_D, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(
            x, fps=fps, padding_mask=padding_mask,
            latent_condition=latent_condition, latent_condition_sigma=latent_condition_sigma,
        )

        emb_B_D, adaln_lora_B_3D = self.t_embedder(timesteps_B)
        emb_B_D = self.affline_norm(emb_B_D)

        if self.block_x_format == "THWBD":
            x_T_H_W_B_D = rearrange(x_B_T_H_W_D, "b t h w d -> t h w b d")
            if extra_pos_emb is not None:
                extra_pos_emb = rearrange(extra_pos_emb, "b t h w d -> t h w b d")
        else:
            raise ValueError(f"Unknown block_x_format: {self.block_x_format}")

        if crossattn_emb is not None and crossattn_emb.dim() == 3:
            crossattn_emb = rearrange(crossattn_emb, "b m d -> m b d")

        for _, block in self.blocks.items():
            x_T_H_W_B_D = block(
                x_T_H_W_B_D, emb_B_D, crossattn_emb, crossattn_mask,
                rope_emb_L_1_1_D=rope_emb, adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )

        x_BT_HW_D = rearrange(x_T_H_W_B_D, "t h w b d -> (b t) (h w) d")
        x_BT_HW_D = self.final_layer(x_BT_HW_D, emb_B_D, adaln_lora_B_3D=adaln_lora_B_3D)

        T = x_B_T_H_W_D.shape[1]
        H = x_B_T_H_W_D.shape[2]
        W = x_B_T_H_W_D.shape[3]
        x_B_C_T_H_W = rearrange(
            x_BT_HW_D,
            "(b t) (h w) (ph pw pt c) -> b c (t pt) (h ph) (w pw)",
            b=B, t=T, h=H, w=W,
            ph=self.patch_spatial, pw=self.patch_spatial, pt=self.patch_temporal,
        )
        return x_B_C_T_H_W


class DiffusionRendererDiT(GeneralDIT):
    """FADiT for diffusion renderer with latent condition concatenation and optional context embedding."""

    def __init__(self, *args, additional_concat_ch=None, use_context_embedding=True, **kwargs):
        self.additional_concat_ch = additional_concat_ch
        self.use_context_embedding = use_context_embedding
        super().__init__(*args, **kwargs)

        if self.use_context_embedding:
            self.context_embedding = torch.nn.Embedding(
                num_embeddings=16, embedding_dim=kwargs.get("crossattn_emb_channels", 1024)
            )
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            torch.nn.init.uniform_(self.context_embedding.weight, -0.3, 0.3)
            torch.set_rng_state(rng_state)

    def build_pos_embed(self):
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
        )
        self.pos_embedder = VideoRopePosition3DEmb(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = SinCosPosEmbAxis(**kwargs)

    def build_patch_embed(self):
        in_channels = self.in_channels
        if self.additional_concat_ch is None:
            self.additional_concat_ch = in_channels
        in_channels = in_channels + self.additional_concat_ch
        in_channels = in_channels + 1 if self.concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
            bias=False,
        )

    def prepare_embedded_sequence(self, x_B_C_T_H_W, fps=None, padding_mask=None,
                                  latent_condition=None, latent_condition_sigma=None):
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, latent_condition], dim=1)
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb and hasattr(self, 'extra_pos_embedder'):
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(self, x, timesteps, crossattn_emb, crossattn_mask=None, fps=None,
                image_size=None, padding_mask=None, scalar_feature=None, data_type=None,
                latent_condition=None, latent_condition_sigma=None,
                condition_video_augment_sigma=None, context_index=None, **kwargs):
        if self.use_context_embedding and context_index is not None:
            input_context_emb = self.context_embedding(context_index.long())
            if input_context_emb.ndim == 2:
                input_context_emb = input_context_emb.unsqueeze(1).clone()
            input_context_emb = input_context_emb.repeat_interleave(crossattn_emb.shape[1], dim=1)
            input_context_emb = input_context_emb.to(device=crossattn_emb.device, dtype=crossattn_emb.dtype)
            crossattn_emb = input_context_emb

        return super().forward(
            x=x, timesteps=timesteps, crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask, fps=fps, image_size=image_size,
            padding_mask=padding_mask, scalar_feature=scalar_feature,
            data_type=data_type, latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )
