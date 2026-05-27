# Cosmos Forward Renderer model — builds network, tokenizer, and runs diffusion sampling.
# Consolidated from model_diffusion_renderer.py + model_t2w.py + inference pipeline.

import os
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import EDMEulerScheduler
from torch import Tensor

from .conditioner import VideoDiffusionRendererCondition
from .network import DiffusionRendererDiT
from .tokenizer import CosmosTokenizer


@contextmanager
def skip_init_linear():
    orig = torch.nn.Linear.reset_parameters
    orig_xu = torch.nn.init.xavier_uniform_
    torch.nn.Linear.reset_parameters = lambda x: x
    torch.nn.init.xavier_uniform_ = lambda x: x
    yield
    torch.nn.Linear.reset_parameters = orig
    torch.nn.init.xavier_uniform_ = orig_xu


def non_strict_load_model(model: torch.nn.Module, state_dict: dict):
    # Strip "net." prefix if present (official checkpoints store keys as "net.x_embedder...").
    if any(k.startswith("net.") for k in state_dict.keys()):
        state_dict = {k[len("net."):] if k.startswith("net.") else k: v
                      for k, v in state_dict.items()}

    model_sd = model.state_dict()
    incorrect = []
    for k in list(state_dict.keys()):
        if k in model_sd:
            if "_extra_state" in k:
                continue
            if tuple(model_sd[k].shape) != tuple(state_dict[k].shape):
                incorrect.append((k, tuple(state_dict[k].shape), tuple(model_sd[k].shape)))
                state_dict.pop(k)
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if "_extra_state" not in k]
    unexpected = [k for k in result.unexpected_keys if "_extra_state" not in k]
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    if incorrect:
        print(f"  Incorrect shapes: {len(incorrect)}")


class CosmosRendererBase(torch.nn.Module):
    """Shared base for Cosmos forward/inverse diffusion renderers."""

    use_context_embedding: bool = False

    def __init__(
        self,
        condition_keys: List[str],
        condition_drop_rate: float = 0.05,
        append_condition_mask: bool = True,
        num_frames: int = 57,
        height: int = 704,
        width: int = 1280,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.condition_keys = condition_keys
        self.condition_drop_rate = condition_drop_rate
        self.append_condition_mask = append_condition_mask
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.sigma_data = sigma_data
        self.dtype = torch.bfloat16
        self.tensor_kwargs = {"device": "cuda", "dtype": self.dtype}

        n_cond = len(condition_keys)
        ch_per_cond = 16 + (1 if append_condition_mask else 0)
        self.additional_concat_ch = ch_per_cond * n_cond

        self.tokenizer = CosmosTokenizer(pixel_chunk_duration=num_frames)

        self.net = None

        self.scheduler = EDMEulerScheduler(
            sigma_max=80, sigma_min=0.02, sigma_data=sigma_data,
        )

    def build_network(self):
        with skip_init_linear():
            self.net = DiffusionRendererDiT(
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
                block_x_format="THWBD",
                crossattn_emb_channels=1024,
                use_cross_attn_mask=False,
                pos_emb_cls="rope3d",
                pos_emb_learnable=False,
                pos_emb_interpolation="crop",
                affline_emb_norm=True,
                use_adaln_lora=True,
                adaln_lora_dim=256,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=1.0,
                extra_per_block_abs_pos_emb=True,
                extra_per_block_abs_pos_emb_type="sincos",
                additional_concat_ch=self.additional_concat_ch,
                use_context_embedding=self.use_context_embedding,
            )

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path} ...")
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        non_strict_load_model(self.net, state_dict)

    def load_tokenizer(self, tokenizer_dir: str):
        print(f"Loading tokenizer from {tokenizer_dir} ...")
        self.tokenizer.load_weights(tokenizer_dir)

    def to_device(self):
        self.net.to(**self.tensor_kwargs)
        self.tokenizer.to("cuda")

    @torch.no_grad()
    def encode(self, state: Tensor) -> Tensor:
        return self.tokenizer.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        return self.tokenizer.decode(latent / self.sigma_data)

    @torch.no_grad()
    def prepare_latent_conditions(
        self,
        data_batch: dict,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Tensor:
        if dtype is None:
            dtype = data_batch[self.condition_keys[0]].dtype
        if device is None:
            device = data_batch[self.condition_keys[0]].device

        B, C, T, H, W = data_batch[self.condition_keys[0]].shape
        latent_shape = (B, 16, T // 8 + 1, H // 8, W // 8)
        latent_mask_shape = (latent_shape[0], 1, latent_shape[2], latent_shape[3], latent_shape[4])

        parts = []
        for key in self.condition_keys:
            if key not in data_batch:
                parts.append(torch.zeros(latent_shape, dtype=dtype, device=device))
                if self.append_condition_mask:
                    parts.append(torch.zeros(latent_mask_shape, dtype=dtype, device=device))
            else:
                cond = data_batch[key].to(device=device, dtype=dtype)
                encoded = self.encode(cond).contiguous()
                parts.append(encoded)
                if self.append_condition_mask:
                    parts.append(torch.ones(latent_mask_shape, dtype=dtype, device=device))

        return torch.cat(parts, dim=1)

    def build_condition(
        self,
        data_batch: dict,
        latent_condition: Tensor,
    ) -> VideoDiffusionRendererCondition:
        return VideoDiffusionRendererCondition(
            crossattn_emb=data_batch["t5_text_embeddings"],
            crossattn_mask=data_batch["t5_text_mask"],
            padding_mask=data_batch.get("padding_mask"),
            fps=data_batch.get("fps"),
            num_frames=data_batch.get("num_frames"),
            image_size=data_batch.get("image_size"),
            latent_condition=latent_condition,
            context_index=data_batch.get("context_index"),
        )

    def _shape_source(self, data_batch: dict) -> Tensor:
        for key in self.condition_keys:
            if key in data_batch:
                return data_batch[key]
        if "video" in data_batch:
            return data_batch["video"]
        raise KeyError("No condition key or 'video' present in data_batch to derive shape.")

    @torch.no_grad()
    def generate(
        self,
        data_batch: dict,
        guidance: float = 0.0,
        num_steps: int = 15,
        seed: int = 1000,
        on_step=None,
    ) -> Tensor:
        latent_condition = self.prepare_latent_conditions(data_batch)
        condition = self.build_condition(data_batch, latent_condition)

        src = self._shape_source(data_batch)
        C = self.tokenizer.channel
        T_pixel = src.shape[2]
        F = (T_pixel - 1) // 8 + 1
        H = src.shape[3] // self.tokenizer.spatial_compression_factor
        W = src.shape[4] // self.tokenizer.spatial_compression_factor
        state_shape = (1, C, F, H, W)

        self.scheduler.set_timesteps(num_steps)
        xt = torch.randn(state_shape, generator=torch.Generator("cpu").manual_seed(seed)).to(**self.tensor_kwargs) * self.scheduler.init_noise_sigma

        from tqdm import tqdm
        for step_idx, t in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising", leave=False)):
            xt = xt.to(**self.tensor_kwargs)
            xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
            t_dev = t.to(**self.tensor_kwargs)
            output = self.net(x=xt_scaled, timesteps=t_dev, **condition.to_dict())
            if guidance > 0:
                uncond = self.build_condition(data_batch, torch.zeros_like(latent_condition))
                output_uncond = self.net(x=xt_scaled, timesteps=t_dev, **uncond.to_dict())
                output = output + guidance * (output - output_uncond)
            xt = self.scheduler.step(output, t, xt).prev_sample
            if on_step is not None:
                on_step(step_idx + 1, num_steps)

        video = self.decode(xt)
        return video


class CosmosForwardRenderer(CosmosRendererBase):
    """Forward renderer: G-buffers + envmap → RGB."""

    use_context_embedding = False

    def __init__(
        self,
        condition_keys: List[str] = None,
        condition_drop_rate: float = 0.05,
        append_condition_mask: bool = True,
        num_frames: int = 57,
        height: int = 704,
        width: int = 1280,
        sigma_data: float = 0.5,
    ):
        if condition_keys is None:
            condition_keys = [
                "basecolor", "normal", "metallic", "roughness",
                "depth", "env_ldr", "env_log", "env_nrm",
            ]
        super().__init__(
            condition_keys=condition_keys,
            condition_drop_rate=condition_drop_rate,
            append_condition_mask=append_condition_mask,
            num_frames=num_frames, height=height, width=width,
            sigma_data=sigma_data,
        )


class CosmosInverseRenderer(CosmosRendererBase):
    """Inverse renderer: RGB → G-buffer (one pass selected via context_index)."""

    use_context_embedding = True

    def __init__(
        self,
        condition_keys: List[str] = None,
        condition_drop_rate: float = 0.0,
        append_condition_mask: bool = False,
        num_frames: int = 57,
        height: int = 704,
        width: int = 1280,
        sigma_data: float = 0.5,
    ):
        if condition_keys is None:
            condition_keys = ["rgb"]
        super().__init__(
            condition_keys=condition_keys,
            condition_drop_rate=condition_drop_rate,
            append_condition_mask=append_condition_mask,
            num_frames=num_frames, height=height, width=width,
            sigma_data=sigma_data,
        )
