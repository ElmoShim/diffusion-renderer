# Cosmos VAE tokenizer for video encode/decode.
# Extracted from cosmos_predict1/diffusion/module/pretrained_vae.py
# Uses JIT-compiled VAE models with mean/std normalization.

import os

import torch
from einops import rearrange


class CosmosTokenizer(torch.nn.Module):
    """Video tokenizer using Cosmos CV8x8x8 JIT VAE.

    Handles:
    - Loading JIT-compiled encoder/decoder
    - Mean/std normalization of latents
    - Chunked encode/decode for long videos
    - Both image (T=1) and video paths
    """

    def __init__(
        self,
        latent_ch=16,
        spatial_compression_factor=8,
        temporal_compression_factor=8,
        pixel_chunk_duration=57,
        max_enc_batch_size=8,
        max_dec_batch_size=4,
    ):
        super().__init__()
        self.channel = latent_ch
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compress_factor = temporal_compression_factor
        self._pixel_chunk_duration = pixel_chunk_duration
        self.max_enc_batch_size = max_enc_batch_size
        self.max_dec_batch_size = max_dec_batch_size
        self.dtype = torch.bfloat16

        self.encoder = None
        self.decoder = None

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        return self._temporal_compress_factor

    @property
    def pixel_chunk_duration(self):
        return self._pixel_chunk_duration

    @property
    def latent_chunk_duration(self):
        return (self._pixel_chunk_duration - 1) // self._temporal_compress_factor + 1

    def get_latent_num_frames(self, num_pixel_frames):
        if num_pixel_frames == 1:
            return 1
        assert num_pixel_frames % self._pixel_chunk_duration == 0
        return num_pixel_frames // self._pixel_chunk_duration * self.latent_chunk_duration

    def load_weights(self, vae_dir):
        """Load encoder, decoder, and normalization stats from directory."""
        self.encoder = torch.load(os.path.join(vae_dir, "encoder.jit"), weights_only=False)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.dtype)

        self.decoder = torch.load(os.path.join(vae_dir, "decoder.jit"), weights_only=False)
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.to(self.dtype)

        # Video mean/std
        latent_mean, latent_std = torch.load(os.path.join(vae_dir, "mean_std.pt"), weights_only=True)
        latent_mean = latent_mean.view(self.channel, -1)[:, :self.latent_chunk_duration]
        latent_std = latent_std.view(self.channel, -1)[:, :self.latent_chunk_duration]
        target_shape = [1, self.channel, self.latent_chunk_duration, 1, 1]
        self.register_buffer("video_latent_mean", latent_mean.to(self.dtype).reshape(*target_shape), persistent=False)
        self.register_buffer("video_latent_std", latent_std.to(self.dtype).reshape(*target_shape), persistent=False)

        # Image mean/std
        img_latent_mean, img_latent_std = torch.load(os.path.join(vae_dir, "image_mean_std.pt"), weights_only=True)
        img_target_shape = [1, self.channel, 1, 1, 1]
        self.register_buffer("image_latent_mean", img_latent_mean.to(self.dtype).reshape(*img_target_shape), persistent=False)
        self.register_buffer("image_latent_std", img_latent_std.to(self.dtype).reshape(*img_target_shape), persistent=False)

    @torch.no_grad()
    def encode(self, state):
        """Encode pixel-space video to latent space.

        Args:
            state: (B, C, T, H, W) in [-1, 1]
        Returns:
            latent: (B, latent_ch, T', H', W') normalized
        """
        B, C, T, H, W = state.shape
        if T == 1:
            return self._encode_image(state)
        return self._encode_video(state)

    @torch.no_grad()
    def decode(self, latent):
        """Decode latent space to pixel space.

        Args:
            latent: (B, latent_ch, T', H', W') normalized
        Returns:
            state: (B, C, T, H, W) in [-1, 1]
        """
        B, C, T, H, W = latent.shape
        if T == 1:
            return self._decode_image(latent)
        return self._decode_video(latent)

    def _encode_image(self, state):
        in_dtype = state.dtype
        encoded = self.encoder(state.squeeze(2).to(self.dtype))
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        encoded = encoded.unsqueeze(2)
        return ((encoded.to(in_dtype) - self.image_latent_mean.to(in_dtype))
                / self.image_latent_std.to(in_dtype))

    def _decode_image(self, latent):
        in_dtype = latent.dtype
        latent = latent * self.image_latent_std.to(in_dtype) + self.image_latent_mean.to(in_dtype)
        decoded = self.decoder(latent.squeeze(2).to(self.dtype))
        return decoded.to(in_dtype).unsqueeze(2)

    def _encode_video(self, state):
        B, C, T, H, W = state.shape
        state_chunked = rearrange(state, "b c (n t) h w -> (b n) c t h w", t=self._pixel_chunk_duration)

        if state_chunked.shape[0] > self.max_enc_batch_size:
            latent_parts = []
            for i in range(0, state_chunked.shape[0], self.max_enc_batch_size):
                chunk = state_chunked[i:i + self.max_enc_batch_size]
                encoded = self.encoder(chunk.to(self.dtype))
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
                latent_parts.append(encoded)
            latent = torch.cat(latent_parts, dim=0)
        else:
            latent = self.encoder(state_chunked.to(self.dtype))
            if isinstance(latent, tuple):
                latent = latent[0]

        in_dtype = state.dtype
        latent = latent.to(in_dtype)
        latent = (latent - self.video_latent_mean.to(in_dtype)) / self.video_latent_std.to(in_dtype)
        latent = rearrange(latent, "(b n) c t h w -> b c (n t) h w", b=B)
        return latent

    def _decode_video(self, latent):
        B, C, T, H, W = latent.shape
        in_dtype = latent.dtype
        latent = latent * self.video_latent_std.to(in_dtype) + self.video_latent_mean.to(in_dtype)
        latent_chunked = rearrange(latent, "b c (n t) h w -> (b n) c t h w", t=self.latent_chunk_duration)

        if latent_chunked.shape[0] > self.max_dec_batch_size:
            state_parts = []
            for i in range(0, latent_chunked.shape[0], self.max_dec_batch_size):
                chunk = latent_chunked[i:i + self.max_dec_batch_size]
                decoded = self.decoder(chunk.to(self.dtype))
                state_parts.append(decoded)
            state = torch.cat(state_parts, dim=0)
        else:
            state = self.decoder(latent_chunked.to(self.dtype))

        state = rearrange(state, "(b n) c t h w -> b c (n t) h w", b=B)
        return state.to(in_dtype)
