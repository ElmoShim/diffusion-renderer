"""Render a .zprj garment with the diffusion renderer.

Usage:
    python render_zprj.py samples/garment.zprj
    python render_zprj.py samples/garment.zprj --mode turntable
    python render_zprj.py samples/garment.zprj --mode rotate-light
    python render_zprj.py samples/garment.zprj --mode still --hdr examples/hdri/pink_sunrise_1k.hdr
"""

import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch

from utils.utils_render import (
    load_mesh, render_gbuffers, precompute_mesh_gpu,
    save_tensor_as_png, save_video,
)


# ── forward rendering ────────────────────────────────────────────────

def forward_render(gbuffers_list, hdr_path, device="cuda", rotate_light=False,
                    seed=None, num_samples=1, on_sample=None, drop_conds=None):
    """Run diffusion forward rendering.

    Args:
        gbuffers_list: single dict {name: (H,W,3)} or list of 24 dicts for multi-frame.
        hdr_path: path to HDR environment map.
        rotate_light: if True, rotate environment light 360 degrees over 24 frames.
        seed: random seed (default: from config). Use different seeds for varied outputs.
        num_samples: number of samples to generate. Each gets a different seed.
            When num_samples > 1, returns list of list of PIL Images.
        on_sample: optional callback(sample_index, seed, frames) called after each sample.
        drop_conds: optional list of condition names to drop (zero in latent space).

    Returns:
        list of PIL Images (num_samples=1), or
        list of list of PIL Images (num_samples>1).
    """
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel
    from src.models.env_encoder import EnvEncoder
    from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
    from utils.utils_env_proj import process_environment_map
    from src.data.rendering_utils import envmap_vec

    cfg = OmegaConf.load("configs/xrgb_inference.yaml")
    weights = cfg.inference_model_weights
    n_frames = cfg.inference_n_frames  # 24
    base_seed = seed if seed is not None else cfg.get("seed", 0)

    # Normalize input: single dict → list of identical frames
    if isinstance(gbuffers_list, dict):
        gbuffers_list = [gbuffers_list] * n_frames

    # Build pipeline
    print("Loading model...")
    env_encoder = EnvEncoder.from_pretrained(weights, subfolder="env_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(weights, subfolder="vae")
    unet = UNetCustomSpatioTemporalConditionModel.from_pretrained(weights, subfolder="unet")
    scheduler = EulerDiscreteScheduler.from_pretrained(weights, subfolder="scheduler")

    dtype = torch.float16
    for m in [env_encoder, vae, unet]:
        m.to(device, dtype=dtype)

    pipeline = RGBXVideoDiffusionPipeline(
        vae=vae, image_encoder=None, feature_extractor=None,
        unet=unet, scheduler=scheduler, env_encoder=env_encoder,
        scale_cond_latents=cfg.model_pipeline.get("scale_cond_latents", False),
        cond_mode="env",
    )
    pipeline.scheduler.register_to_config(timestep_spacing="trailing")
    try:
        pipeline.load_lora_weights(weights, subfolder="lora", adapter_name="real-lora")
    except Exception:
        print("LoRA weights not found, using base weights")
    pipeline = pipeline.to(device).to(dtype)
    pipeline.set_progress_bar_config(desc="Denoising")

    # Stack per-frame G-buffers into (1, F, C, H, W)
    gbuf_names = list(gbuffers_list[0].keys())
    cond_images = {}
    for name in gbuf_names:
        frames_t = torch.stack([gb[name] for gb in gbuffers_list], dim=0)  # (F, H, W, 3)
        cond_images[name] = frames_t.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)  # (1, F, 3, H, W)

    # Environment map
    env_resolution = tuple(cfg.model_pipeline.get("env_resolution", [512, 512]))
    env_dict = process_environment_map(
        hdr_path, resolution=env_resolution, num_frames=n_frames,
        fixed_pose=True, rotate_envlight=rotate_light,
        env_format=["proj", "fixed", "ball"], device=device,
    )
    cond_images["env_ldr"] = env_dict["env_ldr"].unsqueeze(0).permute(0, 1, 4, 2, 3)
    cond_images["env_log"] = env_dict["env_log"].unsqueeze(0).permute(0, 1, 4, 2, 3)
    env_nrm = envmap_vec(env_resolution, device=device) * 0.5 + 0.5
    cond_images["env_nrm"] = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).expand_as(cond_images["env_ldr"])

    cond_labels = cfg.model_pipeline.cond_images
    autocast_ctx = torch.autocast(device, enabled=True) if device == "cuda" else nullcontext()
    h, w = gbuffers_list[0]["basecolor"].shape[:2]

    # Separate drop_conds: randomize basecolor per sample, truly drop the rest
    randomize_conds = []
    actual_drop_conds = None
    if drop_conds:
        randomize_conds = [c for c in drop_conds if c == "basecolor"]
        rest = [c for c in drop_conds if c != "basecolor"]
        actual_drop_conds = rest or None

    all_results = []
    for i in range(num_samples):
        current_seed = base_seed + i
        print(f"Running forward rendering (sample {i+1}/{num_samples}, seed={current_seed}, rotate_light={rotate_light})...")
        generator = torch.Generator(device=device).manual_seed(current_seed)

        # Fill randomized conditions with a random solid color per sample
        sample_cond_images = cond_images
        if randomize_conds:
            sample_cond_images = dict(cond_images)
            rng = torch.Generator().manual_seed(current_seed)
            for name in randomize_conds:
                color = torch.rand(3, generator=rng)
                filled = color.view(1, 1, 3, 1, 1).expand_as(cond_images[name]).to(device)
                sample_cond_images[name] = filled
                print(f"  {name} → random color ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

        with autocast_ctx:
            frames = pipeline(
                sample_cond_images, cond_labels,
                height=h, width=w,
                num_frames=n_frames,
                num_inference_steps=cfg.inference_n_steps,
                min_guidance_scale=cfg.inference_min_guidance_scale,
                max_guidance_scale=cfg.inference_max_guidance_scale,
                fps=cfg.get("fps", 7),
                motion_bucket_id=cfg.get("motion_bucket_id", 127),
                noise_aug_strength=cfg.get("cond_aug", 0),
                generator=generator,
                cross_attention_kwargs={"scale": cfg.get("lora_scale", 0.0)},
                dynamic_guidance=False,
                decode_chunk_size=cfg.get("decode_chunk_size", None),
                drop_conds=actual_drop_conds,
            ).frames[0]  # list of PIL Images

        if on_sample is not None:
            on_sample(i, current_seed, frames)
        all_results.append(frames)

    if num_samples == 1:
        return all_results[0]
    return all_results


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Render a .zprj garment with the diffusion renderer.")
    parser.add_argument("input", type=str, help="Path to .zprj file")
    parser.add_argument("--hdr", type=str, default="examples/hdri/sunny_vondelpark_1k.hdr", help="HDR environment map")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: tmp/<stem>/)")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")
    parser.add_argument("--fov", type=float, default=15.0, help="Camera FOV in degrees")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gbuffer-only", action="store_true", help="Only render G-buffers, skip forward rendering")
    parser.add_argument("--mode", choices=["still", "turntable", "rotate-light"], default="still",
                        help="still: single image | turntable: camera orbits 360° | rotate-light: light rotates 360°")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--gif", action="store_true", help="Save as GIF instead of MP4 (turntable/rotate-light)")
    args = parser.parse_args()

    import zprj_loader

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output or f"tmp/{stem}/"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load zprj
    print(f"Loading {args.input} ...")
    scene = zprj_loader.parse(args.input)
    if not scene.valid:
        print(f"Error: {scene.error}")
        return
    mesh = load_mesh(scene)
    print(f"Mesh: {mesh['positions'].shape[0]} verts, {mesh['faces'].shape[0]} tris")

    # 2. Render G-buffers
    if args.mode == "turntable":
        n_frames = 24
        precomp = precompute_mesh_gpu(mesh, device=args.device)
        print(f"Rendering {n_frames} turntable frames...")
        gbuffers_list = [
            render_gbuffers(mesh, resolution=args.resolution, fov_deg=args.fov,
                            azimuth_deg=i * 360.0 / n_frames, device=args.device, _precomp=precomp)
            for i in range(n_frames)
        ]
    else:
        gbuffers_list = [render_gbuffers(mesh, resolution=args.resolution, fov_deg=args.fov, device=args.device)]

    # Save first frame G-buffers
    for name, tensor in gbuffers_list[0].items():
        save_tensor_as_png(tensor, os.path.join(out_dir, f"{name}.png"))
    print(f"G-buffers saved to {out_dir}")

    if args.gbuffer_only:
        return

    # 3. Forward render
    frames = forward_render(
        gbuffers_list if args.mode == "turntable" else gbuffers_list[0],
        args.hdr, device=args.device,
        rotate_light=(args.mode == "rotate-light"),
    )

    hdr_stem = os.path.splitext(os.path.basename(args.hdr))[0]
    if args.mode == "still":
        out_path = os.path.join(out_dir, f"rendered_{hdr_stem}.png")
        frames[0].save(out_path)
        print(f"Rendered image saved to {out_path}")
    else:
        frames[0].save(os.path.join(out_dir, f"rendered_{hdr_stem}.png"))
        if args.gif:
            gif_path = os.path.join(out_dir, f"{args.mode}_{hdr_stem}.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=1000 // args.fps, loop=0)
            print(f"GIF saved to {gif_path}")
        else:
            video_path = os.path.join(out_dir, f"{args.mode}_{hdr_stem}.mp4")
            save_video(frames, video_path, fps=args.fps)
            print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
