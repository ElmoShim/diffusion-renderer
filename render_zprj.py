"""Render a .zprj garment with the diffusion renderer.

Usage:
    python render_zprj.py samples/garment.zprj
    python render_zprj.py samples/garment.zprj --mode turntable
    python render_zprj.py samples/garment.zprj --mode rotate-light
    python render_zprj.py samples/garment.zprj --mode still --hdr examples/hdri/pink_sunrise_1k.hdr
"""

import argparse
import os

import torch

from utils.utils_render import save_tensor_as_png, save_video
from utils.utils_render_vtk import render_gbuffers, build_scene_actors


_model_cache = None


def _load_model(cfg, device="cuda"):
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    from src.cosmos.model import CosmosForwardRenderer

    model = CosmosForwardRenderer(
        condition_keys=list(cfg.condition_keys),
        condition_drop_rate=0,
        append_condition_mask=cfg.append_condition_mask,
        num_frames=cfg.inference_n_frames,
        height=cfg.inference_res[0],
        width=cfg.inference_res[1],
    )
    model.build_network()

    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.model_checkpoint)
    model.load_checkpoint(ckpt_path)

    tokenizer_dir = os.path.join(cfg.checkpoint_dir, cfg.tokenizer_dir)
    model.load_tokenizer(tokenizer_dir)

    model.to_device()
    model.eval()
    _model_cache = model
    return model


# ── forward rendering ────────────────────────────────────────────────

def forward_render(gbuffers_list, hdr_path, device="cuda", rotate_light=False,
                    seed=None, num_samples=1, on_sample=None, drop_conds=None,
                    on_step=None):
    """Run diffusion forward rendering.

    Args:
        gbuffers_list: single dict {name: (H,W,3)} or list of dicts for multi-frame.
        hdr_path: path to HDR environment map.
        rotate_light: if True, rotate environment light 360 degrees over frames.
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
    from PIL import Image
    from utils.utils_env_proj import process_environment_map
    from src.data.rendering_utils import envmap_vec

    cfg = OmegaConf.load("configs/xrgb_inference.yaml")
    n_frames = cfg.inference_n_frames
    base_seed = seed if seed is not None else cfg.get("seed", 1000)

    if isinstance(gbuffers_list, dict):
        gbuffers_list = [gbuffers_list] * n_frames

    print("Loading model...")
    model = _load_model(cfg, device=device)

    # Stack per-frame G-buffers → (1, 3, T, H, W) in [-1, 1]
    gbuf_names = list(gbuffers_list[0].keys())
    cond_tensors = {}
    for name in gbuf_names:
        frames_t = torch.stack([gb[name] for gb in gbuffers_list], dim=0)  # (T, H, W, 3)
        t = frames_t.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 3, T, H, W)
        cond_tensors[name] = t * 2 - 1  # [0,1] → [-1,1]

    # Environment map
    env_resolution = tuple(cfg.get("env_resolution", [704, 1280]))
    env_dict = process_environment_map(
        hdr_path, resolution=env_resolution, num_frames=n_frames,
        fixed_pose=True, rotate_envlight=rotate_light,
        env_format=["proj", "fixed"], device=device,
    )
    cond_tensors["env_ldr"] = env_dict["env_ldr"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
    cond_tensors["env_log"] = env_dict["env_log"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
    env_nrm = envmap_vec(env_resolution, device=device)  # already [-1,1]
    cond_tensors["env_nrm"] = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(cond_tensors["env_ldr"])

    # Handle drop_conds
    randomize_conds = []
    if drop_conds:
        randomize_conds = [c for c in drop_conds if c == "basecolor"]
        for c in drop_conds:
            if c != "basecolor":
                cond_tensors.pop(c, None)

    # Build data_batch skeleton
    h, w = cfg.inference_res[0], cfg.inference_res[1]
    data_batch = {
        "video": torch.zeros(1, 3, n_frames, h, w, dtype=torch.bfloat16, device=device),
        "t5_text_embeddings": torch.zeros(1, 512, 1024, dtype=torch.bfloat16, device=device),
        "t5_text_mask": torch.ones(1, 512, dtype=torch.bfloat16, device=device),
        "image_size": torch.tensor([[h, w, h, w]], dtype=torch.bfloat16, device=device),
        "fps": torch.tensor([24], dtype=torch.bfloat16, device=device),
        "num_frames": torch.tensor([n_frames], dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros(1, 1, h, w, dtype=torch.bfloat16, device=device),
    }

    all_results = []
    for i in range(num_samples):
        current_seed = base_seed + i
        print(f"Running forward rendering (sample {i+1}/{num_samples}, seed={current_seed}, rotate_light={rotate_light})...")

        sample_cond = dict(cond_tensors)
        if randomize_conds:
            rng = torch.Generator().manual_seed(current_seed)
            for name in randomize_conds:
                color = torch.rand(3, generator=rng) * 2 - 1
                filled = color.view(1, 3, 1, 1, 1).expand_as(cond_tensors[name]).to(device)
                sample_cond[name] = filled
                print(f"  {name} → random color")

        batch = dict(data_batch)
        for k, v in sample_cond.items():
            batch[k] = v.to(dtype=torch.bfloat16)

        def _on_step(step, total):
            if on_step is not None:
                on_step(i, num_samples, step, total)

        video = model.generate(batch, guidance=cfg.get("guidance", 0.0),
                               num_steps=cfg.inference_n_steps, seed=current_seed,
                               on_step=_on_step)

        # video: (1, 3, T, H, W) in [-1, 1] → list of PIL Images
        video_uint8 = ((1 + video[0]).clamp(0, 2) / 2 * 255).to(torch.uint8)
        frames = [Image.fromarray(video_uint8[:, t].permute(1, 2, 0).cpu().numpy()) for t in range(video_uint8.shape[1])]

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
    parser.add_argument("--output", type=str, default="output", help="Output root directory; files go to <output>/<stem>/")
    parser.add_argument("--resolution", type=int, nargs="+", default=[704, 1280],
                        help="Render resolution (H W or single value for square)")
    parser.add_argument("--fov", type=float, default=10.0, help="Camera FOV in degrees")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gbuffer-only", action="store_true", help="Only render G-buffers, skip forward rendering")
    parser.add_argument("--mode", choices=["still", "turntable", "rotate-light"], default="still",
                        help="still: single image | turntable: camera orbits 360° | rotate-light: light rotates 360°")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--gif", action="store_true", help="Save as GIF instead of MP4 (turntable/rotate-light)")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    import zprj_loader

    cfg = OmegaConf.load("configs/xrgb_inference.yaml")
    res = args.resolution if len(args.resolution) == 2 else [args.resolution[0], args.resolution[0]]
    n_frames = cfg.inference_n_frames

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = f"{args.output}/{stem}/"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load zprj
    print(f"Loading {args.input} ...")
    scene = zprj_loader.parse(args.input)
    if not scene.valid:
        print(f"Error: {scene.error}")
        return
    actors_data = build_scene_actors(scene)
    print(f"Scene: {len(actors_data)} mesh parts")

    # 2. Render G-buffers
    if args.mode == "turntable":
        print(f"Rendering {n_frames} turntable frames at {res[0]}x{res[1]}...")
        gbuffers_list = [
            render_gbuffers(scene, resolution=res, fov_deg=args.fov,
                            azimuth_deg=i * 360.0 / n_frames, device=args.device,
                            _actors_data=actors_data)
            for i in range(n_frames)
        ]
    else:
        gbuffers_list = [render_gbuffers(scene, resolution=res, fov_deg=args.fov,
                                         device=args.device, _actors_data=actors_data)]

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
