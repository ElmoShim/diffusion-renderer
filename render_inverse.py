"""Inverse-render with Cosmos Diffusion Renderer: RGB image/video → G-buffer maps.

Estimates basecolor, normal, depth, roughness, and metallic maps from input
images or videos using the Diffusion_Renderer_Inverse_Cosmos_7B model.

Usage:
    # Single image (1 frame)
    python render_inverse.py asset/examples/image_examples/image_1.jpg

    # Folder of images, each processed independently
    python render_inverse.py asset/examples/image_examples/

    # Video file (frames extracted automatically)
    python render_inverse.py asset/examples/video_examples/video1.mp4 --num-frames 57

    # Select only some G-buffer passes
    python render_inverse.py asset/examples/image_examples/image_1.jpg \\
        --passes basecolor normal depth
"""

import argparse
import os
from glob import glob

import numpy as np
import torch
from PIL import Image


GBUFFER_INDEX_MAPPING = {
    "basecolor":        0,
    "metallic":         1,
    "roughness":        2,
    "normal":           3,
    "depth":            4,
    "diffuse_albedo":   5,
    "specular_albedo":  6,
}

DEFAULT_PASSES = ["basecolor", "normal", "depth", "roughness", "metallic"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


_model_cache = None


def _load_model(cfg, device="cuda"):
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    from src.cosmos.model import CosmosInverseRenderer

    model = CosmosInverseRenderer(
        condition_keys=list(cfg.condition_keys),
        condition_drop_rate=0.0,
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


def _load_frames_from_path(path, num_frames, resolution):
    """Load input as (T, H, W, 3) uint8 numpy array, padding/truncating to num_frames."""
    h, w = resolution
    ext = os.path.splitext(path)[1].lower()

    if ext in VIDEO_EXTS:
        import imageio.v3 as imageio
        frames = imageio.imread(path)  # (T, H, W, C) or (H, W, C)
        if frames.ndim == 3:
            frames = frames[None]
        frames = frames[..., :3]
    elif ext in IMAGE_EXTS:
        img = Image.open(path).convert("RGB")
        frames = np.asarray(img)[None]  # (1, H, W, 3)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Resize each frame using PIL
    out = []
    for f in frames:
        pil = Image.fromarray(f).resize((w, h), Image.BILINEAR)
        out.append(np.asarray(pil))
    frames = np.stack(out, axis=0)

    if frames.shape[0] < num_frames:
        pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:num_frames]
    return frames  # (T, H, W, 3) uint8


def _build_data_batch(frames_uint8, cfg, device):
    """Build the data_batch dict expected by CosmosInverseRenderer.generate."""
    h, w = cfg.inference_res
    n_frames = cfg.inference_n_frames

    rgb = torch.from_numpy(frames_uint8).float() / 255.0  # (T, H, W, 3)
    rgb = rgb.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 3, T, H, W)
    rgb = rgb * 2 - 1  # [-1, 1]
    rgb = rgb.to(torch.bfloat16)

    data_batch = {
        "rgb": rgb,
        "video": rgb,  # used by tokenizer encode pathway / shape inference
        "t5_text_embeddings": torch.zeros(1, 512, 1024, dtype=torch.bfloat16, device=device),
        "t5_text_mask": torch.ones(1, 512, dtype=torch.bfloat16, device=device),
        "image_size": torch.tensor([[h, w, h, w]], dtype=torch.bfloat16, device=device),
        "fps": torch.tensor([24], dtype=torch.bfloat16, device=device),
        "num_frames": torch.tensor([n_frames], dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros(1, 1, h, w, dtype=torch.bfloat16, device=device),
        "context_index": torch.zeros(1, dtype=torch.long, device=device),
    }
    return data_batch


def _normalize_normal_video(video):
    """Match official: normalize unit-length only where magnitude is strong."""
    norm = torch.norm(video, dim=1, p=2, keepdim=True)
    normalized = video / norm.clamp(min=1e-12)
    upper, lower = 0.4, 0.2
    blend = torch.clip((norm - lower) / (upper - lower), 0, 1)
    return normalized * blend + video * (1 - blend)


def inverse_render(input_path, passes=None, device="cuda", seed=None):
    """Run inverse rendering on an image or video.

    Args:
        input_path: image, video, or folder path.
        passes: list of G-buffer pass names. Defaults to DEFAULT_PASSES.
        device: torch device.
        seed: random seed.

    Returns:
        dict {pass_name: (T, H, W, 3) uint8 numpy array}, plus key "rgb_input"
        with the (resized) input frames.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/xrgb_inverse_inference.yaml")
    if passes is None:
        passes = DEFAULT_PASSES

    if seed is None:
        seed = cfg.get("seed", 1000)

    model = _load_model(cfg, device=device)

    frames = _load_frames_from_path(input_path, cfg.inference_n_frames, cfg.inference_res)
    data_batch = _build_data_batch(frames, cfg, device=device)

    results = {"rgb_input": frames}
    for pass_name in passes:
        if pass_name not in GBUFFER_INDEX_MAPPING:
            print(f"  Skipping unknown pass: {pass_name}")
            continue
        idx = GBUFFER_INDEX_MAPPING[pass_name]
        data_batch["context_index"] = torch.tensor([idx], dtype=torch.long, device=device)
        print(f"  Estimating {pass_name} (context_index={idx})...")

        video = model.generate(
            data_batch,
            guidance=cfg.get("guidance", 0.0),
            num_steps=cfg.inference_n_steps,
            seed=seed,
        )

        if pass_name == "normal":
            video = _normalize_normal_video(video)

        # (1, 3, T, H, W) in [-1, 1] → (T, H, W, 3) uint8
        video = (1.0 + video).clamp(0, 2) / 2
        out = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        results[pass_name] = out

    return results


def _expand_input(path):
    """Return a list of (display_name, abs_path) pairs."""
    if os.path.isdir(path):
        items = []
        for ext in IMAGE_EXTS + VIDEO_EXTS:
            items.extend(sorted(glob(os.path.join(path, f"*{ext}"))))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in items]
    return [(os.path.splitext(os.path.basename(path))[0], path)]


def _save_results(results, out_dir, fps=24):
    os.makedirs(out_dir, exist_ok=True)
    for name, frames in results.items():
        if frames.shape[0] == 1:
            Image.fromarray(frames[0]).save(os.path.join(out_dir, f"{name}.png"))
        else:
            # Save first frame as preview and full sequence as mp4
            Image.fromarray(frames[0]).save(os.path.join(out_dir, f"{name}.png"))
            try:
                import imageio.v3 as imageio
                imageio.imwrite(os.path.join(out_dir, f"{name}.mp4"), frames, fps=fps)
            except Exception as e:
                print(f"  (mp4 save failed for {name}: {e})")


def main():
    parser = argparse.ArgumentParser(description="Cosmos inverse renderer: RGB → G-buffers.")
    parser.add_argument("input", type=str, help="Image, video, or folder path")
    parser.add_argument("--output", type=str, default="output/inverse",
                        help="Output directory")
    parser.add_argument("--passes", type=str, nargs="+", default=DEFAULT_PASSES,
                        choices=list(GBUFFER_INDEX_MAPPING.keys()),
                        help=f"G-buffer passes to estimate (default: {' '.join(DEFAULT_PASSES)})")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Override inference_n_frames from config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.num_frames is not None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/xrgb_inverse_inference.yaml")
        cfg.inference_n_frames = args.num_frames
        OmegaConf.save(cfg, "configs/xrgb_inverse_inference.yaml")

    items = _expand_input(args.input)
    if not items:
        print(f"No input files found at {args.input}")
        return

    for stem, path in items:
        print(f"\n=== {stem} ({path}) ===")
        results = inverse_render(path, passes=args.passes, device=args.device, seed=args.seed)
        out_dir = os.path.join(args.output, stem)
        _save_results(results, out_dir)
        print(f"  → {out_dir}")


if __name__ == "__main__":
    main()
