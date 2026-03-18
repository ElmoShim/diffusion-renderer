"""Forward-render G-buffer PNGs with the diffusion renderer.

Usage:
    python render_forward.py tmp/garment/
    python render_forward.py tmp/garment/ --mode rotate-light
    python render_forward.py tmp/garment/ --hdr examples/hdri/pink_sunrise_1k.hdr
    python render_forward.py tmp/garment/ --num-samples 10 --seed 42
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from render_zprj import forward_render
from utils.utils_render import save_video

GBUFFER_NAMES = ["basecolor", "normal", "depth", "roughness", "metallic"]


def load_gbuffers(input_dir, resolution=None):
    """Load G-buffer PNGs from a directory into a dict of (H,W,3) float32 tensors.

    Missing G-buffers are filled with zero placeholder tensors and their names
    are returned separately so they can be passed as drop_conds to the pipeline
    (zeroed in latent space, letting the model generate plausible values).
    """
    gbuffers = {}
    loaded = []
    missing = []

    # First pass: load existing files and determine resolution
    for name in GBUFFER_NAMES:
        path = os.path.join(input_dir, f"{name}.png")
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            tensor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            gbuffers[name] = tensor
            loaded.append(name)
        else:
            missing.append(name)

    if not loaded:
        raise FileNotFoundError(f"No G-buffer PNGs found in {input_dir}")

    # Determine resolution from loaded images (or override)
    h, w = gbuffers[loaded[0]].shape[:2]
    if resolution is not None:
        h, w = resolution, resolution

    # Second pass: fill missing with zero placeholders
    for name in missing:
        gbuffers[name] = torch.zeros(h, w, 3, dtype=torch.float32)

    if loaded:
        print(f"  Loaded: {', '.join(loaded)}")
    if missing:
        print(f"  Missing (drop_conds): {', '.join(missing)}")

    return gbuffers, missing


def main():
    parser = argparse.ArgumentParser(description="Forward-render G-buffer PNGs with the diffusion renderer.")
    parser.add_argument("input_dir", type=str, help="Directory containing G-buffer PNGs (basecolor, normal, depth, roughness, metallic). Missing buffers are zero-filled.")
    parser.add_argument("--hdr", type=str, default="examples/hdri/sunny_vondelpark_1k.hdr", help="HDR environment map")
    parser.add_argument("--mode", choices=["still", "rotate-light"], default="still",
                        help="still: single image | rotate-light: light rotates 360°")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--gif", action="store_true", help="Save as GIF instead of MP4 (rotate-light)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate (each with a different seed)")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (default: from config)")
    args = parser.parse_args()

    # Load G-buffers
    print(f"Loading G-buffers from {args.input_dir} ...")
    gbuffers, drop_conds = load_gbuffers(args.input_dir)
    h, w = gbuffers[GBUFFER_NAMES[0]].shape[:2]
    print(f"G-buffer resolution: {w}x{h}")

    # Output directory
    dir_basename = os.path.basename(os.path.normpath(args.input_dir))
    out_dir = f"./tmp/{dir_basename}_rendered"
    os.makedirs(out_dir, exist_ok=True)

    # Save callback — saves each sample immediately after generation
    hdr_stem = os.path.splitext(os.path.basename(args.hdr))[0]

    def save_sample(i, seed, frames):
        suffix = f"_sample{i}" if args.num_samples > 1 else ""
        out_path = os.path.join(out_dir, f"rendered_{hdr_stem}{suffix}.png")
        frames[0].save(out_path)
        print(f"Saved {out_path} (seed={seed})")

        if args.mode == "rotate-light":
            if args.gif:
                gif_path = os.path.join(out_dir, f"rotate-light_{hdr_stem}{suffix}.gif")
                frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                               duration=1000 // args.fps, loop=0)
                print(f"Saved {gif_path}")
            else:
                video_path = os.path.join(out_dir, f"rotate-light_{hdr_stem}{suffix}.mp4")
                save_video(frames, video_path, fps=args.fps)
                print(f"Saved {video_path}")

    # Forward render
    forward_render(
        gbuffers, args.hdr, device=args.device,
        rotate_light=(args.mode == "rotate-light"),
        seed=args.seed, num_samples=args.num_samples,
        on_sample=save_sample,
        drop_conds=drop_conds or None,
    )


if __name__ == "__main__":
    main()
