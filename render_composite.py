"""Composite a .zprj garment onto background G-buffers and forward-render.

Usage:
    python render_composite.py samples/garment.zprj
    python render_composite.py samples/garment.zprj --bg asset/inverse/image_1
    python render_composite.py samples/garment.zprj --gbuffer-only
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from utils.utils_render import save_tensor_as_png
from utils.utils_render_vtk import render_gbuffers


GBUFFER_NAMES = ["basecolor", "normal", "depth", "roughness", "metallic"]


def load_bg_gbuffers(bg_dir, resolution):
    """Load background G-buffer PNGs and resize to target resolution (H, W)."""
    res_h, res_w = resolution
    gb = {}
    for name in GBUFFER_NAMES:
        path = os.path.join(bg_dir, f"{name}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Background G-buffer not found: {path}")
        img = Image.open(path).convert("RGB").resize((res_w, res_h), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gb[name] = torch.from_numpy(arr)
    return gb


def composite_gbuffers(fg_gb, bg_gb, fg_mask):
    """Composite foreground G-buffers onto background using mask.

    fg_mask: (H, W, 1) float tensor, 1 where foreground exists.
    """
    result = {}
    for name in GBUFFER_NAMES:
        fg = fg_gb[name]
        bg = bg_gb[name]
        result[name] = fg * fg_mask + bg * (1 - fg_mask)
    return result


def main():
    parser = argparse.ArgumentParser(description="Composite a .zprj garment onto background G-buffers.")
    parser.add_argument("input", type=str, help="Path to .zprj file")
    parser.add_argument("--bg", type=str, default="asset/inverse/image_1",
                        help="Background G-buffer directory (default: asset/inverse/image_1)")
    parser.add_argument("--hdr", type=str, default="examples/hdri/sunny_vondelpark_1k.hdr",
                        help="HDR environment map")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--resolution", type=int, nargs="+", default=[704, 1280],
                        help="Render resolution (H W)")
    parser.add_argument("--fov", type=float, default=10.0, help="Camera FOV in degrees")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gbuffer-only", action="store_true",
                        help="Only save composited G-buffers, skip forward rendering")
    args = parser.parse_args()

    import zprj_loader

    res = args.resolution if len(args.resolution) == 2 else [args.resolution[0], args.resolution[0]]

    stem = os.path.splitext(os.path.basename(args.input))[0]
    bg_name = os.path.basename(args.bg.rstrip("/"))
    out_dir = os.path.join(args.output, f"{stem}_on_{bg_name}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load background G-buffers
    print(f"Loading background G-buffers from {args.bg} ...")
    bg_gb = load_bg_gbuffers(args.bg, res)

    # 2. Load zprj and render foreground G-buffers (no dummy background)
    print(f"Loading {args.input} ...")
    scene = zprj_loader.parse(args.input)
    if not scene.valid:
        print(f"Error: {scene.error}")
        return
    fg_gb = render_gbuffers(scene, resolution=res, fov_deg=args.fov, device=args.device,
                            background=False)

    # 3. Build foreground mask and move bg to same device
    fg_mask = (fg_gb["normal"].sum(dim=-1, keepdim=True) > 0).float()
    device_t = fg_gb["normal"].device
    for k in bg_gb:
        bg_gb[k] = bg_gb[k].to(device_t)

    # 4. Composite
    print("Compositing foreground onto background ...")
    comp_gb = composite_gbuffers(fg_gb, bg_gb, fg_mask)

    # Save composited G-buffers
    for name, tensor in comp_gb.items():
        save_tensor_as_png(tensor, os.path.join(out_dir, f"{name}.png"))
    save_tensor_as_png(fg_mask.expand_as(fg_gb["normal"]),
                       os.path.join(out_dir, "mask.png"))
    print(f"Composited G-buffers saved to {out_dir}")

    if args.gbuffer_only:
        return

    # 5. Forward render
    from render_zprj import forward_render

    print("Running forward rendering ...")
    frames = forward_render(comp_gb, args.hdr, device=args.device)

    hdr_stem = os.path.splitext(os.path.basename(args.hdr))[0]
    out_path = os.path.join(out_dir, f"rendered_{hdr_stem}.png")
    frames[0].save(out_path)
    print(f"Rendered image saved to {out_path}")


if __name__ == "__main__":
    main()
