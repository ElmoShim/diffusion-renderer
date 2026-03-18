"""Export G-buffer images (basecolor, normal, depth, roughness, metallic) from a .zprj file.

Renders garment patterns, avatar, trims, buttons, and zippers.
Output is compatible with render_forward.py.

Usage:
    python export_gbuffers.py samples/garment.zprj
    python export_gbuffers.py samples/garment.zprj --output tmp/gbuffers/
    python export_gbuffers.py samples/garment.zprj --resolution 1024
"""

import argparse
import os

import zprj_loader
from utils.utils_render import load_mesh, render_gbuffers, save_tensor_as_png


def main():
    parser = argparse.ArgumentParser(description="Export G-buffer PNGs from a .zprj file.")
    parser.add_argument("zprj_file", help="Path to .zprj file")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: tmp/<stem>/)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fov", type=float, default=15.0)
    parser.add_argument("--azimuth", type=float, default=0.0, help="Camera azimuth in degrees")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.zprj_file))[0]
    out_dir = args.output or f"tmp/{stem}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.zprj_file} ...")
    scene = zprj_loader.parse(args.zprj_file)
    if not scene.valid:
        print(f"Error: {scene.error}")
        return

    mesh = load_mesh(scene)
    print(f"Mesh: {mesh['positions'].shape[0]} verts, {mesh['faces'].shape[0]} tris")

    gb = render_gbuffers(mesh, resolution=args.resolution, fov_deg=args.fov,
                         azimuth_deg=args.azimuth, device=args.device)
    for name, tensor in gb.items():
        save_tensor_as_png(tensor, os.path.join(out_dir, f"{name}.png"))

    print(f"G-buffers saved to {out_dir}")


if __name__ == "__main__":
    main()
