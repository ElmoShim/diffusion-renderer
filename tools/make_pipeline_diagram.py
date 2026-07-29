"""Build the README pipeline diagram from a rendered output directory.

Reads G-buffers + forward-rendered results from an output dir (as produced by
render_zprj.py) and composes the animated diagram used in README.md.

Usage:
    uv run tools/make_pipeline_diagram.py                      # output/250000_coat
    uv run tools/make_pipeline_diagram.py --input output/foo --hdr street_lamp_1k
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

GBUFFERS = ["basecolor", "normal", "depth", "roughness", "metallic"]

BG = (255, 255, 255)
INK = (60, 60, 60)
MUTED = (130, 130, 130)
BORDER = (215, 215, 215)

PANEL_W, PANEL_H = 104, 212
RESULT_W, RESULT_H = 150, 306
GAP = 8
LABEL_H = 16


def _font(path, size):
    return ImageFont.truetype(path, size)


def subject_crop(img, pad_x=0.28):
    """Center crop around the garment, keeping the full height."""
    w, h = img.size
    cx = w // 2
    half = int(w * 0.5 * pad_x)
    return img.crop((cx - half, 4, cx + half, h - 4))


def panel(img, size):
    return subject_crop(img).resize(size, Image.LANCZOS)


def draw_panel(canvas, img, x, y, size, label, bold=False):
    canvas.paste(img, (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle([x, y, x + size[0] - 1, y + size[1] - 1], outline=BORDER)
    f = _font(FONT_BOLD if bold else FONT, 11)
    tw = d.textlength(label, font=f)
    d.text((x + (size[0] - tw) / 2, y + size[1] + 4), label,
           fill=INK if bold else MUTED, font=f)


def draw_arrow(d, x0, x1, y, label, sublabel=None):
    d.line([x0, y, x1 - 6, y], fill=MUTED, width=2)
    d.polygon([(x1, y), (x1 - 8, y - 5), (x1 - 8, y + 5)], fill=MUTED)
    f = _font(FONT, 11)
    for i, text in enumerate([label, sublabel]):
        if not text:
            continue
        tw = d.textlength(text, font=f)
        d.text((x0 + (x1 - x0 - tw) / 2, y - 32 + i * 14), text, fill=INK, font=f)


def load_frames(path, n_keep=24):
    import imageio.v3 as iio

    frames = iio.imread(path)
    step = max(1, len(frames) // n_keep)
    return [Image.fromarray(f) for f in frames[::step]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/250000_coat", help="render output directory")
    ap.add_argument("--hdr", default="sunny_vondelpark_1k", help="HDR stem used in filenames")
    ap.add_argument("--output", default="output/readme", help="where to write the diagram")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    gbuf = [Image.open(os.path.join(args.input, f"{n}.png")).convert("RGB") for n in GBUFFERS]
    still = Image.open(os.path.join(args.input, f"rendered_{args.hdr}.png")).convert("RGB")
    video = os.path.join(args.input, f"turntable_{args.hdr}.mp4")
    turn = load_frames(video) if os.path.exists(video) else [still]

    # ── layout ───────────────────────────────────────────────────────
    box_w, box_h = 108, 56
    x_box = 16
    x_arrow1 = x_box + box_w + 12
    x_gbuf = x_arrow1 + 92
    gbuf_w = len(GBUFFERS) * PANEL_W + (len(GBUFFERS) - 1) * GAP
    x_arrow2 = x_gbuf + gbuf_w + 12
    x_res = x_arrow2 + 110
    res_w = 2 * RESULT_W + GAP
    W = x_res + res_w + 16
    H = RESULT_H + LABEL_H + 44

    y_mid = 20 + RESULT_H // 2
    y_gbuf = y_mid - PANEL_H // 2
    y_res = y_mid - RESULT_H // 2

    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)

    # input box
    d.rounded_rectangle([x_box, y_mid - box_h // 2, x_box + box_w, y_mid + box_h // 2],
                        radius=8, outline=(170, 170, 190), fill=(247, 247, 252), width=2)
    fb, fs = _font(FONT_BOLD, 14), _font(FONT, 10)
    d.text((x_box + (box_w - d.textlength(".zprj", font=fb)) / 2, y_mid - 16), ".zprj", fill=INK, font=fb)
    d.text((x_box + (box_w - d.textlength("Garment File", font=fs)) / 2, y_mid + 2),
           "Garment File", fill=MUTED, font=fs)

    draw_arrow(d, x_arrow1, x_gbuf - 6, y_mid, "VTK", "rasterizer")
    draw_arrow(d, x_arrow2, x_res - 6, y_mid, "Diffusion", "Renderer")

    # G-buffer panels
    for i, (img, name) in enumerate(zip(gbuf, GBUFFERS)):
        draw_panel(base, panel(img, (PANEL_W, PANEL_H)), x_gbuf + i * (PANEL_W + GAP),
                   y_gbuf, (PANEL_W, PANEL_H), name)
    fg = _font(FONT_BOLD, 12)
    d.text((x_gbuf + (gbuf_w - d.textlength("G-Buffers", font=fg)) / 2,
            y_gbuf + PANEL_H + LABEL_H + 12), "G-Buffers", fill=INK, font=fg)

    # static result panel (still)
    draw_panel(base, panel(still, (RESULT_W, RESULT_H)), x_res, y_res,
               (RESULT_W, RESULT_H), "still", bold=True)

    # animated result panel (turntable)
    x_turn = x_res + RESULT_W + GAP
    frames = []
    for f in turn:
        frame = base.copy()
        draw_panel(frame, panel(f, (RESULT_W, RESULT_H)), x_turn, y_res,
                   (RESULT_W, RESULT_H), "turntable", bold=True)
        frames.append(frame)

    png_path = os.path.join(args.output, "pipeline_diagram.png")
    gif_path = os.path.join(args.output, "pipeline_diagram.gif")
    frames[len(frames) // 8].save(png_path)  # off-front view so the two panels differ
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=1000 // args.fps, loop=0, optimize=True)
    print(f"Saved {png_path}\nSaved {gif_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
