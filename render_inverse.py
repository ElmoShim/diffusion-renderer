"""Inverse-render: estimate G-buffers from a single image.

Usage:
    python render_inverse.py photo.png
    python render_inverse.py photo.png --output results/
    python render_inverse.py photo.png --resolution 512 --seed 42
"""

import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
from utils.utils_rgbx import convert_rgba_to_rgb_pil
from utils.utils_rgbx_inference import resize_upscale_without_padding

MODEL_PASSES = ["basecolor", "metallic", "roughness", "normal", "depth"]
DEFAULT_WEIGHTS = "./checkpoints/diffusion_renderer-inverse-svd"
DEFAULT_N_FRAMES = 24
DEFAULT_N_STEPS = 20


def load_pipeline(weights_path, device="cuda"):
    """Load the inverse rendering pipeline."""
    missing_kwargs = {}
    missing_kwargs["cond_mode"] = "skip"
    missing_kwargs["use_deterministic_mode"] = False

    model_weights_subfolders = os.listdir(weights_path) if os.path.exists(weights_path) else []
    if "image_encoder" not in model_weights_subfolders:
        missing_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder",
        )
    if "feature_extractor" not in model_weights_subfolders:
        missing_kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor",
        )

    print("Loading inverse rendering model...")
    pipeline = RGBXVideoDiffusionPipeline.from_pretrained(weights_path, **missing_kwargs)
    pipeline = pipeline.to(device)
    pipeline = pipeline.to(torch.float16)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def inverse_render(image_path, pipeline, device="cuda", resolution=512,
                   n_frames=DEFAULT_N_FRAMES, n_steps=DEFAULT_N_STEPS, seed=0):
    """Run inverse rendering on a single image.

    Args:
        image_path: path to input image.
        pipeline: loaded RGBXVideoDiffusionPipeline.
        device: torch device string.
        resolution: target resolution.
        n_frames: number of frames for the SVD model.
        n_steps: number of denoising steps.
        seed: random seed.

    Returns:
        Tuple of (results dict mapping pass name to PIL Image, resized input PIL Image).
    """
    # Load and preprocess input image
    input_image = Image.open(image_path)
    input_image = convert_rgba_to_rgb_pil(input_image, background_color=(0, 0, 0))

    width, height = input_image.size
    if width != resolution or height != resolution:
        input_image = resize_upscale_without_padding(input_image, resolution, resolution)
    width, height = input_image.size

    # Replicate to n_frames
    image_uint8 = np.asarray(input_image)
    input_images = np.stack([image_uint8] * n_frames, axis=0)[None, ...].astype(np.float32) / 255.0
    cond_images = {"rgb": input_images}
    cond_labels = {"rgb": "vae"}

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device, enabled=True)

    # Run each G-buffer pass
    results = {}
    for pass_name in MODEL_PASSES:
        print(f"  Estimating {pass_name}...")
        cond_images["input_context"] = pass_name
        generator = torch.Generator(device=device).manual_seed(seed)

        with autocast_ctx:
            frames = pipeline(
                cond_images, cond_labels,
                height=height, width=width,
                num_frames=n_frames,
                num_inference_steps=n_steps,
                min_guidance_scale=1.0,
                max_guidance_scale=1.0,
                fps=7,
                motion_bucket_id=127,
                noise_aug_strength=0,
                generator=generator,
                decode_chunk_size=8,
            ).frames[0]

        results[pass_name] = frames[0]

    return results, input_image


def main():
    parser = argparse.ArgumentParser(description="Inverse-render: estimate G-buffers from a single image.")
    parser.add_argument("image", type=str, help="Path to input image (PNG, JPG, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: tmp/<stem>_inverse/)")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Model weights directory")
    parser.add_argument("--resolution", type=int, default=512, help="Inference resolution")
    parser.add_argument("--steps", type=int, default=DEFAULT_N_STEPS, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu, mps)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        return

    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.output or f"tmp/{stem}_inverse/"
    os.makedirs(out_dir, exist_ok=True)

    pipeline = load_pipeline(args.weights, device=args.device)

    print(f"Inverse rendering {args.image} ...")
    results, input_image = inverse_render(
        args.image, pipeline,
        device=args.device, resolution=args.resolution,
        n_steps=args.steps, seed=args.seed,
    )

    # Save results
    input_image.save(os.path.join(out_dir, "input.png"))
    for name, img in results.items():
        save_path = os.path.join(out_dir, f"{name}.png")
        img.save(save_path)

    print(f"Results saved to {out_dir}")
    print(f"  input.png + {', '.join(f'{n}.png' for n in MODEL_PASSES)}")


if __name__ == "__main__":
    main()
