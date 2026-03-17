"""Render a .zprj garment with the diffusion renderer.

Usage:
    python render_zprj.py samples/garment.zprj
    python render_zprj.py samples/garment.zprj --mode turntable
    python render_zprj.py samples/garment.zprj --mode rotate-light
    python render_zprj.py samples/garment.zprj --mode still --hdr examples/hdri/pink_sunrise_1k.hdr
"""

import argparse
import io
import math
import os
import zipfile
from contextlib import nullcontext

import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image


# ── zprj loading ─────────────────────────────────────────────────────

def load_images_from_zprj(zprj_path: str) -> dict[str, bytes]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".tiff", ".tif", ".hdr", ".exr", ".dds"}
    images: dict[str, bytes] = {}
    with zipfile.ZipFile(zprj_path, "r") as zf:
        for name in zf.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext in image_exts:
                images[os.path.basename(name)] = zf.read(name)
            if ext == ".zpac":
                try:
                    with zipfile.ZipFile(io.BytesIO(zf.read(name)), "r") as zpac:
                        for inner in zpac.namelist():
                            iext = os.path.splitext(inner)[1].lower()
                            if iext in image_exts:
                                images[os.path.basename(inner)] = zpac.read(inner)
                except (zipfile.BadZipFile, KeyError):
                    pass
    return images


def _resolve_tex(path: str, cache: dict[str, bytes]) -> bytes | None:
    if not path:
        return None
    return cache.get(path) or cache.get(os.path.basename(path))


def _tex_to_tensor(data: bytes | None, ch: int = 3) -> torch.Tensor | None:
    if data is None:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB" if ch == 3 else "L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W,C)
    except Exception:
        return None


def _apply_uv_transform(uvs: np.ndarray, mat) -> np.ndarray:
    result = uvs.copy()
    tw = mat.tile_width if mat.tile_width > 0 else 1.0
    th = mat.tile_height if mat.tile_height > 0 else 1.0
    result[:, 0] /= tw
    result[:, 1] /= th
    xf = mat.diffuse_texture_transform
    angle = getattr(xf, "rotation", 0.0)
    if angle:
        r = math.radians(angle)
        c, s = math.cos(r), math.sin(r)
        u, v = result[:, 0].copy(), result[:, 1].copy()
        result[:, 0] = u * c - v * s
        result[:, 1] = u * s + v * c
    ou, ov = getattr(xf, "offset_u", 0.0), getattr(xf, "offset_v", 0.0)
    if ou or ov:
        result[:, 0] += ou
        result[:, 1] += ov
    return result


# ── mesh ─────────────────────────────────────────────────────────────

def _find_substance_tex(image_cache, mat, kind, colorway_idx=None):
    """Find substance-generated DDS texture matching the material name and kind.

    DDS files follow the pattern: output_<NAME>_<kind>[<id>].dds
    Multiple colorways produce multiple files of the same kind, sorted by name
    and selected by colorway_idx.
    """
    keywords = [kw.lower() for kw in mat.fabric_name.split()]
    candidates = []
    for name, data in sorted(image_cache.items()):
        if kind not in name.lower() or not name.lower().endswith(".dds"):
            continue
        if any(kw in name.lower() for kw in keywords):
            candidates.append((name, data))
    # Fallback: any DDS with the right kind
    if not candidates:
        for name, data in sorted(image_cache.items()):
            if kind in name.lower() and name.lower().endswith(".dds"):
                candidates.append((name, data))
    if not candidates:
        return None
    pick = (colorway_idx if colorway_idx is not None else 0) % len(candidates)
    chosen = candidates[pick][0]
    print(f"  Substance {kind}: {chosen} (colorway {pick}/{len(candidates)})")
    return candidates[pick][1]


def load_mesh(scene, image_cache, colorway_idx=None):
    materials = list(scene.fabric_materials)

    # Determine colorway index
    if colorway_idx is None and scene.colorways:
        colorway_idx = scene.active_colorway_index
    if scene.colorways:
        print(f"  Colorways: {[cw.name for cw in scene.colorways]}, using index {colorway_idx}")

    all_pos, all_faces, all_uvs = [], [], []
    all_bc, all_rough, all_metal = [], [], []
    voff = 0
    tex_bytes = {"diffuse": None, "normal": None, "roughness": None, "metallic": None}
    normal_intensity = 1.0

    for pat in scene.garment_patterns:
        nv, nf = pat.vertex_count, pat.triangle_count
        if nv == 0 or nf == 0:
            continue
        v = np.array(pat.positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(pat.indices, dtype=np.int32).reshape(nf, 3)

        raw_uv = np.array(pat.uvs)
        uv = raw_uv.astype(np.float32).reshape(nv, 2) if (pat.uv_vertex_count == nv and raw_uv.size == nv * 2) else np.zeros((nv, 2), dtype=np.float32)

        mi = pat.material_index
        if mi < 0 and materials:
            mi = 0
        mat = materials[mi] if 0 <= mi < len(materials) else None

        if mat and uv is not None:
            uv = _apply_uv_transform(uv, mat)

        if mat:
            dc = np.array(mat.diffuse_color, dtype=np.float32)
            bc = np.tile(dc[:3], (nv, 1)) if dc.size >= 3 else np.ones((nv, 3), dtype=np.float32)
            ro = np.full(nv, mat.roughness if mat.use_metalness_roughness_pbr else 0.5, dtype=np.float32)
            me = np.full(nv, mat.metalness if mat.use_metalness_roughness_pbr else 0.0, dtype=np.float32)

            # Resolve textures: try material path first, then substance DDS
            for kind, mat_path in [("diffuse", mat.diffuse_texture_path),
                                   ("normal", mat.normal_texture_path),
                                   ("roughness", mat.roughness_texture_path),
                                   ("metallic", mat.metalness_texture_path)]:
                if tex_bytes[kind] is None:
                    tex_bytes[kind] = _resolve_tex(mat_path, image_cache)
                    if tex_bytes[kind] is None:
                        sub_kind = "basecolor" if kind == "diffuse" else kind
                        tex_bytes[kind] = _find_substance_tex(image_cache, mat, sub_kind, colorway_idx)
            if tex_bytes["normal"]:
                normal_intensity = mat.normal_intensity_percent / 100.0 if mat.normal_intensity_percent > 0 else 1.0
        else:
            bc = np.full((nv, 3), 0.5, dtype=np.float32)
            ro = np.full(nv, 0.5, dtype=np.float32)
            me = np.full(nv, 0.0, dtype=np.float32)

        all_pos.append(v)
        all_faces.append(f + voff)
        all_uvs.append(uv)
        all_bc.append(bc)
        all_rough.append(ro)
        all_metal.append(me)
        voff += nv

    textures = {k: _tex_to_tensor(v, 3 if k != "roughness" and k != "metallic" else 1) for k, v in tex_bytes.items()}
    return {
        "positions": np.concatenate(all_pos), "faces": np.concatenate(all_faces),
        "uvs": np.concatenate(all_uvs), "basecolors": np.concatenate(all_bc),
        "roughness": np.concatenate(all_rough), "metallic": np.concatenate(all_metal),
        "textures": textures, "normal_intensity": normal_intensity,
    }


# ── geometry utils ───────────────────────────────────────────────────

def compute_vertex_normals(pos, faces):
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(pos)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
    return vn


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -r @ eye; m[1, 3] = -u @ eye; m[2, 3] = f @ eye
    return m


def perspective(fov_deg, near, far):
    t = math.tan(math.radians(fov_deg) / 2.0)
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = 1.0 / t; p[1, 1] = 1.0 / t
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2.0 * far * near / (far - near)
    p[3, 2] = -1.0
    return p


def auto_camera(positions, fov_deg=15.0, azimuth_deg=0.0):
    """Camera that orbits around the mesh center at the given azimuth angle."""
    center = (positions.max(0) + positions.min(0)) / 2.0
    ext = (positions.max(0) - positions.min(0)).max()
    dist = (ext / 2.0) / math.tan(math.radians(fov_deg) / 2.0) * 1.2
    az = math.radians(azimuth_deg)
    eye = center + np.array([math.sin(az) * dist, 0, math.cos(az) * dist], dtype=np.float32)
    up = np.array([0, -1, 0], dtype=np.float32)
    near = max(dist - ext, 0.1)
    far = dist + ext
    view = look_at(eye, center, up)
    proj = perspective(fov_deg, near, far)
    return proj @ view, view


# ── G-buffer rendering ───────────────────────────────────────────────

def _sample_or_interp(tex, scalar, uv, rast, tri, mask, dev):
    if tex is not None:
        out = dr.texture(tex.to(dev), uv, filter_mode="linear")
        if out.shape[-1] == 1:
            out = out.expand(-1, -1, -1, 3)
        return out * mask
    attr = scalar if scalar.ndim == 2 else scalar.unsqueeze(-1)
    out, _ = dr.interpolate(attr[None, ...], rast, tri)
    if out.shape[-1] == 1:
        out = out.expand(-1, -1, -1, 3)
    return out * mask


def render_gbuffers(mesh, resolution=512, fov_deg=15.0, azimuth_deg=0.0, device="cuda",
                    _precomp=None):
    """Render G-buffers for a single viewpoint.

    Returns dict of {name: (H,W,3) float32 tensor in [0,1]}.
    """
    pos_np = mesh["positions"]
    faces_np = mesh["faces"]
    textures = mesh["textures"]

    mvp, view = auto_camera(pos_np, fov_deg, azimuth_deg)
    mvp_t = torch.from_numpy(mvp).to(device)
    view_t = torch.from_numpy(view).to(device)

    if _precomp is not None:
        pos, tri, nrm = _precomp["pos"], _precomp["tri"], _precomp["nrm"]
        uv_t, bc_t = _precomp["uv"], _precomp["bc"]
        ro_t, me_t = _precomp["ro"], _precomp["me"]
        pos_h, glctx = _precomp["pos_h"], _precomp["glctx"]
    else:
        pos = torch.from_numpy(pos_np).to(device)
        tri = torch.from_numpy(faces_np.astype(np.int32)).to(device)
        nrm = torch.from_numpy(compute_vertex_normals(pos_np, faces_np)).float().to(device)
        uv_t = torch.from_numpy(mesh["uvs"]).float().to(device)
        bc_t = torch.from_numpy(mesh["basecolors"]).float().to(device)
        ro_t = torch.from_numpy(mesh["roughness"]).float().to(device)
        me_t = torch.from_numpy(mesh["metallic"]).float().to(device)
        pos_h = torch.cat([pos, torch.ones(pos.shape[0], 1, device=device)], 1)
        glctx = dr.RasterizeCudaContext()

    clip = (pos_h @ mvp_t.T).unsqueeze(0)
    rast, _ = dr.rasterize(glctx, clip, tri, resolution=[resolution, resolution])
    mask = (rast[..., 3:4] > 0).float()

    # Interpolate UVs — flip V, frac for tiling
    uv_interp, _ = dr.interpolate(uv_t[None, ...], rast, tri)
    uv_s = uv_interp.clone()
    uv_s[..., 1] = 1.0 - uv_s[..., 1]
    uv_s = uv_s.frac()

    gb = {}

    # Normal
    ni, _ = dr.interpolate(nrm[None, ...], rast, tri)
    ni = ni / ni.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    if textures["normal"] is not None:
        nt = dr.texture(textures["normal"].to(device), uv_s, filter_mode="linear")
        tn = nt * 2.0 - 1.0
        tn[..., :2] *= mesh["normal_intensity"]
        tn = tn / tn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        N = ni[0]
        up_vec = torch.tensor([0., 1., 0.], device=device)
        T = torch.cross(up_vec.expand_as(N), N, dim=-1)
        small = T.norm(dim=-1, keepdim=True) < 1e-6
        fb = torch.cross(torch.tensor([1., 0., 0.], device=device).expand_as(N), N, dim=-1)
        T[small.expand_as(T)] = fb[small.expand_as(fb)]
        T = T / T.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        B = torch.cross(N, T, dim=-1)
        t = tn[0]
        ni = (t[..., 0:1] * T + t[..., 1:2] * B + t[..., 2:3] * N)
        ni = (ni / ni.norm(dim=-1, keepdim=True).clamp(min=1e-8)).unsqueeze(0)
    gb["normal"] = ((ni * 0.5 + 0.5) * mask)[0]

    # Depth
    cam_z = -(pos_h @ view_t.T)[:, 2:3]
    cz, _ = dr.interpolate(cam_z[None, ...], rast, tri)
    valid = cz[mask.bool().expand_as(cz)]
    if valid.numel() > 0:
        dn = (cz - valid.min()) / (valid.max() - valid.min() + 1e-8)
    else:
        dn = cz
    gb["depth"] = (dn.clamp(0, 1) * mask + (1 - mask)).expand(-1, -1, -1, 3)[0]

    # Basecolor, Roughness, Metallic
    gb["basecolor"] = _sample_or_interp(textures["diffuse"], bc_t, uv_s, rast, tri, mask, device)[0]
    gb["roughness"] = _sample_or_interp(textures["roughness"], ro_t, uv_s, rast, tri, mask, device)[0]
    gb["metallic"] = _sample_or_interp(textures["metallic"], me_t, uv_s, rast, tri, mask, device)[0]

    return gb


def precompute_mesh_gpu(mesh, device="cuda"):
    """Upload mesh data to GPU once, reuse across multiple render_gbuffers calls."""
    pos = torch.from_numpy(mesh["positions"]).to(device)
    tri = torch.from_numpy(mesh["faces"].astype(np.int32)).to(device)
    nrm = torch.from_numpy(compute_vertex_normals(mesh["positions"], mesh["faces"])).float().to(device)
    return {
        "pos": pos, "tri": tri, "nrm": nrm,
        "uv": torch.from_numpy(mesh["uvs"]).float().to(device),
        "bc": torch.from_numpy(mesh["basecolors"]).float().to(device),
        "ro": torch.from_numpy(mesh["roughness"]).float().to(device),
        "me": torch.from_numpy(mesh["metallic"]).float().to(device),
        "pos_h": torch.cat([pos, torch.ones(pos.shape[0], 1, device=device)], 1),
        "glctx": dr.RasterizeCudaContext(),
    }


# ── forward rendering ────────────────────────────────────────────────

def forward_render(gbuffers_list, hdr_path, device="cuda", rotate_light=False):
    """Run diffusion forward rendering.

    Args:
        gbuffers_list: single dict {name: (H,W,3)} or list of 24 dicts for multi-frame.
        hdr_path: path to HDR environment map.
        rotate_light: if True, rotate environment light 360 degrees over 24 frames.

    Returns: list of PIL Images (length = n_frames).
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

    # Run inference (single pass for all frames)
    print(f"Running forward rendering (rotate_light={rotate_light})...")
    generator = torch.Generator(device=device).manual_seed(cfg.get("seed", 0))
    autocast_ctx = torch.autocast(device, enabled=True) if device == "cuda" else nullcontext()

    h, w = gbuffers_list[0]["basecolor"].shape[:2]
    with autocast_ctx:
        frames = pipeline(
            cond_images, cond_labels,
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
        ).frames[0]  # list of PIL Images

    return frames


# ── save ─────────────────────────────────────────────────────────────

def save_tensor_as_png(tensor, path):
    img = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(path)


# ── main ─────────────────────────────────────────────────────────────

def save_video(frames, path, fps=10):
    """Save a list of PIL Images as H.264 MP4."""
    import imageio
    frames_np = [np.asarray(f) for f in frames]
    imageio.mimsave(path, frames_np, fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"])


def main():
    parser = argparse.ArgumentParser(description="Render a .zprj garment with the diffusion renderer.")
    parser.add_argument("input", type=str, help="Path to .zprj file")
    parser.add_argument("--hdr", type=str, default="examples/hdri/sunny_vondelpark_1k.hdr", help="HDR environment map")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: tmp/<stem>/)")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")
    parser.add_argument("--fov", type=float, default=15.0, help="Camera FOV in degrees")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--colorway", type=int, default=None, help="Colorway index (default: active colorway)")
    parser.add_argument("--gbuffer-only", action="store_true", help="Only render G-buffers, skip forward rendering")
    parser.add_argument("--mode", choices=["still", "turntable", "rotate-light"], default="still",
                        help="still: single image | turntable: camera orbits 360° | rotate-light: light rotates 360°")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
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
    image_cache = load_images_from_zprj(os.path.abspath(args.input))
    mesh = load_mesh(scene, image_cache, colorway_idx=args.colorway)
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

    frames[0].save(os.path.join(out_dir, "rendered.png"))
    if args.mode != "still":
        video_path = os.path.join(out_dir, f"{args.mode}.mp4")
        save_video(frames, video_path, fps=args.fps)
        print(f"Video saved to {video_path}")
    else:
        print(f"Rendered image saved to {os.path.join(out_dir, 'rendered.png')}")


if __name__ == "__main__":
    main()
