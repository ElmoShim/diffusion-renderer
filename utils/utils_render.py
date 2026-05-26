"""Shared rendering utilities for zprj G-buffer and diffusion rendering."""

import io
import math
import os

import numpy as np
import torch
from PIL import Image

from utils.utils_render_vtk import (
    render_gbuffers,
    build_scene_actors,
)


# ── Color helpers ─────────────────────────────────────────────────────

def _normalize_color(color_arr):
    """Normalize color to [0,1]. CLO may store diffuse_color as [0,255] floats."""
    if color_arr.size >= 3 and color_arr[:3].max() > 1.0:
        color_arr = color_arr.copy()
        color_arr[:3] /= 255.0
    return color_arr


# ── Texture helpers ───────────────────────────────────────────────────

def tex_to_tensor(data, ch=3):
    """Convert image bytes to (1,H,W,C) float32 tensor."""
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB" if ch == 3 else "L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return None


def read_tex(scene, path):
    """Read texture bytes from scene, trying full path then basename."""
    if not path:
        return None
    data = scene.read_file(path)
    if not data:
        data = scene.read_file(os.path.basename(path))
    return data or None


def bake_texture_to_verts(scene, tex_path, uvs, nv):
    """Sample a texture at per-vertex UV coords, returning (nv, 3) RGB in [0,1]."""
    data = read_tex(scene, tex_path)
    if data is None:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
        uv = np.array(uvs, dtype=np.float32).reshape(nv, 2)
        # Flip V (OpenGL convention) and wrap to [0,1]
        u = np.mod(uv[:, 0], 1.0)
        v = np.mod(1.0 - uv[:, 1], 1.0)
        px = np.clip((u * w).astype(int), 0, w - 1)
        py = np.clip((v * h).astype(int), 0, h - 1)
        return arr[py, px, :3].copy()
    except Exception:
        return None


def find_substance_tex(scene, mat, kind, colorway_idx=None):
    """Find substance-generated DDS texture by material name and kind."""
    keywords = [kw.lower() for kw in mat.fabric_name.split()]
    all_files = scene.list_files()

    candidates = []
    for name in sorted(all_files):
        basename = os.path.basename(name).lower()
        if kind not in basename or not basename.endswith(".dds"):
            continue
        if any(kw in basename for kw in keywords):
            candidates.append(name)

    if not candidates:
        for name in sorted(all_files):
            basename = os.path.basename(name).lower()
            if kind in basename and basename.endswith(".dds"):
                candidates.append(name)

    if not candidates:
        return None

    pick = (colorway_idx if colorway_idx is not None else 0) % len(candidates)
    chosen = candidates[pick]
    print(f"  Substance {kind}: {os.path.basename(chosen)} (colorway {pick}/{len(candidates)})")
    return scene.read_file(chosen)


def apply_uv_transform(uvs, mat):
    """Transform pattern UVs for texture tiling."""
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


# ── Geometry helpers ──────────────────────────────────────────────────

def compute_vertex_normals(pos, faces):
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(pos)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
    return vn


def apply_transform(positions, mat4x4):
    """Apply 4x4 transform (column-major from parser) to Nx3 positions."""
    M = mat4x4.T
    pos_h = np.hstack([positions, np.ones((len(positions), 1), dtype=np.float32)])
    return (M @ pos_h.T).T[:, :3].astype(np.float32)


def is_identity(mat):
    return mat is None or np.allclose(mat, np.eye(4), atol=1e-6)


# ── Camera ────────────────────────────────────────────────────────────

def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -r @ eye; m[1, 3] = -u @ eye; m[2, 3] = f @ eye
    return m


def perspective(fov_deg, near, far, aspect=1.0):
    t = math.tan(math.radians(fov_deg) / 2.0)
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = 1.0 / (t * aspect)
    p[1, 1] = 1.0 / t
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2.0 * far * near / (far - near)
    p[3, 2] = -1.0
    return p


def auto_camera(positions, fov_deg=15.0, azimuth_deg=0.0, aspect=1.0):
    """Camera that orbits around the mesh center at the given azimuth angle."""
    center = (positions.max(0) + positions.min(0)) / 2.0
    span = positions.max(0) - positions.min(0)
    half_fov = math.radians(fov_deg) / 2.0
    dist_y = (span[1] / 2.0) / math.tan(half_fov)
    dist_x = (span[0] / 2.0) / math.tan(half_fov * aspect)
    dist = max(dist_y, dist_x) * 1.05
    az = math.radians(azimuth_deg)
    eye = center + np.array([math.sin(az) * dist, 0, math.cos(az) * dist], dtype=np.float32)
    up = np.array([0, -1, 0], dtype=np.float32)
    near = max(dist - span.max(), 0.1)
    far = dist * 2.5
    view = look_at(eye, center, up)
    proj = perspective(fov_deg, near, far, aspect=aspect)
    return proj @ view, view, eye


# ── Material helpers ──────────────────────────────────────────────────

def mat_pbr(mat, nv, default_bc=(0.5, 0.5, 0.5), default_ro=0.5, default_me=0.0):
    """Extract per-vertex basecolor, roughness, metallic from a FabricMaterial."""
    try:
        if mat:
            dc = _normalize_color(np.array(mat.diffuse_color, dtype=np.float32))
            bc = np.tile(dc[:3], (nv, 1)) if dc.size >= 3 else np.full((nv, 3), default_bc, dtype=np.float32)
            use_pbr = getattr(mat, "use_metalness_roughness_pbr", False)
            ro = np.full(nv, mat.roughness if use_pbr else default_ro, dtype=np.float32)
            me = np.full(nv, mat.metalness if use_pbr else default_me, dtype=np.float32)
            return bc, ro, me
    except Exception:
        pass
    return (np.full((nv, 3), default_bc, dtype=np.float32),
            np.full(nv, default_ro, dtype=np.float32),
            np.full(nv, default_me, dtype=np.float32))


# ── Mesh collector ────────────────────────────────────────────────────

def new_mesh_collector():
    return {"pos": [], "faces": [], "uvs": [], "bc": [], "ro": [], "me": [], "voff": 0}


def append_mesh(collector, positions, faces, nv, bc, ro, me, uvs=None):
    """Append geometry + attributes to the running mesh collector."""
    if uvs is None:
        uvs = np.zeros((nv, 2), dtype=np.float32)
    collector["pos"].append(positions)
    collector["faces"].append(faces + collector["voff"])
    collector["uvs"].append(uvs)
    collector["bc"].append(bc)
    collector["ro"].append(ro)
    collector["me"].append(me)
    collector["voff"] += nv


def assemble_mesh(collector, textures, normal_intensity=1.0, garment_face_count=None):
    """Finalize a mesh collector into the mesh dict expected by render_gbuffers."""
    result = {
        "positions": np.concatenate(collector["pos"]),
        "faces": np.concatenate(collector["faces"]),
        "uvs": np.concatenate(collector["uvs"]),
        "basecolors": np.concatenate(collector["bc"]),
        "roughness": np.concatenate(collector["ro"]),
        "metallic": np.concatenate(collector["me"]),
        "textures": textures,
        "normal_intensity": normal_intensity,
    }
    if garment_face_count is not None:
        result["garment_face_count"] = garment_face_count
    return result


# ── Mesh loading ──────────────────────────────────────────────────────

def load_mesh(scene, background=True):
    """Load all scene meshes (garment, avatar, trim, button, zipper) with PBR materials.

    Uses zprj_loader v0.2.0 API (scene.read_file()) for embedded texture access.
    """
    materials = list(scene.fabric_materials)

    # Active colorway
    colorway_idx = scene.active_colorway_index if scene.colorways else None
    colorway = (scene.colorways[colorway_idx]
                if colorway_idx is not None and 0 <= colorway_idx < len(scene.colorways)
                else None)
    if scene.colorways:
        print(f"  Colorways: {[cw.name for cw in scene.colorways]}, using index {colorway_idx}")

    col = new_mesh_collector()
    tex_bytes = {"diffuse": None, "normal": None, "roughness": None, "metallic": None}
    normal_intensity = 1.0

    # ── Resolve garment textures from main material ───────────────────
    main_mat = None
    if scene.garment_patterns and materials:
        mi = scene.garment_patterns[0].material_index
        if colorway and len(colorway.pattern_fabric_indices) > 0:
            mi = colorway.pattern_fabric_indices[0]
        if mi < 0:
            mi = 0
        if 0 <= mi < len(materials):
            main_mat = materials[mi]

    if main_mat:
        for kind, mat_path in [("diffuse", main_mat.diffuse_texture_path),
                               ("normal", main_mat.normal_texture_path),
                               ("roughness", main_mat.roughness_texture_path),
                               ("metallic", main_mat.metalness_texture_path)]:
            data = read_tex(scene, mat_path)
            if data:
                tex_bytes[kind] = data
            else:
                sub_kind = "basecolor" if kind == "diffuse" else kind
                sub_data = find_substance_tex(scene, main_mat, sub_kind, colorway_idx)
                if sub_data:
                    tex_bytes[kind] = sub_data
        if tex_bytes["normal"]:
            nip = main_mat.normal_intensity_percent
            normal_intensity = nip / 100.0 if nip > 0 else 1.0

    # ── 1. Garment patterns ───────────────────────────────────────────
    pat_count = 0
    for i, pat in enumerate(scene.garment_patterns):
        nv, nf = pat.vertex_count, pat.triangle_count
        if nv == 0 or nf == 0:
            continue
        v = np.array(pat.positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(pat.indices, dtype=np.int32).reshape(nf, 3)

        raw_uv = np.array(pat.uvs)
        uv = (raw_uv.astype(np.float32).reshape(nv, 2)
              if (pat.uv_vertex_count == nv and raw_uv.size == nv * 2)
              else np.zeros((nv, 2), dtype=np.float32))

        # Per-pattern material via colorway
        mi = pat.material_index
        if colorway and i < len(colorway.pattern_fabric_indices):
            mi = colorway.pattern_fabric_indices[i]
        if mi < 0 and materials:
            mi = 0
        mat = materials[mi] if 0 <= mi < len(materials) else None

        if mat:
            dc = _normalize_color(np.array(mat.diffuse_color, dtype=np.float32))
            bc = np.tile(dc[:3], (nv, 1)) if dc.size >= 3 else np.ones((nv, 3), dtype=np.float32)
            ro = np.full(nv, mat.roughness if mat.use_metalness_roughness_pbr else 0.5, dtype=np.float32)
            me = np.full(nv, mat.metalness if mat.use_metalness_roughness_pbr else 0.0, dtype=np.float32)
            uv = apply_uv_transform(uv, mat)
        else:
            bc, ro, me = mat_pbr(None, nv)

        append_mesh(col, v, f, nv, bc, ro, me, uv)
        pat_count += 1

    garment_face_count = sum(f.shape[0] for f in col["faces"])
    if pat_count:
        print(f"  Garment patterns: {pat_count}")

    # ── 2. Avatar meshes ──────────────────────────────────────────────
    avatar_count = 0
    for mesh in scene.avatar_meshes:
        if mesh.vertex_count == 0 or mesh.triangle_count == 0:
            continue

        mat = mesh.material if mesh.has_material else None
        if mat:
            try:
                alpha = float(mat.diffuse_color[3]) if len(mat.diffuse_color) > 3 else 1.0
                if alpha < 0.01:
                    continue
            except Exception:
                pass

        nv = mesh.vertex_count
        v = np.array(mesh.positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(mesh.indices, dtype=np.int32).reshape(mesh.triangle_count, 3)

        wm = mesh.world_matrix
        if not is_identity(wm):
            v = apply_transform(v, wm)

        if mesh.vertex_colors and len(mesh.vertex_colors) >= nv * 3:
            vc = np.array(mesh.vertex_colors, dtype=np.float32)
            n_comp = len(vc) // nv
            bc = vc.reshape(nv, n_comp)[:, :3].copy()
            if bc.max() > 1.0:
                bc /= 255.0
            _, ro, me = mat_pbr(mat, nv, default_ro=0.5, default_me=0.0)
        else:
            bc, ro, me = mat_pbr(mat, nv, default_bc=(0.85, 0.75, 0.65))
            # Bake per-mesh texture into vertex basecolor when available
            if mat and mat.diffuse_texture_path and mesh.uvs and len(mesh.uvs) >= nv * 2:
                baked = bake_texture_to_verts(scene, mat.diffuse_texture_path, mesh.uvs, nv)
                if baked is not None:
                    bc = baked

        append_mesh(col, v, f, nv, bc, ro, me)
        avatar_count += 1
    if avatar_count:
        print(f"  Avatar meshes: {avatar_count}")

    # ── 3. Trim objects ───────────────────────────────────────────────
    trim_count = 0
    for trim in scene.trim_objects:
        if trim.mesh_vertex_count == 0 or trim.mesh_triangle_count == 0:
            continue
        if not trim.visible:
            continue

        nv = trim.mesh_vertex_count
        v = np.array(trim.mesh_positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(trim.mesh_indices, dtype=np.int32).reshape(trim.mesh_triangle_count, 3)

        tm = trim.transform_matrix
        if not is_identity(tm):
            v = apply_transform(v, tm)

        mat = trim.colorway_material
        bc, ro, me = mat_pbr(mat, nv, default_bc=(0.6, 0.6, 0.7))

        append_mesh(col, v, f, nv, bc, ro, me)
        trim_count += 1
    if trim_count:
        print(f"  Trim objects: {trim_count}")

    # ── 4. Zipper teeth ──────────────────────────────────────────────
    zip_count = 0
    for zi in scene.zipper_instances:
        if zi.teeth_vertex_count == 0 or zi.teeth_triangle_count == 0:
            continue

        nv = zi.teeth_vertex_count
        v = np.array(zi.teeth_positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(zi.teeth_indices, dtype=np.int32).reshape(zi.teeth_triangle_count, 3)

        if zi.has_transform:
            tm = zi.transform
            if not is_identity(tm):
                v = apply_transform(v, tm)

        mat = zi.slider_material
        bc, ro, me = mat_pbr(mat, nv, default_bc=(0.8, 0.8, 0.3), default_ro=0.3, default_me=0.8)

        append_mesh(col, v, f, nv, bc, ro, me)
        zip_count += 1
    if zip_count:
        print(f"  Zipper teeth: {zip_count}")

    # ── 5. Button meshes ─────────────────────────────────────────────
    btn_count = 0
    for bs in scene.button_head_styles:
        if not bs.has_mesh_3d:
            continue
        if bs.mesh_vertex_count == 0 or bs.mesh_triangle_count == 0:
            continue

        nv = bs.mesh_vertex_count
        v = np.array(bs.mesh_positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(bs.mesh_indices, dtype=np.int32).reshape(bs.mesh_triangle_count, 3)

        mat = None
        try:
            if colorway_idx is not None and bs.colorway_materials and colorway_idx < len(bs.colorway_materials):
                mat = bs.colorway_materials[colorway_idx]
        except Exception:
            pass
        bc, ro, me = mat_pbr(mat, nv, default_bc=(0.9, 0.85, 0.7), default_ro=0.3, default_me=0.5)

        append_mesh(col, v, f, nv, bc, ro, me)
        btn_count += 1
    if btn_count:
        print(f"  Button meshes: {btn_count}")

    # ── 6. Background cylinder ──────────────────────────────────────
    orig_pos = np.concatenate(col["pos"])
    orig_bbox = (orig_pos.min(0), orig_pos.max(0))
    orig_face_count = sum(f.shape[0] for f in col["faces"])
    if background:
        _add_background_cylinder(col)

    # ── Assemble ─────────────────────────────────────────────────────
    textures = {k: tex_to_tensor(v, 3 if k not in ("roughness", "metallic") else 1)
                for k, v in tex_bytes.items()}
    m = assemble_mesh(col, textures, normal_intensity, garment_face_count)
    m["orig_bbox"] = orig_bbox
    m["orig_face_count"] = orig_face_count
    return m


def _add_background_cylinder(col, n_seg=64, fov_deg=15.0):
    """Add a closed cylinder (wall + floor disc) enclosing the current mesh.
    Radius is set larger than the camera distance so the camera sits inside."""
    all_pos = np.concatenate(col["pos"])
    bmin, bmax = all_pos.min(0), all_pos.max(0)
    center_xz = (bmin[[0, 2]] + bmax[[0, 2]]) / 2.0
    height = bmax[1] - bmin[1]

    span_x = bmax[0] - bmin[0]
    half_fov = math.radians(fov_deg) / 2.0
    cam_dist = max(height / 2.0 / math.tan(half_fov),
                   span_x / 2.0 / math.tan(half_fov)) * 1.05
    radius = cam_dist * 1.1
    y_floor = bmin[1]
    y_top = bmax[1] + height * 0.6
    cx, cz = center_xz

    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False, dtype=np.float32)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    # Wall: 2 rings of vertices (bottom + top), normals point inward
    wall_bot = np.stack([cx + radius * cos_a, np.full(n_seg, y_floor), cz + radius * sin_a], axis=1)
    wall_top = np.stack([cx + radius * cos_a, np.full(n_seg, y_top), cz + radius * sin_a], axis=1)
    wall_v = np.concatenate([wall_bot, wall_top]).astype(np.float32)

    wall_f = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        wall_f.append([i, j, j + n_seg])
        wall_f.append([i, j + n_seg, i + n_seg])
    wall_f = np.array(wall_f, np.int32)

    nv_wall = wall_v.shape[0]
    bc = np.full((nv_wall, 3), 0.5, np.float32)
    ro = np.full(nv_wall, 0.5, np.float32)
    me = np.zeros(nv_wall, np.float32)
    append_mesh(col, wall_v, wall_f, nv_wall, bc, ro, me)

    # Floor disc: center vertex + ring
    floor_center = np.array([[cx, y_floor, cz]], np.float32)
    floor_ring = np.stack([cx + radius * cos_a, np.full(n_seg, y_floor), cz + radius * sin_a], axis=1)
    floor_v = np.concatenate([floor_center, floor_ring]).astype(np.float32)

    floor_f = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        floor_f.append([0, j + 1, i + 1])
    floor_f = np.array(floor_f, np.int32)

    nv_floor = floor_v.shape[0]
    bc_f = np.full((nv_floor, 3), 0.5, np.float32)
    ro_f = np.full(nv_floor, 0.5, np.float32)
    me_f = np.zeros(nv_floor, np.float32)
    append_mesh(col, floor_v, floor_f, nv_floor, bc_f, ro_f, me_f)

    # Ceiling disc
    ceil_center = np.array([[cx, y_top, cz]], np.float32)
    ceil_ring = np.stack([cx + radius * cos_a, np.full(n_seg, y_top), cz + radius * sin_a], axis=1)
    ceil_v = np.concatenate([ceil_center, ceil_ring]).astype(np.float32)

    ceil_f = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        ceil_f.append([0, j + 1, i + 1])
    ceil_f = np.array(ceil_f, np.int32)

    nv_ceil = ceil_v.shape[0]
    bc_c = np.full((nv_ceil, 3), 0.5, np.float32)
    ro_c = np.full(nv_ceil, 0.5, np.float32)
    me_c = np.zeros(nv_ceil, np.float32)
    append_mesh(col, ceil_v, ceil_f, nv_ceil, bc_c, ro_c, me_c)


# ── Save helpers ──────────────────────────────────────────────────────

def save_tensor_as_png(tensor, path):
    img = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(path)


def save_video(frames, path, fps=10):
    """Save a list of PIL Images as H.264 MP4."""
    import imageio
    frames_np = [np.asarray(f) for f in frames]
    imageio.mimsave(path, frames_np, fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"])
