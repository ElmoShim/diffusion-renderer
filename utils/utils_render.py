"""Shared rendering utilities for zprj G-buffer and diffusion rendering."""

import io
import math
import os

import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image


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

def load_mesh(scene):
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

    # ── Assemble ─────────────────────────────────────────────────────
    textures = {k: tex_to_tensor(v, 3 if k not in ("roughness", "metallic") else 1)
                for k, v in tex_bytes.items()}
    return assemble_mesh(col, textures, normal_intensity, garment_face_count)


# ── G-buffer rendering ───────────────────────────────────────────────

def _sample_or_interp(tex, scalar, uv, rast, tri, mask, dev, tex_mask=None):
    attr = scalar if scalar.ndim == 2 else scalar.unsqueeze(-1)
    if tex is not None:
        tex_out = dr.texture(tex.to(dev), uv, filter_mode="linear")
        if tex_out.shape[-1] == 1:
            tex_out = tex_out.expand(-1, -1, -1, 3)
        if tex_mask is not None:
            interp_out, _ = dr.interpolate(attr[None, ...], rast, tri)
            if interp_out.shape[-1] == 1:
                interp_out = interp_out.expand(-1, -1, -1, 3)
            return (tex_out * tex_mask + interp_out * (1 - tex_mask)) * mask
        return tex_out * mask
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

    # Texture mask: apply textures only to garment faces (first N faces)
    garment_fc = mesh.get("garment_face_count")
    tex_mask = None
    if garment_fc is not None and garment_fc < tri.shape[0]:
        tri_id = rast[..., 3:4]  # 1-indexed triangle ID
        tex_mask = ((tri_id > 0) & (tri_id <= garment_fc)).float()

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
        perturbed = (t[..., 0:1] * T + t[..., 1:2] * B + t[..., 2:3] * N)
        perturbed = (perturbed / perturbed.norm(dim=-1, keepdim=True).clamp(min=1e-8)).unsqueeze(0)
        if tex_mask is not None:
            ni = perturbed * tex_mask + ni * (1 - tex_mask)
        else:
            ni = perturbed
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
    gb["basecolor"] = _sample_or_interp(textures["diffuse"], bc_t, uv_s, rast, tri, mask, device, tex_mask)[0]
    gb["roughness"] = _sample_or_interp(textures["roughness"], ro_t, uv_s, rast, tri, mask, device, tex_mask)[0]
    gb["metallic"] = _sample_or_interp(textures["metallic"], me_t, uv_s, rast, tri, mask, device, tex_mask)[0]

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
