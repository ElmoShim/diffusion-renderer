"""VTK-based G-buffer rendering for zprj scenes.

Replaces the nvdiffrast-based renderer with VTK offscreen rendering.
Returns dict[str, torch.Tensor] with shape (H, W, 3) in [0, 1] — same
interface as the old render_gbuffers.
"""

import io
import math
import os

import numpy as np
import torch
import vtk
from vtk.util import numpy_support
from PIL import Image


# ── GLSL shaders ─────────────────────────────────────────────────────

NORMAL_VERT = """
//VTK::System::Dec
attribute vec4 vertexMC;
//VTK::PositionVC::Dec
//VTK::Normal::Dec
//VTK::Light::Dec
//VTK::TCoord::Dec
//VTK::Color::Dec
//VTK::Clip::Dec
//VTK::Camera::Dec
//VTK::PrimID::Dec
//VTK::Picking::Dec

out vec3 normalViewOut;

void main()
{
  //VTK::Color::Impl
  //VTK::Normal::Impl
  //VTK::TCoord::Impl
  //VTK::Clip::Impl
  //VTK::PrimID::Impl
  //VTK::PositionVC::Impl
  //VTK::Light::Impl
  //VTK::Picking::Impl

  normalViewOut = normalMatrix * normalMC;
}
"""

NORMAL_FRAG = """
//VTK::System::Dec
uniform int PrimitiveIDOffset;
//VTK::CustomUniforms::Dec
//VTK::PositionVC::Dec
//VTK::Camera::Dec
//VTK::Color::Dec
//VTK::Normal::Dec
//VTK::Light::Dec
//VTK::TMap::Dec
//VTK::TCoord::Dec
//VTK::Picking::Dec
//VTK::DepthPeeling::Dec
//VTK::Clip::Dec
//VTK::Output::Dec
//VTK::PrimID::Dec
//VTK::Coincident::Dec
//VTK::ValuePass::Dec
//VTK::Edges::Dec

in vec3 normalViewOut;

void main()
{
  //VTK::PositionVC::Impl
  //VTK::UniformFlow::Impl
  //VTK::Depth::Impl
  //VTK::DepthPeeling::PreColor
  //VTK::PrimID::Impl
  //VTK::Clip::Impl
  //VTK::ValuePass::Impl
  //VTK::Color::Impl
  //VTK::Edges::Impl
  //VTK::Normal::Impl
  //VTK::Light::Impl
  //VTK::TCoord::Impl

  vec3 n = normalize(normalViewOut);
  if (!gl_FrontFacing) n = -n;
  fragOutput0 = vec4(n * 0.5 + 0.5, 1.0);

  if (gl_FragData[0].a <= 0.0) discard;
  //VTK::DepthPeeling::Impl
  //VTK::Picking::Impl
  //VTK::Coincident::Impl
}
"""

_normal_sp = None

def _get_normal_shader_property():
    global _normal_sp
    if _normal_sp is None:
        _normal_sp = vtk.vtkShaderProperty()
        _normal_sp.SetVertexShaderCode(NORMAL_VERT)
        _normal_sp.SetFragmentShaderCode(NORMAL_FRAG)
    return _normal_sp


# ── Geometry helpers ─────────────────────────────────────────────────

def _make_polydata(vertices, faces, normals=None):
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices.astype(np.float64), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(3, numpy_support.numpy_to_vtk(faces.ravel(), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(cells)
    if normals is not None and len(normals) == len(vertices):
        n_arr = numpy_support.numpy_to_vtk(normals.astype(np.float64), deep=True)
        pd.GetPointData().SetNormals(n_arr)
    else:
        gen = vtk.vtkPolyDataNormals()
        gen.SetInputData(pd)
        gen.ComputePointNormalsOn()
        gen.Update()
        pd = gen.GetOutput()
    return pd


def _apply_matrix(polydata, mat4x4):
    mat4x4 = mat4x4.T
    m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            m.SetElement(i, j, float(mat4x4[i, j]))
    t = vtk.vtkTransform()
    t.SetMatrix(m)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(polydata)
    f.SetTransform(t)
    f.Update()
    return f.GetOutput()


def _is_identity(mat4x4):
    return mat4x4 is None or np.allclose(mat4x4, np.eye(4), atol=1e-6)


def _normalize_color(c):
    c = np.array(c, dtype=np.float32)
    if c.size >= 3 and c[:3].max() > 1.0:
        c[:3] /= 255.0
    return c


# ── Texture helpers ──────────────────────────────────────────────────

def _load_vtk_texture(scene, texture_path):
    data = scene.read_file(texture_path)
    if not data:
        data = scene.read_file(os.path.basename(texture_path))
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    arr = np.ascontiguousarray(np.flipud(np.array(img)))
    h, w, _ = arr.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(w, h, 1)
    vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    vtk_arr = numpy_support.numpy_to_vtk(arr.reshape(-1, 3), deep=True)
    vtk_img.GetPointData().SetScalars(vtk_arr)
    tex = vtk.vtkTexture()
    tex.SetInputData(vtk_img)
    tex.SetRepeat(True)
    tex.SetInterpolate(True)
    return tex


def _load_vtk_texture_grayscale(scene, texture_path):
    data = scene.read_file(texture_path)
    if not data:
        data = scene.read_file(os.path.basename(texture_path))
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        return None
    arr = np.flipud(np.array(img))
    arr_rgb = np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1))
    h, w = arr.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(w, h, 1)
    vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    vtk_arr = numpy_support.numpy_to_vtk(arr_rgb.reshape(-1, 3), deep=True)
    vtk_img.GetPointData().SetScalars(vtk_arr)
    tex = vtk.vtkTexture()
    tex.SetInputData(vtk_img)
    tex.SetRepeat(True)
    tex.SetInterpolate(True)
    return tex


def _compute_tcoords(pattern, material):
    uvs = np.array(pattern.uvs).reshape(-1, 2).copy()
    tw = material.tile_width if material.tile_width > 0 else 1.0
    th = material.tile_height if material.tile_height > 0 else 1.0
    uvs[:, 0] /= tw
    uvs[:, 1] /= th
    xf = material.diffuse_texture_transform
    angle = getattr(xf, "rotation", 0.0)
    if angle:
        r = math.radians(angle)
        c, s = math.cos(r), math.sin(r)
        u, v = uvs[:, 0].copy(), uvs[:, 1].copy()
        uvs[:, 0] = u * c - v * s
        uvs[:, 1] = u * s + v * c
    ou = getattr(xf, "offset_u", 0.0)
    ov = getattr(xf, "offset_v", 0.0)
    if ou or ov:
        uvs[:, 0] += ou
        uvs[:, 1] += ov
    return uvs


# ── Scene → actor data ──────────────────────────────────────────────

def _get_pattern_material(scene, pattern_index):
    materials = list(scene.fabric_materials)
    colorway_idx = scene.active_colorway_index if scene.colorways else None
    colorway = (scene.colorways[colorway_idx]
                if colorway_idx is not None and 0 <= colorway_idx < len(scene.colorways)
                else None)
    mi = scene.garment_patterns[pattern_index].material_index
    if colorway and pattern_index < len(colorway.pattern_fabric_indices):
        mi = colorway.pattern_fabric_indices[pattern_index]
    if 0 <= mi < len(materials):
        return materials[mi]
    return None


def _build_background_cylinder(actors_data, n_seg=64, fov_deg=15.0):
    """Add a closed cylinder (wall + floor + ceiling) enclosing all existing actors."""
    all_bounds = [ad["polydata"].GetBounds() for ad in actors_data]
    xmin = min(b[0] for b in all_bounds)
    xmax = max(b[1] for b in all_bounds)
    ymin = min(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)
    zmin = min(b[4] for b in all_bounds)
    zmax = max(b[5] for b in all_bounds)

    cx = (xmin + xmax) / 2
    cz = (zmin + zmax) / 2
    height = ymax - ymin
    span_x = xmax - xmin

    half_fov = math.radians(fov_deg) / 2.0
    cam_dist = max(height / 2.0 / math.tan(half_fov),
                   span_x / 2.0 / math.tan(half_fov)) * 1.05
    radius = cam_dist * 1.1
    y_floor = ymin
    y_top = ymax + height * 0.6

    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False, dtype=np.float32)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    bg_entry = {
        "material": None, "has_tcoords": False,
        "diffuse_color": (0.5, 0.5, 0.5), "roughness": 0.5, "metallic": 0.0,
        "type": "background",
    }

    # Wall
    wall_bot = np.stack([cx + radius * cos_a, np.full(n_seg, y_floor), cz + radius * sin_a], axis=1)
    wall_top = np.stack([cx + radius * cos_a, np.full(n_seg, y_top), cz + radius * sin_a], axis=1)
    wall_v = np.concatenate([wall_bot, wall_top]).astype(np.float32)
    wall_f = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        wall_f.append([i, j, j + n_seg])
        wall_f.append([i, j + n_seg, i + n_seg])
    wall_f = np.array(wall_f, np.int32)
    actors_data.append({**bg_entry, "polydata": _make_polydata(wall_v, wall_f)})

    # Floor disc
    floor_center = np.array([[cx, y_floor, cz]], np.float32)
    floor_ring = np.stack([cx + radius * cos_a, np.full(n_seg, y_floor), cz + radius * sin_a], axis=1)
    floor_v = np.concatenate([floor_center, floor_ring]).astype(np.float32)
    floor_f = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        floor_f.append([0, j + 1, i + 1])
    floor_f = np.array(floor_f, np.int32)
    actors_data.append({**bg_entry, "polydata": _make_polydata(floor_v, floor_f)})

    # Ceiling disc
    ceil_center = np.array([[cx, y_top, cz]], np.float32)
    ceil_ring = np.stack([cx + radius * cos_a, np.full(n_seg, y_top), cz + radius * sin_a], axis=1)
    ceil_v = np.concatenate([ceil_center, ceil_ring]).astype(np.float32)
    ceil_f = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        ceil_f.append([0, j + 1, i + 1])
    ceil_f = np.array(ceil_f, np.int32)
    actors_data.append({**bg_entry, "polydata": _make_polydata(ceil_v, ceil_f)})


def build_scene_actors(scene, background=True):
    """Build per-mesh actor data from a zprj scene.

    Args:
        scene: zprj_loader scene object.
        background: if True, add a dummy background cylinder enclosing the scene.

    Returns list of dicts with keys:
        polydata, material, has_tcoords, diffuse_color, roughness, metallic, type
    """
    actors_data = []

    # Garment patterns
    for i, pat in enumerate(scene.garment_patterns):
        nv, nf = pat.vertex_count, pat.triangle_count
        if nv == 0 or nf == 0:
            continue
        v = np.array(pat.positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(pat.indices, dtype=np.int32).reshape(nf, 3)
        n = None
        if len(pat.normals) == nv * 3:
            n = np.array(pat.normals, dtype=np.float32).reshape(nv, 3)
        pd = _make_polydata(v, f, n)

        mat = _get_pattern_material(scene, i)
        has_tcoords = False
        if mat and pat.uv_vertex_count == nv:
            tcoords = _compute_tcoords(pat, mat)
            tc_arr = numpy_support.numpy_to_vtk(tcoords.astype(np.float32), deep=True)
            tc_arr.SetNumberOfComponents(2)
            pd.GetPointData().SetTCoords(tc_arr)
            has_tcoords = True

        dc = (0.8, 0.8, 0.8)
        if mat:
            c = _normalize_color(np.array(mat.diffuse_color, dtype=np.float32))
            dc = (float(c[0]), float(c[1]), float(c[2]))

        ro_val, me_val = 0.5, 0.0
        if mat and getattr(mat, "use_metalness_roughness_pbr", False):
            ro_val = mat.roughness
            me_val = mat.metalness

        actors_data.append({
            "polydata": pd, "material": mat, "has_tcoords": has_tcoords,
            "diffuse_color": dc, "roughness": ro_val, "metallic": me_val,
            "type": "garment",
        })

    # Avatar meshes
    for mesh in scene.avatar_meshes:
        nv, nf = mesh.vertex_count, mesh.triangle_count
        if nv == 0 or nf == 0:
            continue
        v = np.array(mesh.positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(mesh.indices, dtype=np.int32).reshape(nf, 3)
        n = None
        if len(mesh.normals) == nv * 3:
            n = np.array(mesh.normals, dtype=np.float32).reshape(nv, 3)
        pd = _make_polydata(v, f, n)

        wm = mesh.world_matrix
        if wm is not None and not _is_identity(wm):
            pd = _apply_matrix(pd, wm)

        mat = mesh.material
        if mat and hasattr(mat, "diffuse_color"):
            alpha = float(mat.diffuse_color[3])
            if alpha < 1e-3:
                continue

        has_tcoords = False
        if mat and mat.diffuse_texture_path and mesh.uv_vertex_count == nv:
            uvs = np.array(mesh.uvs).reshape(-1, 2).astype(np.float32)
            tc_arr = numpy_support.numpy_to_vtk(uvs, deep=True)
            tc_arr.SetNumberOfComponents(2)
            pd.GetPointData().SetTCoords(tc_arr)
            has_tcoords = True

        dc = (0.85, 0.75, 0.65)
        if mat and hasattr(mat, "diffuse_color"):
            c = _normalize_color(np.array(mat.diffuse_color, dtype=np.float32))
            dc = (float(c[0]), float(c[1]), float(c[2]))

        if mesh.vertex_colors and len(mesh.vertex_colors) >= nv * 3:
            vc = np.array(mesh.vertex_colors, dtype=np.float32)
            n_comp = len(vc) // nv
            rgb = vc.reshape(nv, n_comp)[:, :3].copy()
            if rgb.max() > 1.0:
                rgb /= 255.0
            rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            vtk_colors = numpy_support.numpy_to_vtk(
                np.ascontiguousarray(rgb_u8), deep=True)
            pd.GetPointData().SetScalars(vtk_colors)

        actors_data.append({
            "polydata": pd, "material": mat, "has_tcoords": has_tcoords,
            "diffuse_color": dc, "roughness": 0.5, "metallic": 0.0,
            "type": "avatar",
        })

    # Trim objects
    for trim in scene.trim_objects:
        nv, nf = trim.mesh_vertex_count, trim.mesh_triangle_count
        if nv == 0 or nf == 0:
            continue
        v = np.array(trim.mesh_positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(trim.mesh_indices, dtype=np.int32).reshape(nf, 3)
        n = None
        if len(trim.mesh_normals) == nv * 3:
            n = np.array(trim.mesh_normals, dtype=np.float32).reshape(nv, 3)
        pd = _make_polydata(v, f, n)
        tm = trim.transform_matrix
        if tm is not None and not _is_identity(tm):
            pd = _apply_matrix(pd, tm)
        actors_data.append({
            "polydata": pd, "material": None, "has_tcoords": False,
            "diffuse_color": (0.6, 0.6, 0.7), "roughness": 0.5, "metallic": 0.0,
            "type": "trim",
        })

    # Zipper teeth
    for zi in scene.zipper_instances:
        nv, nf = zi.teeth_vertex_count, zi.teeth_triangle_count
        if nv == 0 or nf == 0:
            continue
        v = np.array(zi.teeth_positions, dtype=np.float32).reshape(nv, 3)
        f = np.array(zi.teeth_indices, dtype=np.int32).reshape(nf, 3)
        pd = _make_polydata(v, f)
        if zi.has_transform:
            tm = zi.transform
            if tm is not None and not _is_identity(tm):
                pd = _apply_matrix(pd, tm)
        actors_data.append({
            "polydata": pd, "material": None, "has_tcoords": False,
            "diffuse_color": (0.8, 0.8, 0.3), "roughness": 0.3, "metallic": 0.8,
            "type": "zipper",
        })

    # Button meshes
    for bs in scene.button_head_styles:
        if not bs.has_mesh_3d or bs.mesh_vertex_count == 0 or bs.mesh_triangle_count == 0:
            continue
        v = np.array(bs.mesh_positions, dtype=np.float32).reshape(bs.mesh_vertex_count, 3)
        f = np.array(bs.mesh_indices, dtype=np.int32).reshape(bs.mesh_triangle_count, 3)
        pd = _make_polydata(v, f)
        actors_data.append({
            "polydata": pd, "material": None, "has_tcoords": False,
            "diffuse_color": (0.9, 0.85, 0.7), "roughness": 0.3, "metallic": 0.5,
            "type": "button",
        })

    if background and actors_data:
        _build_background_cylinder(actors_data)

    return actors_data


# ── Camera ───────────────────────────────────────────────────────────

def _setup_camera(renderer, actors_data, fov_deg=15.0, azimuth_deg=0.0, aspect=1.0):
    fg = [ad for ad in actors_data if ad["type"] != "background"]
    all_bounds = [ad["polydata"].GetBounds() for ad in (fg or actors_data)]
    xmin = min(b[0] for b in all_bounds)
    xmax = max(b[1] for b in all_bounds)
    ymin = min(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)
    zmin = min(b[4] for b in all_bounds)
    zmax = max(b[5] for b in all_bounds)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    sx, sy = xmax - xmin, ymax - ymin

    half_fov = math.radians(fov_deg) / 2.0
    dist_y = (sy / 2.0) / math.tan(half_fov)
    dist_x = (sx / 2.0) / math.tan(half_fov * aspect)
    dist = max(dist_y, dist_x) * 1.05

    az = math.radians(azimuth_deg)
    cam = renderer.GetActiveCamera()
    cam.SetPosition(cx + math.sin(az) * dist, cy, cz + math.cos(az) * dist)
    cam.SetFocalPoint(cx, cy, cz)
    cam.SetViewUp(0, 1, 0)
    cam.SetViewAngle(fov_deg)
    span = max(sx, sy, zmax - zmin)
    cam.SetClippingRange(max(dist - span, 0.1), dist * 2.5)


# ── Capture ──────────────────────────────────────────────────────────

def _capture_rgb(win):
    win.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    return np.flipud(arr.reshape(h, w, 3)).copy()


def _capture_zbuffer(win):
    win.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToZBuffer()
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    return np.flipud(arr.reshape(h, w)).copy()


# ── Public API ───────────────────────────────────────────────────────

def render_gbuffers(scene, resolution=512, fov_deg=20.0, azimuth_deg=0.0,
                    device="cpu", _actors_data=None, background=True):
    """Render G-buffers for a single viewpoint using VTK.

    Args:
        scene: zprj_loader scene object.
        resolution: int or (H, W) tuple.
        fov_deg: camera field of view.
        azimuth_deg: camera azimuth angle.
        device: torch device for output tensors.
        _actors_data: pre-built actor data (for multi-frame reuse).
        background: if True, include background cylinder (ignored if _actors_data given).

    Returns:
        dict of {name: (H, W, 3) float32 torch.Tensor in [0, 1]}.
    """
    if isinstance(resolution, (list, tuple)):
        res_h, res_w = resolution
    else:
        res_h = res_w = resolution
    aspect = res_w / res_h

    actors_data = _actors_data or build_scene_actors(scene, background=background)
    if not actors_data:
        raise ValueError("No geometry found in scene")

    win = vtk.vtkRenderWindow()
    win.SetSize(res_w, res_h)
    win.SetOffScreenRendering(True)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)
    win.AddRenderer(ren)

    _setup_camera(ren, actors_data, fov_deg, azimuth_deg, aspect)

    tex_cache = {}
    gb = {}

    def _to_tensor(arr_uint8):
        return torch.from_numpy(arr_uint8.astype(np.float32) / 255.0).to(device)

    # --- Basecolor (texture-mapped, unlit) ---
    ren.RemoveAllViewProps()
    for ad in actors_data:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        prop.SetColor(*ad["diffuse_color"])
        mat = ad["material"]
        if ad["has_tcoords"] and mat and mat.diffuse_texture_path:
            key = ("diffuse", mat.diffuse_texture_path)
            if key not in tex_cache:
                tex_cache[key] = _load_vtk_texture(scene, mat.diffuse_texture_path)
            vtk_tex = tex_cache[key]
            if vtk_tex:
                actor.SetTexture(vtk_tex)
        ren.AddActor(actor)
    gb["basecolor"] = _to_tensor(_capture_rgb(win))

    # --- Normal (custom shader) ---
    normal_sp = vtk.vtkShaderProperty()
    normal_sp.SetVertexShaderCode(NORMAL_VERT)
    normal_sp.SetFragmentShaderCode(NORMAL_FRAG)
    ren.RemoveAllViewProps()
    for ad in actors_data:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetShaderProperty(normal_sp)
        ren.AddActor(actor)
    gb["normal"] = _to_tensor(_capture_rgb(win))

    # --- Depth (from Z-buffer, foreground-normalized) ---
    # Render foreground-only to get depth range excluding background cylinder
    fg_actors = [ad for ad in actors_data if ad["type"] != "background"]
    ren.RemoveAllViewProps()
    for ad in fg_actors:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        ren.AddActor(actor)
    fg_zbuf = _capture_zbuffer(win)
    fg_valid = fg_zbuf[fg_zbuf < 1.0 - 1e-6]

    # Render full scene (with background) for final depth
    ren.RemoveAllViewProps()
    for ad in actors_data:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        ren.AddActor(actor)
    zbuf = _capture_zbuffer(win)
    bg_mask = zbuf >= 1.0 - 1e-6

    if fg_valid.size > 0:
        depth_norm = (zbuf - fg_valid.min()) / (fg_valid.max() - fg_valid.min() + 1e-8)
    else:
        depth_norm = zbuf.copy()
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_norm[bg_mask] = 1.0
    depth_rgb = np.stack([depth_norm] * 3, axis=-1).astype(np.float32)
    gb["depth"] = torch.from_numpy(depth_rgb).to(device)

    # --- Roughness (texture or uniform, unlit) ---
    ren.RemoveAllViewProps()
    for ad in actors_data:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        rv = ad["roughness"]
        prop.SetColor(rv, rv, rv)
        mat = ad["material"]
        if ad["has_tcoords"] and mat and mat.roughness_texture_path:
            key = ("roughness", mat.roughness_texture_path)
            if key not in tex_cache:
                tex_cache[key] = _load_vtk_texture_grayscale(
                    scene, mat.roughness_texture_path)
            vtk_tex = tex_cache[key]
            if vtk_tex:
                actor.SetTexture(vtk_tex)
        ren.AddActor(actor)
    gb["roughness"] = _to_tensor(_capture_rgb(win))

    # --- Metallic (texture or uniform, unlit) ---
    ren.RemoveAllViewProps()
    for ad in actors_data:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        mv = ad["metallic"]
        prop.SetColor(mv, mv, mv)
        mat = ad["material"]
        if ad["has_tcoords"] and mat and mat.metalness_texture_path:
            key = ("metallic", mat.metalness_texture_path)
            if key not in tex_cache:
                tex_cache[key] = _load_vtk_texture_grayscale(
                    scene, mat.metalness_texture_path)
            vtk_tex = tex_cache[key]
            if vtk_tex:
                actor.SetTexture(vtk_tex)
        ren.AddActor(actor)
    gb["metallic"] = _to_tensor(_capture_rgb(win))

    win.Finalize()
    return gb
