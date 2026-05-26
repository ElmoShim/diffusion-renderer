"""VTK-only G-buffer renderer for .zprj files.

Renders basecolor, normal (view-space), depth, roughness, metallic
using VTK offscreen rendering with per-pattern texture mapping
and custom GLSL shaders (no default VTK lighting).

Usage:
    uv run scratch/vtk_gbuffer.py samples/garment.zprj
    uv run scratch/vtk_gbuffer.py samples/garment.zprj --output output/my_gbuffers/
    uv run scratch/vtk_gbuffer.py samples/garment.zprj --resolution 1024
"""

import argparse
import io
import math
import os
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support
from PIL import Image

import zprj_loader


# ---------------------------------------------------------------------------
# Shaders — borrowed from GarmentMapDiffusion, adapted per G-buffer channel
# ---------------------------------------------------------------------------

# Normal: view-space normal encoded as (n+1)/2
# Uses normalMatrix * normalMC to get view-space normals, bypassing VTK lighting.
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


def _make_normal_shader_property():
    sp = vtk.vtkShaderProperty()
    sp.SetVertexShaderCode(NORMAL_VERT)
    sp.SetFragmentShaderCode(NORMAL_FRAG)
    return sp


# ---------------------------------------------------------------------------
# Geometry / texture helpers (from vtk_render.py)
# ---------------------------------------------------------------------------

def make_polydata(vertices, faces, normals=None):
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


def apply_matrix(polydata, mat4x4):
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


def is_identity(mat4x4):
    return mat4x4 is None or np.allclose(mat4x4, np.eye(4), atol=1e-6)


def load_vtk_texture(scene, texture_path):
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


def load_vtk_texture_grayscale(scene, texture_path):
    """Load a texture and convert to grayscale RGB for roughness/metallic."""
    data = scene.read_file(texture_path)
    if not data:
        data = scene.read_file(os.path.basename(texture_path))
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        return None
    arr = np.array(img)
    arr = np.flipud(arr)
    arr_rgb = np.stack([arr, arr, arr], axis=-1)
    arr_rgb = np.ascontiguousarray(arr_rgb)
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


def compute_tcoords(pattern, material):
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


def normalize_color(c):
    c = np.array(c, dtype=np.float32)
    if c.size >= 3 and c[:3].max() > 1.0:
        c[:3] /= 255.0
    return c


# ---------------------------------------------------------------------------
# Scene actor builder — per-pattern actors with texture mapping
# ---------------------------------------------------------------------------

def get_pattern_material(scene, pattern_index):
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


def build_scene_actors(scene):
    """Build per-mesh actor data: list of (polydata, material, has_tcoords).

    Each entry has tcoords set on the polydata if UV mapping is available.
    Returns list of dicts with keys: polydata, material, has_tcoords, diffuse_color.
    """
    actors_data = []
    tex_cache = {}

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
        pd = make_polydata(v, f, n)

        mat = get_pattern_material(scene, i)
        has_tcoords = False
        if mat and pat.uv_vertex_count == nv:
            tcoords = compute_tcoords(pat, mat)
            tc_arr = numpy_support.numpy_to_vtk(tcoords.astype(np.float32), deep=True)
            tc_arr.SetNumberOfComponents(2)
            pd.GetPointData().SetTCoords(tc_arr)
            has_tcoords = True

        dc = (0.8, 0.8, 0.8)
        if mat:
            c = normalize_color(np.array(mat.diffuse_color, dtype=np.float32))
            dc = (float(c[0]), float(c[1]), float(c[2]))

        ro_val = 0.5
        me_val = 0.0
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
        pd = make_polydata(v, f, n)

        wm = mesh.world_matrix
        if wm is not None and not is_identity(wm):
            pd = apply_matrix(pd, wm)

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
            c = normalize_color(np.array(mat.diffuse_color, dtype=np.float32))
            dc = (float(c[0]), float(c[1]), float(c[2]))

        # Avatar vertex colors
        if mesh.vertex_colors and len(mesh.vertex_colors) >= nv * 3:
            vc = np.array(mesh.vertex_colors, dtype=np.float32)
            n_comp = len(vc) // nv
            rgb = vc.reshape(nv, n_comp)[:, :3].copy()
            if rgb.max() > 1.0:
                rgb /= 255.0
            rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            vtk_colors = numpy_support.numpy_to_vtk(np.ascontiguousarray(rgb_u8), deep=True)
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
        pd = make_polydata(v, f, n)
        tm = trim.transform_matrix
        if tm is not None and not is_identity(tm):
            pd = apply_matrix(pd, tm)
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
        pd = make_polydata(v, f)
        if zi.has_transform:
            tm = zi.transform
            if tm is not None and not is_identity(tm):
                pd = apply_matrix(pd, tm)
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
        pd = make_polydata(v, f)
        actors_data.append({
            "polydata": pd, "material": None, "has_tcoords": False,
            "diffuse_color": (0.9, 0.85, 0.7), "roughness": 0.3, "metallic": 0.5,
            "type": "button",
        })

    return actors_data


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def setup_camera(renderer, actors_data, fov_deg=15.0, azimuth_deg=0.0):
    all_bounds = []
    for ad in actors_data:
        b = ad["polydata"].GetBounds()
        all_bounds.append(b)

    xmin = min(b[0] for b in all_bounds)
    xmax = max(b[1] for b in all_bounds)
    ymin = min(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)
    zmin = min(b[4] for b in all_bounds)
    zmax = max(b[5] for b in all_bounds)

    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin

    half_fov = math.radians(fov_deg) / 2.0
    dist_y = (sy / 2.0) / math.tan(half_fov)
    dist_x = (sx / 2.0) / math.tan(half_fov)
    dist = max(dist_y, dist_x) * 1.05

    az = math.radians(azimuth_deg)
    eye_x = cx + math.sin(az) * dist
    eye_y = cy
    eye_z = cz + math.cos(az) * dist

    cam = renderer.GetActiveCamera()
    cam.SetPosition(eye_x, eye_y, eye_z)
    cam.SetFocalPoint(cx, cy, cz)
    cam.SetViewUp(0, 1, 0)
    cam.SetViewAngle(fov_deg)
    near = max(dist - max(sx, sy, sz), 0.1)
    far = dist * 2.5
    cam.SetClippingRange(near, far)


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def capture_framebuffer(win):
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


def capture_zbuffer(win):
    win.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToZBuffer()
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    return np.flipud(arr.reshape(h, w)).copy()


# ---------------------------------------------------------------------------
# Multi-pass G-buffer renderer
# ---------------------------------------------------------------------------

def render_gbuffers_vtk(scene, actors_data, resolution=512, fov_deg=15.0, azimuth_deg=0.0):
    if isinstance(resolution, int):
        res_w = res_h = resolution
    else:
        res_w, res_h = resolution

    win = vtk.vtkRenderWindow()
    win.SetSize(res_w, res_h)
    win.SetOffScreenRendering(True)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)
    win.AddRenderer(ren)

    setup_camera(ren, actors_data, fov_deg, azimuth_deg)

    gbuffers = {}
    tex_cache = {}

    # --- Pass 1: Basecolor (texture-mapped, unlit) ---
    ren.RemoveAllViewProps()
    for ad in actors_data:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Unlit: full ambient, no diffuse/specular
        prop = actor.GetProperty()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        prop.SetColor(*ad["diffuse_color"])

        # Apply texture if available
        mat = ad["material"]
        if ad["has_tcoords"] and mat and mat.diffuse_texture_path:
            tex_key = ("diffuse", mat.diffuse_texture_path)
            if tex_key not in tex_cache:
                tex_cache[tex_key] = load_vtk_texture(scene, mat.diffuse_texture_path)
            vtk_tex = tex_cache[tex_key]
            if vtk_tex:
                actor.SetTexture(vtk_tex)

        ren.AddActor(actor)
    gbuffers["basecolor"] = capture_framebuffer(win)
    print("  Rendered: basecolor")

    # --- Pass 2: Normal (custom shader, unlit) ---
    normal_sp = _make_normal_shader_property()
    ren.RemoveAllViewProps()
    for ad in actors_data:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ad["polydata"])
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetShaderProperty(normal_sp)
        ren.AddActor(actor)
    gbuffers["normal"] = capture_framebuffer(win)
    print("  Rendered: normal")

    # --- Pass 3: Depth (from Z-buffer) ---
    # Render any pass to populate Z-buffer
    zbuf = capture_zbuffer(win)
    bg_mask = zbuf >= 1.0 - 1e-6
    fg = zbuf[~bg_mask]
    if fg.size > 0:
        z_min, z_max = fg.min(), fg.max()
        depth_norm = (zbuf - z_min) / (z_max - z_min + 1e-8)
    else:
        depth_norm = zbuf.copy()
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_norm[bg_mask] = 1.0
    depth_rgb = np.stack([depth_norm] * 3, axis=-1)
    gbuffers["depth"] = (depth_rgb * 255).astype(np.uint8)
    print("  Rendered: depth")

    # --- Pass 4: Roughness (texture or uniform, unlit) ---
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
            tex_key = ("roughness", mat.roughness_texture_path)
            if tex_key not in tex_cache:
                tex_cache[tex_key] = load_vtk_texture_grayscale(scene, mat.roughness_texture_path)
            vtk_tex = tex_cache[tex_key]
            if vtk_tex:
                actor.SetTexture(vtk_tex)

        ren.AddActor(actor)
    gbuffers["roughness"] = capture_framebuffer(win)
    print("  Rendered: roughness")

    # --- Pass 5: Metallic (texture or uniform, unlit) ---
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
            tex_key = ("metallic", mat.metalness_texture_path)
            if tex_key not in tex_cache:
                tex_cache[tex_key] = load_vtk_texture_grayscale(scene, mat.metalness_texture_path)
            vtk_tex = tex_cache[tex_key]
            if vtk_tex:
                actor.SetTexture(vtk_tex)

        ren.AddActor(actor)
    gbuffers["metallic"] = capture_framebuffer(win)
    print("  Rendered: metallic")

    win.Finalize()
    return gbuffers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VTK G-buffer renderer for .zprj files")
    parser.add_argument("zprj_file", help="Path to .zprj file")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fov", type=float, default=15.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.zprj_file))[0]
    out_dir = args.output or f"output/vtk_gbuffers/{stem}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.zprj_file} ...")
    scene = zprj_loader.parse(args.zprj_file)
    if not scene.valid:
        print(f"Error: {scene.error}")
        sys.exit(1)

    print("Building scene actors ...")
    actors_data = build_scene_actors(scene)
    if not actors_data:
        print("No geometry found.")
        sys.exit(1)
    print(f"  {len(actors_data)} mesh parts")

    print(f"Rendering G-buffers at {args.resolution}x{args.resolution} ...")
    gb = render_gbuffers_vtk(scene, actors_data,
                             resolution=args.resolution,
                             fov_deg=args.fov,
                             azimuth_deg=args.azimuth)

    for name, arr in gb.items():
        path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(arr).save(path)

    print(f"\nG-buffers saved to {out_dir}/")


if __name__ == "__main__":
    main()
