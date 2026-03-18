"""VTK rendering of all visual data from a .zprj file.

Renders:
  - Garment patterns (colored by active colorway, with diffuse textures)
  - Avatar meshes (with vertex colors, semi-transparent)
  - Trim object meshes
  - Zipper teeth meshes
  - Button 3D meshes
  - Sewing lines (colored by seam group)

Usage: uv run examples/python/vtk_rendering.py <file.zprj>
"""

import io
import math
import sys

import numpy as np
import vtk
from vtk.util import numpy_support

import zprj_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_polydata(vertices, faces, normals=None):
    """Create vtkPolyData from Nx3 vertex and Mx3 face arrays."""
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))

    cells = vtk.vtkCellArray()
    cells.SetData(3, numpy_support.numpy_to_vtk(faces.ravel(), deep=True))

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(cells)

    if normals is not None and len(normals) == len(vertices):
        pd.GetPointData().SetNormals(
            numpy_support.numpy_to_vtk(normals, deep=True)
        )
    else:
        gen = vtk.vtkPolyDataNormals()
        gen.SetInputData(pd)
        gen.Update()
        pd = gen.GetOutput()

    return pd


def apply_matrix(polydata, mat4x4):
    """Apply a 4x4 numpy matrix transform to polydata.

    Matrices from the C++ parser are stored column-major (flat float[16]),
    but numpy reshapes them row-major, so they arrive transposed.
    """
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
    return np.allclose(mat4x4, np.eye(4), atol=1e-6)


def set_vertex_colors(polydata, color_floats, vertex_count):
    """Assign per-vertex RGB colors from a flat float list (values 0-1)."""
    vc = np.array(color_floats)
    n_comp = len(vc) // vertex_count
    if n_comp < 3:
        return
    vc = vc.reshape(vertex_count, n_comp)
    rgb = (np.clip(vc[:, :3], 0, 1) * 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    vtk_colors = numpy_support.numpy_to_vtk(rgb, deep=True)
    polydata.GetPointData().SetScalars(vtk_colors)


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------

def load_vtk_texture(scene, texture_path):
    """Load an embedded image as a vtkTexture. Returns None on failure."""
    try:
        from PIL import Image
    except ImportError:
        return None

    data = scene.read_file(texture_path)
    if not data:
        return None

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

    arr = np.array(img)
    arr = np.flipud(arr)  # VTK expects bottom-to-top
    arr = np.ascontiguousarray(arr)

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


def compute_tcoords(pattern, material):
    """Transform pattern UVs (mm) to tiled texture coordinates."""
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


# ---------------------------------------------------------------------------
# Garment patterns
# ---------------------------------------------------------------------------

def get_pattern_color(scene, pattern_index):
    """Get diffuse RGB from the active colorway for a pattern."""

    try:
        cw_idx = scene.active_colorway_index
        if cw_idx < 0 or cw_idx >= len(scene.colorways):
            return None
        cw = scene.colorways[cw_idx]

        # Determine fabric material index for this pattern
        if pattern_index < len(cw.pattern_fabric_indices):
            fabric_idx = cw.pattern_fabric_indices[pattern_index]
        else:
            fabric_idx = scene.garment_patterns[pattern_index].material_index

        # Try per-colorway chips first, then fall back to scene-level
        # (new format stores chips in scene.style_assignments)
        for sa in (cw.style_assignments, scene.style_assignments):
            if fabric_idx < len(sa.fabrics):
                chips = sa.fabrics[fabric_idx].chips
                if chips:
                    c = chips[0].color  # numpy array [R, G, B, A]
                    return (float(c[0]), float(c[1]), float(c[2]))
    except Exception:
        pass
    return None


def get_pattern_material(scene, pattern_index):
    """Get the fabric material for a pattern (via active colorway mapping)."""
    try:
        cw_idx = scene.active_colorway_index
        cw = scene.colorways[cw_idx]
        if pattern_index < len(cw.pattern_fabric_indices):
            mat_idx = cw.pattern_fabric_indices[pattern_index]
        else:
            mat_idx = scene.garment_patterns[pattern_index].material_index
        if 0 <= mat_idx < len(scene.fabric_materials):
            return scene.fabric_materials[mat_idx]
    except Exception:
        pass
    return None


def add_garment_patterns(renderer, scene):
    # Pre-load textures per material (cache to avoid duplicates)
    tex_cache = {}  # material index -> vtkTexture or None

    count = 0
    textured = 0
    for i, pattern in enumerate(scene.garment_patterns):
        if pattern.vertex_count == 0 or pattern.triangle_count == 0:
            continue

        v = np.array(pattern.positions).reshape(-1, 3)
        f = np.array(pattern.indices).reshape(-1, 3)

        n = None
        if len(pattern.normals) == pattern.vertex_count * 3:
            n = np.array(pattern.normals).reshape(-1, 3)

        pd = make_polydata(v, f, n)

        # Try to apply texture
        mat = get_pattern_material(scene, i)
        vtk_tex = None
        if mat and mat.diffuse_texture_path:
            mat_idx = pattern.material_index
            if mat_idx not in tex_cache:
                tex_cache[mat_idx] = load_vtk_texture(scene, mat.diffuse_texture_path)
            vtk_tex = tex_cache[mat_idx]

        if vtk_tex and pattern.uv_vertex_count == pattern.vertex_count:
            tcoords = compute_tcoords(pattern, mat)
            tc_arr = numpy_support.numpy_to_vtk(
                tcoords.astype(np.float32), deep=True
            )
            tc_arr.SetNumberOfComponents(2)
            pd.GetPointData().SetTCoords(tc_arr)

        color = get_pattern_color(scene, i) or (0.8, 0.8, 0.8)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)

        if vtk_tex and pattern.uv_vertex_count == pattern.vertex_count:
            actor.SetTexture(vtk_tex)
            textured += 1

        renderer.AddActor(actor)
        count += 1

    if count:
        msg = f"  Garment patterns: {count}"
        if textured:
            msg += f" ({textured} textured)"
        print(msg)


# ---------------------------------------------------------------------------
# Avatar meshes
# ---------------------------------------------------------------------------

def add_avatar_meshes(renderer, scene):
    tex_cache = {}  # texture_path -> vtkTexture or None
    count = 0
    textured = 0
    for mesh in scene.avatar_meshes:
        if mesh.vertex_count == 0 or mesh.triangle_count == 0:
            continue

        v = np.array(mesh.positions).reshape(-1, 3)
        f = np.array(mesh.indices).reshape(-1, 3)

        n = None
        if len(mesh.normals) == mesh.vertex_count * 3:
            n = np.array(mesh.normals).reshape(-1, 3)

        pd = make_polydata(v, f, n)

        if mesh.vertex_colors:
            set_vertex_colors(pd, mesh.vertex_colors, mesh.vertex_count)

        # Texture mapping
        mat = mesh.material
        vtk_tex = None
        if mat and mat.diffuse_texture_path and mesh.uv_vertex_count == mesh.vertex_count:
            tex_path = mat.diffuse_texture_path
            if tex_path not in tex_cache:
                tex_cache[tex_path] = load_vtk_texture(scene, tex_path)
            vtk_tex = tex_cache[tex_path]
            if vtk_tex:
                uvs = np.array(mesh.uvs).reshape(-1, 2).astype(np.float32)
                tc_arr = numpy_support.numpy_to_vtk(uvs, deep=True)
                tc_arr.SetNumberOfComponents(2)
                pd.GetPointData().SetTCoords(tc_arr)

        wm = mesh.world_matrix
        if wm is not None and not is_identity(wm):
            pd = apply_matrix(pd, wm)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if vtk_tex:
            actor.SetTexture(vtk_tex)
            textured += 1

        # Skip fully transparent meshes
        if mat and hasattr(mat, "diffuse_color"):
            alpha = float(mat.diffuse_color[3])
            if alpha < 1e-3:
                continue

        # Color & opacity from material
        if mat and hasattr(mat, "diffuse_color"):
            dc = mat.diffuse_color
            if not vtk_tex and not mesh.vertex_colors:
                actor.GetProperty().SetColor(float(dc[0]), float(dc[1]), float(dc[2]))
            actor.GetProperty().SetOpacity(float(dc[3]))
        elif not mesh.vertex_colors and not vtk_tex:
            actor.GetProperty().SetColor(0.85, 0.75, 0.65)

        renderer.AddActor(actor)
        count += 1

    if count:
        msg = f"  Avatar meshes: {count}"
        if textured:
            msg += f" ({textured} textured)"
        print(msg)


# ---------------------------------------------------------------------------
# Trim objects
# ---------------------------------------------------------------------------

def add_trim_objects(renderer, scene):
    count = 0
    for trim in scene.trim_objects:
        if trim.mesh_vertex_count == 0 or trim.mesh_triangle_count == 0:
            continue

        v = np.array(trim.mesh_positions).reshape(-1, 3)
        f = np.array(trim.mesh_indices).reshape(-1, 3)

        n = None
        if len(trim.mesh_normals) == trim.mesh_vertex_count * 3:
            n = np.array(trim.mesh_normals).reshape(-1, 3)

        pd = make_polydata(v, f, n)

        tm = trim.transform_matrix
        if tm is not None and not is_identity(tm):
            pd = apply_matrix(pd, tm)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.6, 0.6, 0.7)
        renderer.AddActor(actor)
        count += 1

    if count:
        print(f"  Trim objects: {count}")


# ---------------------------------------------------------------------------
# Zipper teeth
# ---------------------------------------------------------------------------

def add_zipper_teeth(renderer, scene):
    count = 0
    for zi in scene.zipper_instances:
        if zi.teeth_vertex_count == 0 or zi.teeth_triangle_count == 0:
            continue

        v = np.array(zi.teeth_positions).reshape(-1, 3)
        f = np.array(zi.teeth_indices).reshape(-1, 3)
        pd = make_polydata(v, f)

        if zi.has_transform:
            tm = zi.transform
            if tm is not None and not is_identity(tm):
                pd = apply_matrix(pd, tm)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.3)
        renderer.AddActor(actor)
        count += 1

    if count:
        print(f"  Zipper teeth: {count}")


# ---------------------------------------------------------------------------
# Button 3D meshes
# ---------------------------------------------------------------------------

def add_button_meshes(renderer, scene):
    count = 0
    for bs in scene.button_head_styles:
        if not bs.has_mesh_3d:
            continue
        if bs.mesh_vertex_count == 0 or bs.mesh_triangle_count == 0:
            continue

        v = np.array(bs.mesh_positions).reshape(-1, 3)
        f = np.array(bs.mesh_indices).reshape(-1, 3)
        pd = make_polydata(v, f)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.85, 0.7)
        renderer.AddActor(actor)
        count += 1

    if count:
        print(f"  Button meshes: {count}")


# ---------------------------------------------------------------------------
# Sewing lines
# ---------------------------------------------------------------------------

def add_sewing_lines(renderer, scene):
    if not scene.has_seam_pair_group_data:
        return

    pattern_verts = []
    for pattern in scene.garment_patterns:
        if pattern.vertex_count > 0:
            pattern_verts.append(
                np.array(pattern.positions).reshape(-1, 3)
            )
        else:
            pattern_verts.append(np.empty((0, 3)))

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    pt_idx = 0

    for group in scene.seam_line_pair_groups:
        if not group.active:
            continue

        gc = group.color
        r, g, b = int(gc[0] * 255), int(gc[1] * 255), int(gc[2] * 255)

        for pair in group.pairs:
            pi0 = pair.param_line_0.pattern_self_index
            pi1 = pair.param_line_1.pattern_self_index
            idx0 = list(pair.mesh_point_indices_0)
            idx1 = list(pair.mesh_point_indices_1)
            if not idx0 or not idx1:
                continue
            if pi0 >= len(pattern_verts) or pi1 >= len(pattern_verts):
                continue

            v0 = pattern_verts[pi0]
            v1 = pattern_verts[pi1]
            n = min(len(idx0), len(idx1))

            for i in range(n):
                if idx0[i] < len(v0) and idx1[i] < len(v1):
                    pts.InsertNextPoint(v0[idx0[i]])
                    pts.InsertNextPoint(v1[idx1[i]])
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, pt_idx)
                    line.GetPointIds().SetId(1, pt_idx + 1)
                    lines.InsertNextCell(line)
                    colors.InsertNextTuple3(r, g, b)
                    pt_idx += 2

    if lines.GetNumberOfCells() > 0:
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        pd.GetCellData().SetScalars(colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2)
        renderer.AddActor(actor)
        print(f"  Sewing lines: {lines.GetNumberOfCells()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="VTK viewer for .zprj files")
    parser.add_argument("zprj_file", help="Path to .zprj file")
    parser.add_argument("--debug", action="store_true",
                        help="Save screenshot to tmp/ and exit without interactive window")
    args = parser.parse_args()

    scene = zprj_loader.parse(args.zprj_file)

    if not scene.valid:
        print(f"Error: {scene.error}")
        sys.exit(1)

    print("Rendering scene...")

    ren = vtk.vtkRenderer()
    ren.SetBackground(0.15, 0.15, 0.2)

    add_garment_patterns(ren, scene)
    add_avatar_meshes(ren, scene)
    add_trim_objects(ren, scene)
    add_zipper_teeth(ren, scene)
    add_button_meshes(ren, scene)
    add_sewing_lines(ren, scene)

    ren.ResetCamera()

    win = vtk.vtkRenderWindow()
    win.SetSize(1280, 960)
    win.SetWindowName("zprj_loader VTK Viewer")
    win.AddRenderer(ren)

    if args.debug:
        win.SetOffScreenRendering(True)
        win.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(win)
        w2i.Update()

        out_dir = Path("tmp")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "vtk_rendering.png"

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(out_path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"Saved: {out_path}")
    else:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.SetRenderWindow(win)
        win.Render()
        iren.Start()


if __name__ == "__main__":
    main()
