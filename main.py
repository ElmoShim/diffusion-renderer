import sys
import os
import io
import math
import zipfile
import zprj_loader
import vtk

import numpy as np
from PIL import Image
from vtk.util import numpy_support


# ── in-memory texture loading from zprj archive ────────────────────────


def load_images_from_zprj(zprj_path: str) -> dict[str, bytes]:
    """Read all image files from zprj (and nested zpac) into memory.
    Returns {basename: raw_bytes}."""
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


def texture_from_bytes(data: bytes, srgb: bool = False) -> vtk.vtkTexture | None:
    """Decode image bytes (JPEG/PNG/etc.) via PIL and create a vtkTexture in-memory."""
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

    arr = np.asarray(pil_img)  # (H, W, 3) uint8
    # VTK images are bottom-up, PIL is top-down
    arr = np.flip(arr, axis=0).copy()

    h, w, c = arr.shape
    vtk_arr = numpy_support.numpy_to_vtk(arr.reshape(-1, c), deep=True)

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(w, h, 1)
    image_data.GetPointData().SetScalars(vtk_arr)

    texture = vtk.vtkTexture()
    texture.SetInputData(image_data)
    texture.InterpolateOn()
    texture.SetRepeat(True)
    if srgb:
        texture.SetUseSRGBColorSpace(True)
    return texture


# ── UV transform ────────────────────────────────────────────────────────


def apply_texture_transform(uvs: np.ndarray, mat) -> np.ndarray:
    """Convert mm-space UVs to tiled texture coordinates using tile size + transform."""
    result = uvs.copy()

    tw = mat.tile_width if mat.tile_width > 0 else 1.0
    th = mat.tile_height if mat.tile_height > 0 else 1.0
    result[:, 0] /= tw
    result[:, 1] /= th

    transform = mat.diffuse_texture_transform

    angle = getattr(transform, "rotation", 0.0)
    if angle:
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        u = result[:, 0].copy()
        v = result[:, 1].copy()
        result[:, 0] = u * c - v * s
        result[:, 1] = u * s + v * c

    ou = getattr(transform, "offset_u", 0.0)
    ov = getattr(transform, "offset_v", 0.0)
    if ou or ov:
        result[:, 0] += ou
        result[:, 1] += ov

    return result


# ── polydata construction ───────────────────────────────────────────────


def make_polydata(
    v: np.ndarray,
    f: np.ndarray,
    uvs: np.ndarray | None = None,
) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(v, deep=True))

    faces = vtk.vtkCellArray()
    faces.SetData(3, numpy_support.numpy_to_vtk(f.ravel(), deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(faces)

    # Normals
    normals_gen = vtk.vtkPolyDataNormals()
    normals_gen.SetInputData(polydata)
    normals_gen.SetSplitting(False)
    normals_gen.Update()
    polydata.GetPointData().SetNormals(
        normals_gen.GetOutput().GetPointData().GetNormals()
    )

    # Texture coordinates
    if uvs is not None:
        tc = numpy_support.numpy_to_vtk(uvs.astype(np.float32), deep=True)
        tc.SetName("TCoords")
        polydata.GetPointData().SetTCoords(tc)

    return polydata


# ── material application ────────────────────────────────────────────────


def apply_material_colors(prop: vtk.vtkProperty, mat, has_base_texture: bool = False):
    if has_base_texture and mat.use_metalness_roughness_pbr:
        # PBR multiplies DiffuseColor with texture; use white to show texture as-is
        prop.SetDiffuseColor(1.0, 1.0, 1.0)
    else:
        dc = np.array(mat.diffuse_color, dtype=float)
        if dc.size >= 3:
            prop.SetDiffuseColor(dc[0], dc[1], dc[2])
    prop.SetDiffuse(1.0)

    ac = np.array(mat.ambient_color, dtype=float)
    if ac.size >= 3:
        prop.SetAmbientColor(ac[0], ac[1], ac[2])
        prop.SetAmbient(0.1)

    sc = np.array(mat.specular_color, dtype=float)
    if sc.size >= 3:
        prop.SetSpecularColor(sc[0], sc[1], sc[2])
        prop.SetSpecular(0.3)

    ec = np.array(mat.emission_color, dtype=float)
    if ec.size >= 3:
        prop.SetEmissiveFactor(ec[0], ec[1], ec[2])

    if mat.opacity < 1.0:
        prop.SetOpacity(mat.opacity)


def apply_pbr(prop: vtk.vtkProperty, mat):
    if mat.use_metalness_roughness_pbr:
        prop.SetInterpolationToPBR()
        prop.SetMetallic(mat.metalness)
        prop.SetRoughness(mat.roughness)


# ── main ────────────────────────────────────────────────────────────────


def main():
    debug = "--debug" in sys.argv
    colorway_idx = None
    remaining = []
    argv_iter = iter(sys.argv[1:])
    for arg in argv_iter:
        if arg == "--debug":
            continue
        if arg == "--colorway":
            colorway_idx = int(next(argv_iter))
        else:
            remaining.append(arg)
    path = remaining[0] if remaining else "samples/garment.zprj"

    scene = zprj_loader.parse(path)

    if not scene.valid:
        print(f"Error: {scene.error}")
        sys.exit(1)

    zprj_path = os.path.abspath(path)

    # Load all images from the archive into memory
    image_cache = load_images_from_zprj(zprj_path)
    print(f"Loaded {len(image_cache)} images in-memory: {list(image_cache.keys())}")

    materials = list(scene.fabric_materials)

    # ── colorway info ──
    if scene.colorways:
        if colorway_idx is None:
            colorway_idx = scene.active_colorway_index
        print(f"Colorways ({len(scene.colorways)}), using index {colorway_idx}:")
        for i, cw in enumerate(scene.colorways):
            marker = " *" if i == colorway_idx else ""
            print(f"  [{i}] {cw.name}{marker}")

    # ── dump material info ──
    for i, mat in enumerate(materials):
        print(f"Material[{i}]: {mat.fabric_name}")
        print(f"  diffuse_color  = {list(mat.diffuse_color)}")
        if mat.diffuse_texture_path:
            print(f"  diffuse_tex    = {mat.diffuse_texture_path}")
        if mat.normal_texture_path:
            print(f"  normal_tex     = {mat.normal_texture_path}")

    def resolve_texture(tex_name: str) -> bytes | None:
        """Look up texture bytes by name (or basename) from the image cache."""
        if not tex_name:
            return None
        if tex_name in image_cache:
            return image_cache[tex_name]
        bn = os.path.basename(tex_name)
        return image_cache.get(bn)

    def find_substance_texture(mat, kind: str = "basecolor", idx: int | None = None) -> bytes | None:
        """Find substance-generated DDS texture matching the material name.
        Looks for files like output_<NAME>_basecolor[...].dds.
        When idx is given, picks the idx-th match (for colorway selection)."""
        keywords = [kw.lower() for kw in mat.fabric_name.split()]
        # Collect all matching DDS files, sorted by name for stable ordering
        candidates: list[tuple[str, bytes]] = []
        for cached_name, data in sorted(image_cache.items()):
            if kind not in cached_name.lower() or not cached_name.lower().endswith(".dds"):
                continue
            # Prefer files matching the fabric name
            if any(kw in cached_name.lower() for kw in keywords):
                candidates.append((cached_name, data))
        # Fallback: any DDS with the right kind
        if not candidates:
            for cached_name, data in sorted(image_cache.items()):
                if kind in cached_name.lower() and cached_name.lower().endswith(".dds"):
                    candidates.append((cached_name, data))
        if not candidates:
            return None
        pick = (idx if idx is not None else 0) % len(candidates)
        return candidates[pick][1]

    # ── renderer setup ──
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1024, 1024)
    if debug:
        renWin.SetOffScreenRendering(True)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0.18, 0.20, 0.25)
    ren.SetBackground2(0.42, 0.47, 0.58)
    ren.GradientBackgroundOn()
    renWin.AddRenderer(ren)

    iren = None
    if not debug:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.SetRenderWindow(renWin)

    # ── garment patterns ──
    for pattern in scene.garment_patterns:
        num_verts = pattern.vertex_count
        num_faces = pattern.triangle_count
        print(f"Pattern: {pattern.name}  verts={num_verts}  faces={num_faces}  mat={pattern.material_index}")

        v = np.array(pattern.positions).reshape(num_verts, 3)
        f = np.array(pattern.indices).reshape(num_faces, 3)

        # UVs
        uv_data = None
        raw_uv = np.array(pattern.uvs)
        if pattern.uv_vertex_count == num_verts and raw_uv.size == num_verts * 2:
            uv_data = raw_uv.reshape(num_verts, 2)

        mat_idx = pattern.material_index
        if mat_idx < 0 and materials:
            mat_idx = 0
        mat = materials[mat_idx] if 0 <= mat_idx < len(materials) else None

        if uv_data is not None and mat:
            uv_data = apply_texture_transform(uv_data, mat)

        polydata = make_polydata(v, f, uvs=uv_data)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # ── apply material ──
        if mat:
            prop = actor.GetProperty()

            # Resolve diffuse / base color texture first (needed for color setup)
            has_base_tex = False
            if uv_data is not None:
                tex_bytes = resolve_texture(mat.diffuse_texture_path)
                tex_label = mat.diffuse_texture_path
                if not tex_bytes:
                    tex_bytes = find_substance_texture(mat, "basecolor", colorway_idx)
                    tex_label = "(substance basecolor)"
                if tex_bytes:
                    is_pbr = mat.use_metalness_roughness_pbr
                    tex = texture_from_bytes(tex_bytes, srgb=is_pbr)
                    if tex:
                        has_base_tex = True
                        if is_pbr:
                            prop.SetBaseColorTexture(tex)
                        else:
                            actor.SetTexture(tex)
                        print(f"  + diffuse texture: {tex_label}")
                elif mat.diffuse_texture_path:
                    print(f"  - diffuse texture not found: {mat.diffuse_texture_path}")

            apply_material_colors(prop, mat, has_base_texture=has_base_tex)
            apply_pbr(prop, mat)

            # Normal map
            if mat.normal_texture_path and uv_data is not None:
                tex_bytes = resolve_texture(mat.normal_texture_path)
                if tex_bytes:
                    tex = texture_from_bytes(tex_bytes)
                    if tex:
                        prop.SetTexture("normalTex", tex)
                        prop.SetNormalScale(mat.normal_intensity_percent / 100.0)
                        print(f"  + normal texture: {mat.normal_texture_path}")

            # Roughness/ORM map
            if mat.roughness_texture_path and uv_data is not None:
                tex_bytes = resolve_texture(mat.roughness_texture_path)
                if tex_bytes:
                    tex = texture_from_bytes(tex_bytes)
                    if tex:
                        prop.SetTexture("materialTex", tex)
                        print(f"  + roughness texture: {mat.roughness_texture_path}")

            # Emissive map
            if mat.emissive_texture_path and uv_data is not None:
                tex_bytes = resolve_texture(mat.emissive_texture_path)
                if tex_bytes:
                    tex = texture_from_bytes(tex_bytes)
                    if tex:
                        prop.SetTexture("emissiveTex", tex)
                        print(f"  + emissive texture: {mat.emissive_texture_path}")

        # ── print textures (graphics on the pattern) ──
        for gfx in pattern.print_textures:
            if not gfx.visible or not gfx.texture_path:
                continue
            tex_bytes = resolve_texture(gfx.texture_path)
            if tex_bytes:
                print(f"  + print graphic: {gfx.texture_path} ({gfx.graphic_type_str})")
                if actor.GetTexture() is None and uv_data is not None:
                    tex = texture_from_bytes(tex_bytes)
                    if tex:
                        actor.SetTexture(tex)

        ren.AddActor(actor)

    # ── sewing visualization ──
    pattern_verts = []
    for pattern in scene.garment_patterns:
        v = np.array(pattern.positions).reshape(pattern.vertex_count, 3)
        pattern_verts.append(v)

    if scene.has_seam_pair_group_data:
        sew_points = vtk.vtkPoints()
        sew_lines = vtk.vtkCellArray()
        pt_idx = 0

        for group in scene.seam_line_pair_groups:
            if not group.active:
                continue
            for pair in group.pairs:
                pi0 = pair.param_line_0.pattern_self_index
                pi1 = pair.param_line_1.pattern_self_index
                idx0 = list(pair.mesh_point_indices_0)
                idx1 = list(pair.mesh_point_indices_1)
                if not idx0 or not idx1 or pi0 >= len(pattern_verts) or pi1 >= len(pattern_verts):
                    continue
                v0 = pattern_verts[pi0]
                v1 = pattern_verts[pi1]
                n = min(len(idx0), len(idx1))
                for i in range(n):
                    if idx0[i] < len(v0) and idx1[i] < len(v1):
                        sew_points.InsertNextPoint(v0[idx0[i]])
                        sew_points.InsertNextPoint(v1[idx1[i]])
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, pt_idx)
                        line.GetPointIds().SetId(1, pt_idx + 1)
                        sew_lines.InsertNextCell(line)
                        pt_idx += 2

        sew_polydata = vtk.vtkPolyData()
        sew_polydata.SetPoints(sew_points)
        sew_polydata.SetLines(sew_lines)

        sew_mapper = vtk.vtkPolyDataMapper()
        sew_mapper.SetInputData(sew_polydata)
        sew_actor = vtk.vtkActor()
        sew_actor.SetMapper(sew_mapper)
        sew_actor.GetProperty().SetColor(1.0, 0.3, 0.3)
        sew_actor.GetProperty().SetLineWidth(2)
        ren.AddActor(sew_actor)
        print(f"Sewing: {sew_lines.GetNumberOfCells()} lines from {len(scene.seam_line_pair_groups)} groups")

    ren.ResetCamera()
    renWin.Render()

    if debug:
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(renWin)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        out_path = "tmp/debug_render.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer.SetFileName(out_path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"Debug image saved: {os.path.abspath(out_path)}")
    else:
        iren.Start()


if __name__ == "__main__":
    main()
