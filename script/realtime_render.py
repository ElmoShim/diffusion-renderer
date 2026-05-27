"""4-panel interactive G-buffer viewer for .zprj files.

Panels:
  Top-left:     Basecolor  (unlit, texture-mapped color)
  Top-right:    Normal     (view-space normals encoded as RGB)
  Bottom-left:  Depth      (NDC depth, brighter = closer)
  Bottom-right: Roughness  (material roughness, grayscale)

All panels share one camera — navigate in any panel to move all.

Usage: uv run script/realtime_render.py <file.zprj>
"""

import sys
import argparse
from pathlib import Path

import vtk

import zprj_loader

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.utils_render_vtk import (
    build_scene_actors,
    populate_gbuffer_renderer,
    enable_translucency,
    _setup_camera,
)


DEPTH_VERT = """
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
}
"""

DEPTH_FRAG = """
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

  float d = 1.0 - gl_FragCoord.z;
  fragOutput0 = vec4(d, d, d, 1.0);

  if (gl_FragData[0].a <= 0.0) discard;
  //VTK::DepthPeeling::Impl
  //VTK::Picking::Impl
  //VTK::Coincident::Impl
}
"""

# (gbuffer_type, viewport_xmin_ymin_xmax_ymax, label)
PANELS = [
    ("basecolor", (0.0, 0.5, 0.5, 1.0), "Basecolor"),
    ("normal",    (0.5, 0.5, 1.0, 1.0), "Normal"),
    ("depth",     (0.0, 0.0, 0.5, 0.5), "Depth"),
    ("roughness", (0.5, 0.0, 1.0, 0.5), "Roughness"),
]


def _add_label(renderer, text):
    label = vtk.vtkTextActor()
    label.SetInput(text)
    label.GetTextProperty().SetFontSize(18)
    label.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    label.GetTextProperty().SetBold(True)
    label.GetTextProperty().SetShadow(True)
    label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    label.GetPositionCoordinate().SetValue(0.02, 0.03)
    renderer.AddActor2D(label)


def _build_renderer(gbuffer_type, actors_data, scene, tex_cache):
    ren = vtk.vtkRenderer()

    if gbuffer_type == "depth":
        # Interactive depth uses a per-fragment NDC-depth shader (the offscreen
        # G-buffer renderer captures the real z-buffer instead).
        ren.SetBackground(0.0, 0.0, 0.0)
        sp = vtk.vtkShaderProperty()
        sp.SetVertexShaderCode(DEPTH_VERT)
        sp.SetFragmentShaderCode(DEPTH_FRAG)
        for ad in actors_data:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(ad["polydata"])
            mapper.SetScalarVisibility(False)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetShaderProperty(sp)
            ren.AddActor(actor)
        return ren

    # basecolor / normal / roughness share the offscreen renderer's actor setup
    # (textures, opacity maps, shaders) so the viewer matches render_gbuffers.
    populate_gbuffer_renderer(ren, gbuffer_type, actors_data, scene, tex_cache)
    enable_translucency(ren)
    return ren


def main():
    parser = argparse.ArgumentParser(description="4-panel G-buffer viewer for .zprj files")
    parser.add_argument("zprj_file", help="Path to .zprj file")
    args = parser.parse_args()

    print(f"Loading {args.zprj_file}...")
    scene = zprj_loader.parse(args.zprj_file)
    if not scene.valid:
        print(f"Error: {scene.error}")
        sys.exit(1)

    print("Building scene geometry...")
    actors_data = build_scene_actors(scene, background=False)
    if not actors_data:
        print("Error: No geometry found in scene")
        sys.exit(1)
    print(f"  {len(actors_data)} mesh(es) loaded")

    tex_cache = {}

    win = vtk.vtkRenderWindow()
    win.SetSize(1280, 960)
    win.SetWindowName(f"G-Buffer Viewer — {Path(args.zprj_file).name}")
    # Depth peeling (for opacity-mapped fabrics) needs an alpha buffer and no MSAA.
    win.SetAlphaBitPlanes(True)
    win.SetMultiSamples(0)

    renderers = []
    for gbuf_type, viewport, label_text in PANELS:
        ren = _build_renderer(gbuf_type, actors_data, scene, tex_cache)
        ren.SetViewport(*viewport)
        _add_label(ren, label_text)
        if gbuf_type == "basecolor":
            ren.UseFXAAOn()
        win.AddRenderer(ren)
        renderers.append(ren)

    # Set up initial camera on first renderer, then share it with all others.
    # Each panel is 640x480 (half of 1280x960), aspect = 4/3.
    _setup_camera(renderers[0], actors_data, fov_deg=20.0, aspect=640 / 480)
    shared_cam = renderers[0].GetActiveCamera()
    for ren in renderers[1:]:
        ren.SetActiveCamera(shared_cam)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(win)
    win.Render()
    iren.Start()


if __name__ == "__main__":
    main()
