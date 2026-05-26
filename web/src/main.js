import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';

import {
  buildSceneActors, buildFloorDisc, computeForegroundBounds,
} from './geometry.js';

function setShaderReplacements(mapper, replacements) {
  const vsp = mapper.getViewSpecificProperties();
  vsp.OpenGL = vsp.OpenGL || {};
  vsp.OpenGL.ShaderReplacements = replacements.map((r) => ({
    shaderType: r.shaderType,
    originalValue: r.originalValue,
    replacementValue: r.replacementValue,
    replaceFirst: r.replaceFirst || false,
  }));
}

// ── WASM loader ─────────────────────────────────────────────────────

async function loadWasmScene(url) {
  // Dynamic import of the WASM loader from public/ — must use variable to skip Rollup
  const loaderUrl = new URL('/zprj_loader.js', window.location.origin).href;
  const createModule = (await import(/* @vite-ignore */ loaderUrl)).default;
  const wasmModule = await createModule({
    locateFile: (path) => path.endsWith('.wasm') ? '/zprj_loader.wasm' : path,
  });

  const resp = await fetch(url);
  const buf = await resp.arrayBuffer();
  const uint8 = new Uint8Array(buf);
  const ptr = wasmModule._malloc(uint8.byteLength);
  wasmModule.HEAPU8.set(uint8, ptr);

  let scene;
  try {
    scene = wasmModule.parseFromBuffer(ptr, uint8.byteLength);
  } finally {
    wasmModule._free(ptr);
  }

  if (!scene.valid) {
    throw new Error(scene.error || 'Failed to parse zprj');
  }

  return scene;
}

// ── Panel creation ──────────────────────────────────────────────────

function createPanel(container, bgColor = [0, 0, 0]) {
  const grw = vtkGenericRenderWindow.newInstance();
  grw.setContainer(container);
  grw.resize();
  grw.getRenderer().setBackground(...bgColor);
  return {
    grw,
    renderer: grw.getRenderer(),
    renderWindow: grw.getRenderWindow(),
  };
}

// ── Camera sync ─────────────────────────────────────────────────────

function syncCameras(panels) {
  let syncing = false;

  function copyCamera(srcPanel) {
    if (syncing) return;
    syncing = true;
    const srcCam = srcPanel.renderer.getActiveCamera();
    const pos = srcCam.getPosition();
    const fp = srcCam.getFocalPoint();
    const vu = srcCam.getViewUp();
    const va = srcCam.getViewAngle();
    const cr = srcCam.getClippingRange();
    for (const p of panels) {
      if (p === srcPanel) continue;
      const cam = p.renderer.getActiveCamera();
      cam.setPosition(...pos);
      cam.setFocalPoint(...fp);
      cam.setViewUp(...vu);
      cam.setViewAngle(va);
      cam.setClippingRange(...cr);
      p.renderWindow.render();
    }
    syncing = false;
  }

  for (const panel of panels) {
    panel.renderer.getActiveCamera().onModified(() => copyCamera(panel));
  }
}

// ── Camera framing ──────────────────────────────────────────────────

function frameCameras(panels, actorsData, fovDeg = 15) {
  const { minP, maxP } = computeForegroundBounds(actorsData);
  const cx = (minP[0] + maxP[0]) / 2;
  const cy = (minP[1] + maxP[1]) / 2;
  const cz = (minP[2] + maxP[2]) / 2;
  const sx = maxP[0] - minP[0];
  const sy = maxP[1] - minP[1];
  const sz = maxP[2] - minP[2];
  const halfFov = ((fovDeg * Math.PI) / 180) / 2;
  const distY = sy / 2 / Math.tan(halfFov);
  const distX = sx / 2 / Math.tan(halfFov);
  const dist = Math.max(distY, distX) * 1.05;
  const span = Math.max(sx, sy, sz);

  for (const p of panels) {
    const cam = p.renderer.getActiveCamera();
    cam.setPosition(cx, cy, cz + dist);
    cam.setFocalPoint(cx, cy, cz);
    cam.setViewUp(0, 1, 0);
    cam.setViewAngle(fovDeg);
    cam.setClippingRange(Math.max(dist - span, 0.1), dist * 2.5);
  }
}

// ── Shader configs ──────────────────────────────────────────────────

const NORMAL_SHADER = [
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: `
      vec3 n = normalize(normalVCVSOutput);
      if (!gl_FrontFacing) n = -n;
      gl_FragData[0] = vec4(n * 0.5 + 0.5, 1.0);
    `,
    replaceFirst: true,
  },
];

const DEPTH_SHADER = [
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: `
      float d = gl_FragCoord.z;
      gl_FragData[0] = vec4(vec3(d), 1.0);
    `,
    replaceFirst: true,
  },
];

// ── Floor helper ────────────────────────────────────────────────────

function addFloorActor(renderer, floorPD, color, shaderReplacements) {
  if (!floorPD) return;
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(floorPD);
  if (shaderReplacements) setShaderReplacements(mapper, shaderReplacements);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setAmbient(1.0);
  actor.getProperty().setDiffuse(0.0);
  actor.getProperty().setSpecular(0.0);
  actor.getProperty().setColor(...color);
  renderer.addActor(actor);
}

// ── Panel populate functions ────────────────────────────────────────

function populateBasecolor(renderer, actorsData, floorPD) {
  for (const ad of actorsData) {
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(ad.polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    const prop = actor.getProperty();
    prop.setAmbient(1.0);
    prop.setDiffuse(0.0);
    prop.setSpecular(0.0);
    prop.setColor(...ad.diffuseColor);
    if (ad.diffuseTex) actor.addTexture(ad.diffuseTex);
    renderer.addActor(actor);
  }
  addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], null);
}

function populateNormal(renderer, actorsData, floorPD) {
  for (const ad of actorsData) {
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(ad.polyData);
    setShaderReplacements(mapper, NORMAL_SHADER);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    renderer.addActor(actor);
  }
  addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], NORMAL_SHADER);
}

function populateDepth(renderer, actorsData, floorPD) {
  for (const ad of actorsData) {
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(ad.polyData);
    setShaderReplacements(mapper, DEPTH_SHADER);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    renderer.addActor(actor);
  }
  addFloorActor(renderer, floorPD, [1, 1, 1], DEPTH_SHADER);
}

function populateRoughness(renderer, actorsData, floorPD) {
  for (const ad of actorsData) {
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(ad.polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    const prop = actor.getProperty();
    prop.setAmbient(1.0);
    prop.setDiffuse(0.0);
    prop.setSpecular(0.0);
    const rv = ad.roughness;
    prop.setColor(rv, rv, rv);
    if (ad.roughnessTex) actor.addTexture(ad.roughnessTex);
    renderer.addActor(actor);
  }
  addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], null);
}

// ── Main ────────────────────────────────────────────────────────────

function setStatus(msg) {
  document.getElementById('loading').textContent = msg;
}

async function main() {
  setStatus('Loading WASM module...');
  const scene = await loadWasmScene('/sample.zprj');

  setStatus('Building geometry...');
  const actorsData = await buildSceneActors(scene);
  const floorPD = buildFloorDisc(actorsData);

  setStatus('Initializing renderers...');
  const panels = [
    createPanel(document.getElementById('panel-basecolor'), [0.5, 0.5, 0.5]),
    createPanel(document.getElementById('panel-normal'), [0.5, 0.5, 1.0]),
    createPanel(document.getElementById('panel-depth'), [1, 1, 1]),
    createPanel(document.getElementById('panel-roughness'), [0.5, 0.5, 0.5]),
  ];

  populateBasecolor(panels[0].renderer, actorsData, floorPD);
  populateNormal(panels[1].renderer, actorsData, floorPD);
  populateDepth(panels[2].renderer, actorsData, floorPD);
  populateRoughness(panels[3].renderer, actorsData, floorPD);

  frameCameras(panels, actorsData);
  syncCameras(panels);

  for (const p of panels) p.renderWindow.render();

  document.getElementById('loading').classList.add('hidden');

  window.addEventListener('resize', () => {
    for (const p of panels) p.grw.resize();
  });
}

main().catch((err) => {
  document.getElementById('loading').textContent = 'Error: ' + err.message;
  console.error(err);
});
