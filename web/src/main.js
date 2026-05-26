import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';

import {
  buildSceneActors, buildFloorDisc, computeForegroundBounds,
} from './geometry.js';

const RENDER_W = 1280;
const RENDER_H = 704;
const BG_PRESETS = ['image_1', 'image_2', 'image_5', 'image_10'];
const GBUFFER_NAMES = ['basecolor', 'normal', 'depth', 'roughness'];

let activeBgPreset = null;

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

let cachedWasmModule = null;

async function getWasmModule() {
  if (cachedWasmModule) return cachedWasmModule;
  const loaderUrl = new URL('/zprj_loader.js', window.location.origin).href;
  const createModule = (await import(/* @vite-ignore */ loaderUrl)).default;
  cachedWasmModule = await createModule({
    locateFile: (path) => path.endsWith('.wasm') ? '/zprj_loader.wasm' : path,
  });
  return cachedWasmModule;
}

async function parseZprjBuffer(buf) {
  const mod = await getWasmModule();
  const uint8 = new Uint8Array(buf);
  const ptr = mod._malloc(uint8.byteLength);
  mod.HEAPU8.set(uint8, ptr);
  let scene;
  try {
    scene = mod.parseFromBuffer(ptr, uint8.byteLength);
  } finally {
    mod._free(ptr);
  }
  if (!scene.valid) throw new Error(scene.error || 'Failed to parse zprj');
  return scene;
}

async function loadZprjFromUrl(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const buf = await resp.arrayBuffer();
  return parseZprjBuffer(buf);
}

// ── Panel creation ──────────────────────────────────────────────────

function createPanel(panelEl, bgColor = [0, 0, 0]) {
  const canvasWrap = panelEl.querySelector('.panel-canvas-wrap');
  const grw = vtkGenericRenderWindow.newInstance({ background: bgColor });
  grw.setContainer(canvasWrap);
  grw.resize();
  return {
    panelEl, grw,
    renderer: grw.getRenderer(),
    renderWindow: grw.getRenderWindow(),
    openGLRenderWindow: grw.getApiSpecificRenderWindow(),
    floorActor: null,
    defaultBg: [bgColor[0], bgColor[1], bgColor[2], 1],
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

// ── Camera framing (1280:704 horizontal aspect) ─────────────────────

function frameCameras(panels, actorsData, fovDeg = 15) {
  const { minP, maxP } = computeForegroundBounds(actorsData);
  const cx = (minP[0] + maxP[0]) / 2;
  const cy = (minP[1] + maxP[1]) / 2;
  const cz = (minP[2] + maxP[2]) / 2;
  const sx = maxP[0] - minP[0];
  const sy = maxP[1] - minP[1];
  const sz = maxP[2] - minP[2];
  const aspect = RENDER_W / RENDER_H;
  // VTK view angle is vertical FOV
  const halfVFov = ((fovDeg * Math.PI) / 180) / 2;
  const distY = sy / 2 / Math.tan(halfVFov);
  const distX = (sx / 2) / (Math.tan(halfVFov) * aspect);
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
  if (!floorPD) return null;
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
  return actor;
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
  return addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], null);
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
  return addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], NORMAL_SHADER);
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
  return addFloorActor(renderer, floorPD, [1, 1, 1], DEPTH_SHADER);
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
  return addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], null);
}

// ── Offscreen render at exact 1280x704 ──────────────────────────────

async function captureAtTargetSize(panel) {
  // captureNextImage({size: [W, H]}) renders into an offscreen canvas
  // at the requested size, leaving the main panel untouched.
  const promise = panel.openGLRenderWindow.captureNextImage('image/png', {
    size: [RENDER_W, RENDER_H],
  });
  panel.renderWindow.render();
  return promise;
}

async function dataUrlToImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}

async function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

// Composite background + foreground capture into a single PNG blob.
// `bgUrl` may be null to skip background.
async function compositeCapture(captureDataUrl, bgUrl) {
  const canvas = document.createElement('canvas');
  canvas.width = RENDER_W;
  canvas.height = RENDER_H;
  const ctx = canvas.getContext('2d');
  if (bgUrl) {
    try {
      const bg = await loadImage(bgUrl);
      ctx.drawImage(bg, 0, 0, RENDER_W, RENDER_H);
    } catch (e) {
      console.warn('Failed to load bg', bgUrl, e);
    }
  }
  const fg = await dataUrlToImage(captureDataUrl);
  ctx.drawImage(fg, 0, 0, RENDER_W, RENDER_H);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
}

// ── Background preset ──────────────────────────────────────────────

function setBackgroundPreset(panels, presetName) {
  activeBgPreset = presetName;
  // presetName: null = clear (floor + opaque bg), else folder name (transparent bg, no floor)
  for (const p of panels) {
    const img = p.panelEl.querySelector('.panel-bg-img');
    if (presetName === null) {
      if (img) { img.classList.remove('visible'); img.src = ''; }
      p.renderer.setBackground(...p.defaultBg);
      if (p.floorActor) p.floorActor.setVisibility(true);
    } else {
      if (img) {
        const buf = img.dataset.buf;
        img.src = `/inverse/${presetName}/${buf}.png`;
        img.classList.add('visible');
      }
      // Transparent renderer bg so the <img> behind canvas shows through
      p.renderer.setBackground(0, 0, 0, 0);
      if (p.floorActor) p.floorActor.setVisibility(false);
    }
    p.renderWindow.render();
  }
}

// ── Render via server ──────────────────────────────────────────────

function showModal(status) {
  const modal = document.getElementById('result-modal');
  const statusEl = document.getElementById('result-modal-status');
  const img = document.getElementById('result-modal-img');
  modal.classList.add('visible');
  img.style.display = 'none';
  statusEl.innerHTML = `<span class="spinner"></span><span>${status}</span>`;
}

function setModalStatus(text) {
  document.getElementById('result-modal-status').innerHTML =
    `<span class="spinner"></span><span>${text}</span>`;
}

function setModalResult(url) {
  const img = document.getElementById('result-modal-img');
  const statusEl = document.getElementById('result-modal-status');
  img.src = url;
  img.style.display = '';
  statusEl.textContent = '';
}

function setModalError(msg) {
  document.getElementById('result-modal-status').innerHTML =
    `<span style="color:#e66">Error: ${msg}</span>`;
}

function hideModal() {
  document.getElementById('result-modal').classList.remove('visible');
}

async function renderViaServer(panels) {
  const renderBtn = document.getElementById('render-btn');
  renderBtn.disabled = true;
  showModal('Capturing G-buffers...');

  try {
    const formData = new FormData();
    for (let i = 0; i < panels.length; i++) {
      const name = GBUFFER_NAMES[i];
      const dataUrl = await captureAtTargetSize(panels[i]);
      const bgUrl = activeBgPreset
        ? `/inverse/${activeBgPreset}/${name}.png`
        : null;
      const blob = await compositeCapture(dataUrl, bgUrl);
      formData.append(name, blob, `${name}.png`);
    }

    setModalStatus('Saving on server...');
    const resp = await fetch('/render', { method: 'POST', body: formData });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Server error ${resp.status}: ${text}`);
    }
    const info = await resp.json();
    const statusEl = document.getElementById('result-modal-status');
    const img = document.getElementById('result-modal-img');
    img.style.display = 'none';
    statusEl.innerHTML = `<span style="color:#9c9">Saved to ${info.saved_dir}/ (${info.files.join(', ')})</span>`;
  } catch (err) {
    console.error(err);
    setModalError(err.message);
  } finally {
    renderBtn.disabled = false;
  }
}

// ── Build & populate scene ──────────────────────────────────────────

async function buildAndPopulate(scene, panels) {
  const actorsData = await buildSceneActors(scene);
  const floorPD = buildFloorDisc(actorsData);

  // Clear existing actors
  for (const p of panels) {
    p.renderer.removeAllViewProps();
    p.floorActor = null;
  }

  panels[0].floorActor = populateBasecolor(panels[0].renderer, actorsData, floorPD);
  panels[1].floorActor = populateNormal(panels[1].renderer, actorsData, floorPD);
  panels[2].floorActor = populateDepth(panels[2].renderer, actorsData, floorPD);
  panels[3].floorActor = populateRoughness(panels[3].renderer, actorsData, floorPD);

  frameCameras(panels, actorsData);
  for (const p of panels) p.renderWindow.render();
}

// ── Main ────────────────────────────────────────────────────────────

function setStatus(msg) {
  document.getElementById('loading').textContent = msg;
}

async function main() {
  setStatus('Initializing renderers...');
  const panels = [
    createPanel(document.getElementById('panel-basecolor'), [0.5, 0.5, 0.5]),
    createPanel(document.getElementById('panel-normal'), [0.5, 0.5, 1.0]),
    createPanel(document.getElementById('panel-depth'), [1, 1, 1]),
    createPanel(document.getElementById('panel-roughness'), [0.5, 0.5, 0.5]),
  ];

  setStatus('Loading WASM module + sample scene...');
  const scene = await loadZprjFromUrl('/sample.zprj');

  setStatus('Building geometry...');
  await buildAndPopulate(scene, panels);

  syncCameras(panels);

  // Background preset thumbnails
  const thumbs = document.getElementById('bg-thumbs');
  const clearThumb = document.createElement('div');
  clearThumb.className = 'bg-thumb clear active';
  clearThumb.textContent = 'None';
  clearThumb.dataset.preset = '';
  thumbs.appendChild(clearThumb);
  for (const preset of BG_PRESETS) {
    const t = document.createElement('div');
    t.className = 'bg-thumb';
    t.style.backgroundImage = `url(/inverse/${preset}/rgb_input.png)`;
    t.dataset.preset = preset;
    t.title = preset;
    thumbs.appendChild(t);
  }
  thumbs.addEventListener('click', (e) => {
    const t = e.target.closest('.bg-thumb');
    if (!t) return;
    for (const sib of thumbs.children) sib.classList.remove('active');
    t.classList.add('active');
    setBackgroundPreset(panels, t.dataset.preset || null);
  });

  // Render button
  document.getElementById('render-btn').addEventListener('click', () => {
    renderViaServer(panels);
  });

  // Modal close
  document.getElementById('result-modal-close').addEventListener('click', hideModal);
  document.getElementById('result-modal').addEventListener('click', (e) => {
    if (e.target.id === 'result-modal') hideModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideModal();
  });

  // File upload
  const fileInput = document.getElementById('zprj-file');
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStatus('Loading ' + file.name + '...');
    document.getElementById('loading').classList.remove('hidden');
    try {
      const buf = await file.arrayBuffer();
      const newScene = await parseZprjBuffer(buf);
      await buildAndPopulate(newScene, panels);
      document.getElementById('zprj-name').textContent = file.name;
    } catch (err) {
      console.error(err);
      alert('Failed to load: ' + err.message);
    } finally {
      document.getElementById('loading').classList.add('hidden');
    }
  });

  document.getElementById('loading').classList.add('hidden');

  window.addEventListener('resize', () => {
    for (const p of panels) p.grw.resize();
  });
}

main().catch((err) => {
  document.getElementById('loading').textContent = 'Error: ' + err.message;
  console.error(err);
});
