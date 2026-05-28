import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';

import {
  buildSceneActors, buildFloorDisc, computeForegroundBounds,
} from './geometry.js';
import { decodeRGBE, tonemapToImageData } from './rgbe.js';

const RENDER_W = 1280;
const RENDER_H = 704;
const BG_PRESETS = []; // populated at runtime from /bg_presets
const HDR_PRESETS = ['sunny_vondelpark_1k', 'pink_sunrise_1k', 'street_lamp_1k', 'circus_arena_1k'];
const GBUFFER_NAMES = ['basecolor', 'normal', 'depth', 'roughness'];

// Per-tab session ID for BG uploads — only this tab can see its own uploads.
// sessionStorage survives reloads but not tab close; a new tab gets a new id.
const BG_SID = (() => {
  let s = sessionStorage.getItem('bg_sid');
  if (!s) {
    s = (crypto.randomUUID ? crypto.randomUUID()
                           : ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
                               (c ^ (Math.random() * 16 >> c / 4)).toString(16)));
    sessionStorage.setItem('bg_sid', s);
  }
  return s;
})();
const BG_SID_PARAM = `sid=${encodeURIComponent(BG_SID)}`;

let activeBgPreset = null;
let activeHdr = HDR_PRESETS[0];
let currentActorsData = null;
let activeFov = 15;

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
  // NOTE: the normal/roughness/depth passes render opacity-mapped materials as
  // opaque alpha-test cutouts (see OPACITY_CUTOFF), NOT translucent — vtk.js has
  // no depth peeling, only a single-pass weighted-blended OIT that washes out
  // high-depth-complexity hair. These depth-peeling calls are vtk.js no-ops (kept
  // harmless). The basecolor pass is the exception: it stays translucent
  // (forceTranslucent + OIT) so knit gaps blend over the background.
  const renderer = grw.getRenderer();
  renderer.setUseDepthPeeling(true);
  renderer.setMaximumNumberOfPeels(8);
  renderer.setOcclusionRatio(0);
  return {
    panelEl, grw,
    renderer,
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

// Opacity is handled as an ALPHA-TEST CUTOUT, not translucent blending: vtk.js
// has no depth peeling, only weighted-blended OIT (a single-pass approximation
// that under-accumulates opacity for high-depth-complexity geometry like hair —
// many overlapping sparse-alpha cards stay see-through). Rendering opaque and
// discarding sub-threshold fragments lets the z-buffer occlude layer-over-layer,
// so the union of opaque strands fills in dense, matching Python's depth-peeled
// density. Tune this if hair/knit reads too sparse (lower) or too solid (higher).
const OPACITY_CUTOFF = 0.25;

// Output the view-space normal as color. Camera looks along -Z, so a
// front-facing normal has z > 0; flip back-facing ones toward the camera.
const NORMAL_BODY =
  'vec3 n = normalize(normalVCVSOutput);\n' +
  'if (n.z < 0.0) n = -n;';

const NORMAL_SHADER = [
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: `${NORMAL_BODY}\n      gl_FragData[0] = vec4(n * 0.5 + 0.5, 1.0);`,
    replaceFirst: true,
  },
];

// Opacity variant with NO normal map: discard opacity-map holes (alpha-test
// cutout), render the surface opaque. Used for materials that have an opacity
// map but no normal map (otherwise NORMAL_MAP_OPACITY_SHADER applies).
const NORMAL_OPACITY_SHADER = [
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: `if (texture2D(texture1, tcoordVCVSOutput).a < ${OPACITY_CUTOFF}) discard;\n      ${NORMAL_BODY}\n      gl_FragData[0] = vec4(n * 0.5 + 0.5, 1.0);`,
    replaceFirst: true,
  },
];

// ── Normal-map (tangent-space) variants ─────────────────────────────
// vtk.js has no built-in normal mapping (no PBR interpolation / SetNormalTexture),
// so we perturb the view-space normal in the shader with a TBN built from the
// per-vertex tangents geometry.js attaches (point-data array "tangent", bound as
// the custom vertex attribute `tangentMC` via mapper.setCustomShaderAttributes).
// This matches Python's SetNormalTexture path, which is what gives hair / fabric
// their surface detail in the normal G-buffer. The normal map (RGB) and optional
// opacity (A) are packed into one texture (texture1) by loadNormalComposite.
//
// All replacements use replaceFirst:true so they apply in vtk.js's PRE pass while
// the //VTK::Normal::* hooks are still present; each re-includes its hook so
// vtk.js's own normalVCVSOutput / normalMatrix / normalMC injection still runs.
const NORMAL_MAP_VS = [
  {
    shaderType: 'Vertex',
    originalValue: '//VTK::Normal::Dec',
    replacementValue: '//VTK::Normal::Dec\n      attribute vec4 tangentMC;\n      varying vec3 tangentVCVSOutput;\n      varying float tangentWVSOutput;',
    replaceFirst: true,
  },
  {
    shaderType: 'Vertex',
    originalValue: '//VTK::Normal::Impl',
    replacementValue: '//VTK::Normal::Impl\n      tangentVCVSOutput = normalMatrix * tangentMC.xyz;\n      tangentWVSOutput = tangentMC.w;',
    replaceFirst: true,
  },
];

const NORMAL_MAP_FS_DEC = {
  shaderType: 'Fragment',
  originalValue: '//VTK::Normal::Dec',
  replacementValue: '//VTK::Normal::Dec\n      varying vec3 tangentVCVSOutput;\n      varying float tangentWVSOutput;',
  replaceFirst: true,
};

// cutout: when true, discard opacity-map holes (alpha-test) before shading and
// render opaque — same cutout the other passes use. Nv (vtk.js's normalized,
// front-face-flipped view normal) + the per-vertex tangent form the TBN; the
// tangent-space normal map is decoded (rgb*2-1), rotated into view space, then
// flipped toward the camera (n.z < 0) like the geometric pass.
function normalMapShader(cutout) {
  return [
    ...NORMAL_MAP_VS,
    NORMAL_MAP_FS_DEC,
    {
      shaderType: 'Fragment',
      originalValue: '//VTK::Light::Impl',
      replacementValue: [
        'vec4 nmap = texture2D(texture1, tcoordVCVSOutput);',
        cutout ? `if (nmap.a < ${OPACITY_CUTOFF}) discard;` : '',
        'vec3 Nv = normalize(normalVCVSOutput);',
        'vec3 Tv = normalize(tangentVCVSOutput - dot(tangentVCVSOutput, Nv) * Nv);',
        'vec3 Bv = cross(Nv, Tv) * tangentWVSOutput;',
        'vec3 nt = nmap.xyz * 2.0 - 1.0;',
        'vec3 nn = normalize(mat3(Tv, Bv, Nv) * nt);',
        'if (nn.z < 0.0) nn = -nn;',
        'gl_FragData[0] = vec4(nn * 0.5 + 0.5, 1.0);',
      ].filter(Boolean).join('\n      '),
      replaceFirst: true,
    },
  ];
}

const NORMAL_MAP_SHADER = normalMapShader(false);
const NORMAL_MAP_OPACITY_SHADER = normalMapShader(true);

// For the roughness G-buffer: keep VTK.js's color/lighting, then discard
// opacity-map holes (alpha-test cutout) so the surface renders opaque and the
// z-buffer occludes overlapping layers (dense hair) instead of OIT washing.
const OPACITY_DISCARD_SHADER = [
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: `//VTK::Light::Impl\n      if (texture2D(texture1, tcoordVCVSOutput).a < ${OPACITY_CUTOFF}) discard;`,
    replaceFirst: true,
  },
];

// For the basecolor G-buffer: keep VTK.js's color/lighting, then override the
// fragment alpha with the opacity map (texture1.a). forceTranslucent + OIT then
// blend knit gaps over the background (translucent, not alpha-test cutout).
const OPACITY_ALPHA_SHADER = [
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: '//VTK::Light::Impl\n      gl_FragData[0].a = texture2D(texture1, tcoordVCVSOutput).a;',
    replaceFirst: true,
  },
];

// Depth state: absolute view-space distances (positive units).
// Output = 1 - clamp((dist-near)/(far-near), 0, 1)  → 1=near, 0=far
const depthState = { near: 1, far: 100 };
// Slider scale: 0..SLIDER_MAX maps to [sliderMin, sliderMax] (view-space units).
// Set by autoFitDepth: ±5× the fitted range gives the user room around it.
const depthSlider = { min: 0, max: 100 };

// near/far are expressed in focal-relative units (0 = focal point).
// Effective range = [focalDist + near, focalDist + far].
const DEPTH_SHADER = [
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Dec',
    replacementValue: `
      //VTK::Light::Dec
      uniform float u_depthNear;
      uniform float u_depthFar;
      uniform float u_focalDist;
    `,
    replaceFirst: true,
  },
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: `
      float distFromFocal = (-vertexVCVSOutput.z) - u_focalDist;
      float d = clamp((distFromFocal - u_depthNear) / max(u_depthFar - u_depthNear, 1e-6), 0.0, 1.0);
      gl_FragData[0] = vec4(vec3(d), 1.0);
    `,
    replaceFirst: true,
  },
];

// Depth is a hard opacity cutout, not a blend: a depth value can't be averaged
// (blending front and behind would invent a distance that is neither), so an
// opacity-map hole is discarded and the depth of whatever is behind the fabric
// (body, floor, far background) shows through. Matches the Python depth pass.
const DEPTH_OPACITY_SHADER = [
  DEPTH_SHADER[0],
  {
    shaderType: 'Fragment',
    originalValue: '//VTK::Light::Impl',
    replacementValue: `
      if (texture2D(texture1, tcoordVCVSOutput).a < ${OPACITY_CUTOFF}) discard;
      float distFromFocal = (-vertexVCVSOutput.z) - u_focalDist;
      float d = clamp((distFromFocal - u_depthNear) / max(u_depthFar - u_depthNear, 1e-6), 0.0, 1.0);
      gl_FragData[0] = vec4(vec3(d), 1.0);
    `,
    replaceFirst: true,
  },
];

function addDepthUniformCallback(mapper) {
  const vsp = mapper.getViewSpecificProperties();
  vsp.ShadersCallbacks = vsp.ShadersCallbacks || [];
  vsp.ShadersCallbacks.push({
    callback: (_userData, cellBO, ren) => {
      const prog = cellBO.getProgram();
      const cam = ren.getActiveCamera();
      const pos = cam.getPosition();
      const fp = cam.getFocalPoint();
      const dx = fp[0] - pos[0], dy = fp[1] - pos[1], dz = fp[2] - pos[2];
      const focalDist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      if (prog.isUniformUsed('u_depthNear')) prog.setUniformf('u_depthNear', depthState.near);
      if (prog.isUniformUsed('u_depthFar')) prog.setUniformf('u_depthFar', depthState.far);
      if (prog.isUniformUsed('u_focalDist')) prog.setUniformf('u_focalDist', focalDist);
    },
  });
}

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
    if (ad.hasOpacity && ad.diffuseTex) {
      // VTK.js's automatic `opacity *= texture1.a` doesn't take effect here, so
      // sample the alpha explicitly in the shader. forceTranslucent + OIT then
      // blend knit gaps over the background (translucent, not alpha-test cutout).
      setShaderReplacements(mapper, OPACITY_ALPHA_SHADER);
      actor.setForceTranslucent(true);
    }
    renderer.addActor(actor);
  }
  return addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], null);
}

function populateNormal(renderer, actorsData, floorPD) {
  for (const ad of actorsData) {
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(ad.polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);

    if (ad.hasNormalMap && ad.normalTex) {
      // Perturb the normal with the material's normal map (TBN from per-vertex
      // tangents), matching Python. normalTex packs the normal map in RGB and the
      // opacity map in A; the opacity variant discards holes (alpha-test, opaque).
      mapper.setCustomShaderAttributes(['tangent']);
      setShaderReplacements(mapper, ad.hasOpacity ? NORMAL_MAP_OPACITY_SHADER : NORMAL_MAP_SHADER);
      actor.addTexture(ad.normalTex);
    } else if (ad.hasOpacity && ad.opacityTex) {
      // Opacity, no normal map: alpha-test cutout (opaque), not translucent.
      setShaderReplacements(mapper, NORMAL_OPACITY_SHADER);
      actor.addTexture(ad.opacityTex);
    } else {
      setShaderReplacements(mapper, NORMAL_SHADER);
    }
    renderer.addActor(actor);
  }
  return addFloorActor(renderer, floorPD, [0.5, 0.5, 0.5], NORMAL_SHADER);
}

function populateDepth(renderer, actorsData, floorPD) {
  for (const ad of actorsData) {
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(ad.polyData);
    const useOpacity = ad.hasOpacity && ad.opacityTex;
    setShaderReplacements(mapper, useOpacity ? DEPTH_OPACITY_SHADER : DEPTH_SHADER);
    addDepthUniformCallback(mapper);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    // Bind the opacity map; the shader discards holes (rendered opaque) so the
    // depth of whatever is behind the fabric shows through — no translucency.
    if (useOpacity) actor.addTexture(ad.opacityTex);
    renderer.addActor(actor);
  }
  const floorActor = addFloorActor(renderer, floorPD, [1, 1, 1], DEPTH_SHADER);
  if (floorActor) addDepthUniformCallback(floorActor.getMapper());
  return floorActor;
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
    // roughnessTex is the composite (roughness/white RGB + opacity alpha) when
    // the fabric has an opacity map, so the alpha shader can read texture1.a.
    const opTex = ad.hasOpacity ? (ad.roughnessTex || ad.opacityTex) : ad.roughnessTex;
    if (opTex) actor.addTexture(opTex);
    if (ad.hasOpacity && opTex) {
      // Alpha-test cutout (opaque), not translucent — see OPACITY_DISCARD_SHADER.
      setShaderReplacements(mapper, OPACITY_DISCARD_SHADER);
    }
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

// ── HDR thumbnail decode ────────────────────────────────────────────

function hdrBufferToCanvas(arrayBuffer, maxW = 256) {
  const hdr = decodeRGBE(arrayBuffer);
  const full = document.createElement('canvas');
  full.width = hdr.width;
  full.height = hdr.height;
  const fctx = full.getContext('2d');
  fctx.putImageData(tonemapToImageData(hdr), 0, 0);
  // Downscale to thumbnail size for memory
  const scale = Math.min(1, maxW / hdr.width);
  const w = Math.round(hdr.width * scale);
  const h = Math.round(hdr.height * scale);
  const thumb = document.createElement('canvas');
  thumb.width = w; thumb.height = h;
  thumb.getContext('2d').drawImage(full, 0, 0, w, h);
  return thumb;
}

async function decodeHdrToCanvas(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
  const buf = await resp.arrayBuffer();
  return hdrBufferToCanvas(buf);
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
        img.src = `/inverse/${presetName}/${buf}.png?${BG_SID_PARAM}`;
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

// In-memory render history. Latest first.
const gallery = [];

function fmtTime(d = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function openModal(item) {
  const modal = document.getElementById('result-modal');
  const img = document.getElementById('result-modal-img');
  const video = document.getElementById('result-modal-video');
  const meta = document.getElementById('result-modal-meta');
  const status = document.getElementById('result-modal-status');
  const title = document.getElementById('result-modal-title');

  title.textContent = item.resultUrl ? 'Rendered Result' : 'Render (in progress)';
  meta.textContent = [
    item.timestamp,
    item.mode && item.mode !== 'still' ? `Mode: ${item.mode}` : null,
    item.hdr ? `HDR: ${item.hdr}` : null,
    item.bg ? `BG: ${item.bg}` : 'BG: none',
  ].filter(Boolean).join(' · ');

  if (item.resultUrl) {
    if (item.resultType === 'video') {
      video.src = item.resultUrl;
      video.style.display = '';
      img.style.display = 'none';
      img.removeAttribute('src');
      video.play().catch(() => {});
    } else {
      img.src = item.resultUrl;
      img.style.display = '';
      video.style.display = 'none';
      video.removeAttribute('src');
    }
    status.textContent = '';
  } else {
    img.removeAttribute('src');
    img.style.display = 'none';
    video.style.display = 'none';
    video.removeAttribute('src');
    status.innerHTML = '<span class="spinner"></span><span>Rendering...</span>';
  }

  // Fill G-buffer thumbs
  for (const tile of document.querySelectorAll('.modal-gbuffer')) {
    const buf = tile.dataset.buf;
    const tileImg = tile.querySelector('img');
    tileImg.src = item.gbuffers[buf] || '';
  }

  modal.classList.add('visible');
  modal._activeItem = item;
}

function hideModal() {
  document.getElementById('result-modal').classList.remove('visible');
}

function refreshGallery() {
  const root = document.getElementById('gallery');
  root.innerHTML = '';
  for (const item of gallery) {
    const el = document.createElement('div');
    el.className = 'gallery-item' + (item.resultUrl ? '' : ' pending');
    if (item.resultUrl) {
      if (item.resultType === 'video') {
        const v = document.createElement('video');
        v.src = item.resultUrl;
        v.autoplay = true; v.loop = true; v.muted = true; v.playsInline = true;
        el.appendChild(v);
      } else {
        const img = document.createElement('img');
        img.src = item.resultUrl;
        el.appendChild(img);
      }
    } else {
      if (item.gbuffers.basecolor) {
        const img = document.createElement('img');
        img.src = item.gbuffers.basecolor;
        img.style.opacity = '0.25';
        el.appendChild(img);
      }
      el.appendChild(makePendingSVG());
    }
    const stamp = document.createElement('div');
    stamp.className = 'stamp';
    stamp.textContent = item.timestamp + (item.hdr ? ` · ${item.hdr}` : '');
    el.appendChild(stamp);
    el.addEventListener('click', () => openModal(item));
    root.appendChild(el);
  }
}

function setupModalHover() {
  const bigImg = document.getElementById('result-modal-img');
  const bigVideo = document.getElementById('result-modal-video');
  let savedImgSrc = null;
  let savedVideoVisible = false;
  for (const tile of document.querySelectorAll('.modal-gbuffer')) {
    tile.addEventListener('mouseenter', () => {
      const item = document.getElementById('result-modal')._activeItem;
      const src = item?.gbuffers[tile.dataset.buf];
      if (!src) return;
      if (savedImgSrc === null) {
        savedImgSrc = bigImg.src;
        savedVideoVisible = bigVideo.style.display !== 'none';
      }
      // Swap to G-buffer (image)
      bigImg.src = src;
      bigImg.style.display = '';
      bigVideo.style.display = 'none';
    });
    tile.addEventListener('mouseleave', () => {
      if (savedImgSrc !== null) {
        bigImg.src = savedImgSrc;
        if (savedVideoVisible) {
          bigImg.style.display = 'none';
          bigVideo.style.display = '';
        }
        savedImgSrc = null;
        savedVideoVisible = false;
      }
    });
  }
}

function makeBlackBlob(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').fillRect(0, 0, w, h); // black by default
  return new Promise((resolve) => c.toBlob(resolve, 'image/png'));
}

function makeBlackDataUrl(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').fillRect(0, 0, w, h);
  return c.toDataURL('image/png');
}

// ── SSE render progress ─────────────────────────────────────────────

// Circular SVG progress: radius 22, stroke-width 4
const _CIRC_R = 22;
const _CIRC_STROKE = 4;
const _CIRC_C = _CIRC_R + _CIRC_STROKE;          // cx = cy = 26
const _CIRC_SIZE = (_CIRC_R + _CIRC_STROKE) * 2; // 52
const _CIRC_CIRCUMFERENCE = 2 * Math.PI * _CIRC_R;

function makePendingSVG() {
  const ns = 'http://www.w3.org/2000/svg';
  const wrap = document.createElement('div');
  wrap.className = 'gallery-progress';

  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', _CIRC_SIZE);
  svg.setAttribute('height', _CIRC_SIZE);
  svg.setAttribute('viewBox', `0 0 ${_CIRC_SIZE} ${_CIRC_SIZE}`);

  // Track circle
  const track = document.createElementNS(ns, 'circle');
  track.setAttribute('cx', _CIRC_C);
  track.setAttribute('cy', _CIRC_C);
  track.setAttribute('r', _CIRC_R);
  track.setAttribute('fill', 'none');
  track.setAttribute('stroke', 'rgba(255,255,255,0.12)');
  track.setAttribute('stroke-width', _CIRC_STROKE);
  svg.appendChild(track);

  // Progress arc
  const arc = document.createElementNS(ns, 'circle');
  arc.setAttribute('cx', _CIRC_C);
  arc.setAttribute('cy', _CIRC_C);
  arc.setAttribute('r', _CIRC_R);
  arc.setAttribute('fill', 'none');
  arc.setAttribute('stroke', '#c95a2e');
  arc.setAttribute('stroke-width', _CIRC_STROKE);
  arc.setAttribute('stroke-linecap', 'round');
  arc.setAttribute('stroke-dasharray', _CIRC_CIRCUMFERENCE);
  arc.setAttribute('stroke-dashoffset', _CIRC_CIRCUMFERENCE); // 0% initially
  arc.setAttribute('transform', `rotate(-90 ${_CIRC_C} ${_CIRC_C})`);
  arc.dataset.progressArc = '1';
  svg.appendChild(arc);

  // Percentage text
  const text = document.createElementNS(ns, 'text');
  text.setAttribute('x', _CIRC_C);
  text.setAttribute('y', _CIRC_C + 1);
  text.setAttribute('text-anchor', 'middle');
  text.setAttribute('dominant-baseline', 'middle');
  text.setAttribute('fill', '#ddd');
  text.setAttribute('font-size', '10');
  text.setAttribute('font-family', 'system-ui, sans-serif');
  text.dataset.progressPct = '1';
  text.textContent = '0%';
  svg.appendChild(text);

  const label = document.createElement('div');
  label.className = 'gallery-progress-label';
  label.dataset.progressLabel = '1';
  label.textContent = 'Preparing…';

  wrap.appendChild(svg);
  wrap.appendChild(label);
  return wrap;
}

function _updateItemProgress(item, phase, step, total, queuePos) {
  // Find the DOM element for this gallery item
  const root = document.getElementById('gallery');
  const idx = gallery.indexOf(item);
  if (idx < 0 || !root.children[idx]) return;
  const el = root.children[idx];

  const arc = el.querySelector('[data-progress-arc]');
  const pctText = el.querySelector('[data-progress-pct]');
  const label = el.querySelector('[data-progress-label]');
  if (!arc) return;

  if (phase === 'queued') {
    arc.setAttribute('stroke-dashoffset', _CIRC_CIRCUMFERENCE);
    if (pctText) pctText.textContent = '';
    if (label) label.textContent = `Queue #${queuePos}`;
    arc.setAttribute('stroke', '#888');
  } else if (phase === 'start') {
    arc.setAttribute('stroke', '#c95a2e');
    arc.setAttribute('stroke-dashoffset', _CIRC_CIRCUMFERENCE);
    if (pctText) pctText.textContent = '0%';
    if (label) label.textContent = 'Preparing…';
  } else if (phase === 'denoising') {
    const pct = total > 0 ? Math.round((step / total) * 100) : 0;
    const offset = _CIRC_CIRCUMFERENCE * (1 - pct / 100);
    arc.setAttribute('stroke', '#c95a2e');
    arc.setAttribute('stroke-dashoffset', offset);
    if (pctText) pctText.textContent = pct + '%';
    if (label) label.textContent = `Step ${step} / ${total}`;
  }
}

function watchJobProgress(jobId, galleryItem) {
  const src = new EventSource(`/progress?job_id=${encodeURIComponent(jobId)}`);
  src.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.phase === 'queued') {
      _updateItemProgress(galleryItem, 'queued', 0, 0, ev.position);
    } else if (ev.phase === 'start') {
      _updateItemProgress(galleryItem, 'start', 0, 0, 0);
    } else if (ev.phase === 'denoising') {
      _updateItemProgress(galleryItem, 'denoising', ev.step, ev.total, 0);
    } else if (ev.phase === 'done' || ev.phase === 'error') {
      src.close();
    }
  };
  src.onerror = () => src.close();
}

// ── BG inverse-render upload ───────────────────────────────────────

function makeBgThumbProgress() {
  const wrap = document.createElement('div');
  wrap.className = 'gallery-progress';
  wrap.style.position = 'absolute';
  wrap.style.inset = '0';
  wrap.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';
  const size = 36, r = 14, sw = 3, c = r + sw;
  const circ = 2 * Math.PI * r;
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', size); svg.setAttribute('height', size);
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  const track = document.createElementNS(ns, 'circle');
  track.setAttribute('cx', c); track.setAttribute('cy', c); track.setAttribute('r', r);
  track.setAttribute('fill', 'none');
  track.setAttribute('stroke', 'rgba(255,255,255,0.12)');
  track.setAttribute('stroke-width', sw);
  svg.appendChild(track);
  const arc = document.createElementNS(ns, 'circle');
  arc.setAttribute('cx', c); arc.setAttribute('cy', c); arc.setAttribute('r', r);
  arc.setAttribute('fill', 'none');
  arc.setAttribute('stroke', '#c95a2e');
  arc.setAttribute('stroke-width', sw);
  arc.setAttribute('stroke-linecap', 'round');
  arc.setAttribute('stroke-dasharray', circ);
  arc.setAttribute('stroke-dashoffset', circ);
  arc.setAttribute('transform', `rotate(-90 ${c} ${c})`);
  svg.appendChild(arc);
  wrap.appendChild(svg);
  return { wrap, arc, circ };
}

async function uploadBgImage(file, thumbs, addBgThumb, selectBgThumb) {
  const placeholder = document.createElement('div');
  placeholder.className = 'bg-thumb pending';
  placeholder.title = file.name;
  const prog = makeBgThumbProgress();
  placeholder.appendChild(prog.wrap);
  const lab = document.createElement('div');
  lab.className = 'bg-thumb-label';
  lab.textContent = 'Queued…';
  placeholder.appendChild(lab);
  thumbs.appendChild(placeholder);

  const jobId = ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
    (c ^ (Math.random() * 16 >> c / 4)).toString(16));

  const src = new EventSource(`/progress?job_id=${encodeURIComponent(jobId)}`);
  src.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.phase === 'queued') {
      lab.textContent = `Queue #${ev.position}`;
    } else if (ev.phase === 'start') {
      lab.textContent = 'Preparing…';
      prog.arc.setAttribute('stroke-dashoffset', prog.circ);
    } else if (ev.phase === 'inverse_pass') {
      lab.textContent = `${ev.pass_name} (${ev.pass_idx + 1}/${ev.num_passes})`;
    } else if (ev.phase === 'denoising') {
      const pct = ev.total > 0 ? ev.step / ev.total : 0;
      prog.arc.setAttribute('stroke-dashoffset', prog.circ * (1 - pct));
    } else if (ev.phase === 'done' || ev.phase === 'error') {
      src.close();
    }
  };
  src.onerror = () => src.close();

  try {
    const form = new FormData();
    form.append('job_id', jobId);
    form.append('sid', BG_SID);
    form.append('image', file, file.name);
    const resp = await fetch('/upload_bg', { method: 'POST', body: form });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Server error ${resp.status}: ${text}`);
    }
    const info = await resp.json();
    placeholder.remove();
    const t = addBgThumb(info.name);
    if (t) selectBgThumb(t);
  } catch (err) {
    console.error(err);
    src.close();
    lab.textContent = 'Failed';
    placeholder.classList.remove('pending');
    placeholder.style.pointerEvents = 'auto';
    placeholder.addEventListener('click', () => placeholder.remove(), { once: true });
    alert('BG inverse render failed: ' + err.message);
  }
}

async function renderViaServer(panels, mode = 'still') {
  const renderBtn = document.getElementById('render-btn');
  const rotateBtn = document.getElementById('rotate-light-btn');
  renderBtn.disabled = true;
  rotateBtn.disabled = true;

  const jobId = ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
    (c ^ (Math.random() * 16 >> c / 4)).toString(16));
  const item = {
    timestamp: fmtTime(),
    hdr: activeHdr,
    bg: activeBgPreset,
    mode,
    gbuffers: {},
    resultUrl: null,
  };
  gallery.unshift(item);
  refreshGallery();
  watchJobProgress(jobId, item);

  try {
    const formData = new FormData();
    formData.append('job_id', jobId);
    formData.append('hdr', activeHdr);
    formData.append('mode', mode);
    if (activeBgPreset) formData.append('bg_preset', activeBgPreset);
    for (const p of panels) {
      const name = p.bufName;
      const dataUrl = await captureAtTargetSize(p);
      const bgUrl = activeBgPreset
        ? `/inverse/${activeBgPreset}/${name}.png?${BG_SID_PARAM}`
        : null;
      const blob = await compositeCapture(dataUrl, bgUrl);
      item.gbuffers[name] = URL.createObjectURL(blob);
      formData.append(name, blob, `${name}.png`);
    }
    // Metallic: composite with background if available, else black
    const metallicBgUrl = activeBgPreset ? `/inverse/${activeBgPreset}/metallic.png?${BG_SID_PARAM}` : null;
    const metallicBlob = metallicBgUrl
      ? await compositeCapture(makeBlackDataUrl(RENDER_W, RENDER_H), metallicBgUrl)
      : await makeBlackBlob(RENDER_W, RENDER_H);
    item.gbuffers.metallic = URL.createObjectURL(metallicBlob);
    formData.append('metallic', metallicBlob, 'metallic.png');
    refreshGallery();

    const resp = await fetch('/render', { method: 'POST', body: formData });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Server error ${resp.status}: ${text}`);
    }
    const respCtype = resp.headers.get('Content-Type') || '';
    if (respCtype.startsWith('image/') || respCtype.startsWith('video/')) {
      const blob = await resp.blob();
      item.resultUrl = URL.createObjectURL(blob);
      item.resultType = respCtype.startsWith('video/') ? 'video' : 'image';
    } else {
      const info = await resp.json();
      item.savedDir = info.saved_dir;
      item.resultUrl = item.gbuffers.basecolor;
      item.resultType = 'image';
    }
    refreshGallery();
    // If the user has this item's modal open, refresh it
    const modal = document.getElementById('result-modal');
    if (modal.classList.contains('visible') && modal._activeItem === item) {
      openModal(item);
    }
  } catch (err) {
    console.error(err);
    item.error = err.message;
    refreshGallery();
    alert('Render failed: ' + err.message);
  } finally {
    renderBtn.disabled = false;
    rotateBtn.disabled = false;
  }
}

// ── Build & populate scene ──────────────────────────────────────────

async function buildAndPopulate(scene, panels) {
  const actorsData = await buildSceneActors(scene);
  currentActorsData = actorsData;
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

  frameCameras(panels, actorsData, activeFov);
  for (const p of panels) p.renderWindow.render();
}

// ── Depth UI ────────────────────────────────────────────────────────

// Slider value (0..SLIDER_MAX) maps linearly to [depthSlider.min, depthSlider.max].
const SLIDER_MAX = 1000;
const MIN_GAP = 5;

function sliderToDist(v) {
  return depthSlider.min + (v / SLIDER_MAX) * (depthSlider.max - depthSlider.min);
}
function distToSlider(d) {
  const span = depthSlider.max - depthSlider.min;
  if (span <= 0) return 0;
  return Math.round(((d - depthSlider.min) / span) * SLIDER_MAX);
}

function writeSliders(nv, cv, fv) {
  document.getElementById('depth-near').value = nv;
  document.getElementById('depth-center').value = cv;
  document.getElementById('depth-far').value = fv;
}

function updateDepthFill() {
  const nearEl = document.getElementById('depth-near');
  const farEl = document.getElementById('depth-far');
  const centerEl = document.getElementById('depth-center');
  const fill = document.getElementById('depth-fill');
  if (!fill || !nearEl || !farEl) return;
  const nv = parseFloat(nearEl.value);
  const fv = parseFloat(farEl.value);
  const cv = parseFloat(centerEl.value);
  const nPct = (nv / SLIDER_MAX) * 100;
  const fPct = (fv / SLIDER_MAX) * 100;
  fill.style.left = `${nPct}%`;
  fill.style.width = `${Math.max(fPct - nPct, 0)}%`;
  // Center handle stays on top so it remains grabbable; near/far swap by side
  centerEl.style.zIndex = '3';
  const mid = (nv + fv) / 2;
  if (cv < mid) {
    nearEl.style.zIndex = '2'; farEl.style.zIndex = '1';
  } else {
    nearEl.style.zIndex = '1'; farEl.style.zIndex = '2';
  }
}

function updateDepthValueLabels() {
  document.getElementById('depth-near-val').textContent = `near ${depthState.near.toFixed(1)}`;
  document.getElementById('depth-far-val').textContent = `far ${depthState.far.toFixed(1)}`;
}

function depthSliderChanged(panels, source) {
  const nearEl = document.getElementById('depth-near');
  const centerEl = document.getElementById('depth-center');
  const farEl = document.getElementById('depth-far');
  let nv = parseInt(nearEl.value, 10);
  let cv = parseInt(centerEl.value, 10);
  let fv = parseInt(farEl.value, 10);

  if (source === 'center') {
    // Center moves: shift near/far by the delta, clamped to [0, SLIDER_MAX].
    const prevCenter = (nv + fv) / 2;
    let delta = cv - prevCenter;
    const halfL = prevCenter - nv;
    const halfR = fv - prevCenter;
    const minDelta = -nv;
    const maxDelta = SLIDER_MAX - fv;
    delta = Math.max(minDelta, Math.min(maxDelta, delta));
    nv = Math.round(prevCenter + delta - halfL);
    fv = Math.round(prevCenter + delta + halfR);
    cv = Math.round((nv + fv) / 2);
  } else if (source === 'near') {
    // Near moves: mirror around center (far moves the opposite way by the same delta)
    // Clamp so half-width stays within [MIN_GAP/2, min(cv, SLIDER_MAX - cv)]
    let half = cv - nv;
    const maxHalf = Math.min(cv, SLIDER_MAX - cv);
    half = Math.max(MIN_GAP / 2, Math.min(half, maxHalf));
    nv = Math.round(cv - half);
    fv = Math.round(cv + half);
  } else if (source === 'far') {
    let half = fv - cv;
    const maxHalf = Math.min(cv, SLIDER_MAX - cv);
    half = Math.max(MIN_GAP / 2, Math.min(half, maxHalf));
    nv = Math.round(cv - half);
    fv = Math.round(cv + half);
  }
  writeSliders(nv, cv, fv);
  depthState.near = sliderToDist(nv);
  depthState.far = sliderToDist(fv);
  updateDepthValueLabels();
  updateDepthFill();
  const depthPanel = panels[2];
  if (depthPanel) depthPanel.renderWindow.render();
}

// Returns focal-relative z range: 0 = focal point, negative = closer to camera.
function computeFocalRelativeZRange(panel, actorsData) {
  if (!actorsData || !actorsData.length) return null;
  const cam = panel.renderer.getActiveCamera();
  const pos = cam.getPosition();
  const fp = cam.getFocalPoint();
  const dx = fp[0] - pos[0], dy = fp[1] - pos[1], dz = fp[2] - pos[2];
  const focalDist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
  const vx = dx / focalDist, vy = dy / focalDist, vz = dz / focalDist;
  let zmin = Infinity, zmax = -Infinity;
  for (const ad of actorsData) {
    const b = ad.polyData.getBounds();
    for (let i = 0; i < 8; i++) {
      const x = (i & 1) ? b[1] : b[0];
      const y = (i & 2) ? b[3] : b[2];
      const z = (i & 4) ? b[5] : b[4];
      const dist = (x - pos[0]) * vx + (y - pos[1]) * vy + (z - pos[2]) * vz;
      if (dist < zmin) zmin = dist;
      if (dist > zmax) zmax = dist;
    }
  }
  // Translate to focal-relative (focal = 0)
  return { near: zmin - focalDist, far: zmax - focalDist };
}

function autoFitDepth(panels) {
  const depthPanel = panels[2];
  const range = computeFocalRelativeZRange(depthPanel, currentActorsData);
  if (!range) return;
  // Expand fitted range to 6× around its center (2× of previous 3×)
  const center = (range.near + range.far) / 2;
  const halfFit = (range.far - range.near) / 2;
  const near = center - halfFit * 6;
  const far = center + halfFit * 6;
  const delta = far - near;
  // Slider domain: ±3× the (expanded) range
  depthSlider.min = near - delta * 3;
  depthSlider.max = far + delta * 3;
  depthState.near = near;
  depthState.far = far;
  const nv = distToSlider(near);
  const fv = distToSlider(far);
  writeSliders(nv, Math.round((nv + fv) / 2), fv);
  updateDepthValueLabels();
  updateDepthFill();
  depthPanel.renderWindow.render();
}

// ── Main ────────────────────────────────────────────────────────────

function setStatus(msg) {
  document.getElementById('loading').textContent = msg;
}

async function main() {
  setStatus('Initializing renderers...');
  const panels = [
    createPanel(document.getElementById('panel-basecolor'), [0.5, 0.5, 0.5]),
    createPanel(document.getElementById('panel-normal'), [0.5, 0.5, 0.5]),
    createPanel(document.getElementById('panel-depth'), [1, 1, 1]),
    createPanel(document.getElementById('panel-roughness'), [0.5, 0.5, 0.5]),
  ];
  panels[0].bufName = 'basecolor';
  panels[1].bufName = 'normal';
  panels[2].bufName = 'depth';
  panels[3].bufName = 'roughness';

  setStatus('Loading WASM module + sample scene...');
  const scene = await loadZprjFromUrl('/sample.zprj');

  setStatus('Building geometry...');
  await buildAndPopulate(scene, panels);

  syncCameras(panels);
  autoFitDepth(panels);

  // Fixed 2x2 grid for the 4 G-buffer panels
  const mainEl = document.getElementById('main');
  mainEl.style.setProperty('--cols', 2);
  mainEl.style.setProperty('--rows', 2);

  // Depth slider wiring
  document.getElementById('depth-near').addEventListener('input', () => depthSliderChanged(panels, 'near'));
  document.getElementById('depth-center').addEventListener('input', () => depthSliderChanged(panels, 'center'));
  document.getElementById('depth-far').addEventListener('input', () => depthSliderChanged(panels, 'far'));
  document.getElementById('depth-fit').addEventListener('click', () => autoFitDepth(panels));

  // Background preset thumbnails — loaded dynamically from server
  const thumbs = document.getElementById('bg-thumbs');
  const clearThumb = document.createElement('div');
  clearThumb.className = 'bg-thumb clear active';
  clearThumb.textContent = 'None';
  clearThumb.dataset.preset = '';
  thumbs.appendChild(clearThumb);

  function addBgThumb(preset) {
    if (BG_PRESETS.includes(preset)) return null;
    BG_PRESETS.push(preset);
    const t = document.createElement('div');
    t.className = 'bg-thumb';
    t.style.backgroundImage = `url(/inverse/${preset}/rgb_input.png?${BG_SID_PARAM}&t=${Date.now()})`;
    t.dataset.preset = preset;
    t.title = preset;
    thumbs.appendChild(t);
    return t;
  }

  function selectBgThumb(el) {
    for (const sib of thumbs.children) sib.classList.remove('active');
    el.classList.add('active');
    setBackgroundPreset(panels, el.dataset.preset || null);
  }

  try {
    const bgResp = await fetch(`/bg_presets?${BG_SID_PARAM}`);
    const presets = await bgResp.json();
    for (const preset of presets) addBgThumb(preset);
  } catch (e) {
    console.warn('Failed to load bg presets', e);
  }
  thumbs.addEventListener('click', (e) => {
    const t = e.target.closest('.bg-thumb');
    if (!t || t.classList.contains('pending')) return;
    selectBgThumb(t);
  });

  // ── BG image upload → server-side inverse render ──────────────────
  document.getElementById('bg-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    await uploadBgImage(file, thumbs, addBgThumb, selectBgThumb);
  });

  // HDR gallery (sidebar)
  const hdrThumbs = document.getElementById('hdr-thumbs');
  for (const name of HDR_PRESETS) {
    const div = document.createElement('div');
    div.className = 'hdr-thumb';
    if (name === activeHdr) div.classList.add('active');
    div.dataset.hdr = name;
    div.innerHTML = `<div class="name">${name}</div>`;
    hdrThumbs.appendChild(div);
    // Async decode + render preview
    decodeHdrToCanvas(`/hdri/${name}.hdr`)
      .then((canvas) => {
        // Replace the .name child's preceding spot
        div.insertBefore(canvas, div.firstChild);
      })
      .catch((err) => {
        console.warn('HDR thumb failed', name, err);
        div.querySelector('.name').textContent = `${name} (failed)`;
      });
  }
  hdrThumbs.addEventListener('click', (e) => {
    const t = e.target.closest('.hdr-thumb');
    if (!t) return;
    for (const sib of hdrThumbs.children) sib.classList.remove('active');
    t.classList.add('active');
    activeHdr = t.dataset.hdr;
  });

  // HDR upload (POST to server, also add to client gallery)
  document.getElementById('hdr-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const baseName = file.name.replace(/\.hdr$/i, '');
    try {
      const form = new FormData();
      form.append('hdr', file, file.name);
      const resp = await fetch('/upload_hdr', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
      const info = await resp.json();
      const hdrName = info.name || baseName;
      // Add to client gallery
      const div = document.createElement('div');
      div.className = 'hdr-thumb';
      div.dataset.hdr = hdrName;
      div.innerHTML = `<div class="name">${hdrName}</div>`;
      hdrThumbs.appendChild(div);
      const buf = await file.arrayBuffer();
      const canvas = hdrBufferToCanvas(buf);
      div.insertBefore(canvas, div.firstChild);
      // Auto-select
      for (const sib of hdrThumbs.children) sib.classList.remove('active');
      div.classList.add('active');
      activeHdr = hdrName;
    } catch (err) {
      console.error(err);
      alert('HDR upload failed: ' + err.message);
    }
    e.target.value = '';
  });

  // FOV slider
  const fovSlider = document.getElementById('fov-slider');
  const fovVal = document.getElementById('fov-val');
  fovSlider.addEventListener('input', () => {
    activeFov = parseInt(fovSlider.value, 10);
    fovVal.textContent = activeFov;
    if (currentActorsData) {
      frameCameras(panels, currentActorsData, activeFov);
      for (const p of panels) p.renderWindow.render();
    }
  });

  // Render buttons
  document.getElementById('render-btn').addEventListener('click', () => {
    renderViaServer(panels, 'still');
  });
  document.getElementById('rotate-light-btn').addEventListener('click', () => {
    renderViaServer(panels, 'rotate_light');
  });

  // Modal close + hover preview
  document.getElementById('result-modal-close').addEventListener('click', hideModal);
  document.getElementById('result-modal').addEventListener('click', (e) => {
    if (e.target.id === 'result-modal') hideModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideModal();
  });
  setupModalHover();
  refreshGallery();

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
      autoFitDepth(panels);
      document.getElementById('zprj-name').textContent = file.name;
      document.getElementById('zprj-title').textContent = file.name;
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
