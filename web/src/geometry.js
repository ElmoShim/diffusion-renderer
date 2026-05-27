import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkTexture from '@kitware/vtk.js/Rendering/Core/Texture';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkPolyDataNormals from '@kitware/vtk.js/Filters/Core/PolyDataNormals';

function isIdentity(m) {
  if (!m || m.length !== 16) return true;
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      if (Math.abs(m[i * 4 + j] - (i === j ? 1 : 0)) > 1e-6) return false;
  return true;
}

function applyMatrix(polyData, m) {
  const pts = polyData.getPoints().getData();
  const nv = pts.length / 3;
  const out = new Float32Array(pts.length);
  for (let i = 0; i < nv; i++) {
    const x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
    out[i * 3]     = m[0] * x + m[4] * y + m[8]  * z + m[12];
    out[i * 3 + 1] = m[1] * x + m[5] * y + m[9]  * z + m[13];
    out[i * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  }
  polyData.getPoints().setData(out, 3);

  const nArr = polyData.getPointData().getNormals();
  if (nArr) {
    const nd = nArr.getData();
    const nout = new Float32Array(nd.length);
    for (let i = 0; i < nv; i++) {
      const nx = nd[i * 3], ny = nd[i * 3 + 1], nz = nd[i * 3 + 2];
      let rx = m[0] * nx + m[4] * ny + m[8]  * nz;
      let ry = m[1] * nx + m[5] * ny + m[9]  * nz;
      let rz = m[2] * nx + m[6] * ny + m[10] * nz;
      const len = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
      nout[i * 3] = rx / len;
      nout[i * 3 + 1] = ry / len;
      nout[i * 3 + 2] = rz / len;
    }
    nArr.setData(nout);
  }
}

export function makePolyData(positions, indices, vertexCount, triangleCount, normals) {
  const polyData = vtkPolyData.newInstance();
  polyData.getPoints().setData(new Float32Array(positions), 3);

  const cells = new Uint32Array(triangleCount * 4);
  for (let t = 0; t < triangleCount; t++) {
    cells[t * 4] = 3;
    cells[t * 4 + 1] = indices[t * 3];
    cells[t * 4 + 2] = indices[t * 3 + 1];
    cells[t * 4 + 3] = indices[t * 3 + 2];
  }
  polyData.getPolys().setData(cells);

  if (normals && normals.length === vertexCount * 3) {
    const normalArray = vtkDataArray.newInstance({
      numberOfComponents: 3,
      values: new Float32Array(normals),
      name: 'Normals',
    });
    polyData.getPointData().setNormals(normalArray);
    return polyData;
  }

  const normalsFilter = vtkPolyDataNormals.newInstance();
  normalsFilter.setInputData(polyData);
  normalsFilter.update();
  return normalsFilter.getOutputData();
}

// ── Texture tile size ─────────────────────────────────────────────────────

// Displayed texture tile size { tw, th } in mm — the "width"/"height" from
// CLO's Fabric panel transformation, exposed per-texture as physicalWidth/
// physicalHeight (zprj-loader >= 1.2). Reproduces CLO's on-garment scale at any
// zoom. mat.tileWidth/tileHeight is the bolt width (~1117.6 mm), NOT this value.
function textureTileMm(mat) {
  if (mat) {
    for (const xf of [mat.diffuseTextureTransform, mat.normalTextureTransform]) {
      if (xf && xf.physicalWidth > 0 && xf.physicalHeight > 0) {
        return { tw: xf.physicalWidth, th: xf.physicalHeight };
      }
    }
  }
  return {
    tw: mat && mat.tileWidth > 0 ? mat.tileWidth : 1,
    th: mat && mat.tileHeight > 0 ? mat.tileHeight : 1,
  };
}

// ─────────────────────────────────────────────────────────────────────────────

export function setTexCoords(polyData, uvs, vertexCount, tileWidth, tileHeight, texTransform) {
  // Pattern UVs are in mm; dividing by the displayed tile size in mm yields
  // texture repeats (one tile spans tileWidth mm of fabric), matching CLO.
  const tw = tileWidth > 0 ? tileWidth : 1;
  const th = tileHeight > 0 ? tileHeight : 1;
  const scaled = new Float32Array(vertexCount * 2);
  for (let i = 0; i < vertexCount; i++) {
    scaled[i * 2] = uvs[i * 2] / tw;
    scaled[i * 2 + 1] = uvs[i * 2 + 1] / th;
  }
  const angle = texTransform?.rotation ?? 0;
  if (angle) {
    const r = (angle * Math.PI) / 180;
    const c = Math.cos(r), s = Math.sin(r);
    for (let i = 0; i < vertexCount; i++) {
      const u = scaled[i * 2], v = scaled[i * 2 + 1];
      scaled[i * 2] = u * c - v * s;
      scaled[i * 2 + 1] = u * s + v * c;
    }
  }
  // Texture position offset (mm) → tile units (consistent with the scale above).
  const ou = texTransform?.offsetU ?? 0;
  const ov = texTransform?.offsetV ?? 0;
  if (ou || ov) {
    for (let i = 0; i < vertexCount; i++) {
      scaled[i * 2] += ou / tw;
      scaled[i * 2 + 1] += ov / th;
    }
  }
  const tcoords = vtkDataArray.newInstance({
    numberOfComponents: 2,
    values: scaled,
    name: 'TCoords',
  });
  polyData.getPointData().setTCoords(tcoords);
}

async function loadImageFromScene(scene, texturePath) {
  if (!texturePath || !scene.hasFile(texturePath)) return null;
  const raw = scene.readFile(texturePath);
  if (!raw || raw.length === 0) return null;
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw[i];
  const ext = texturePath.split('.').pop()?.toLowerCase() || '';
  const mime = ext === 'png' ? 'image/png' : 'image/jpeg';
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  return new Promise((resolve) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => resolve(null);
    el.src = url;
  });
}

function makeTexture(imageOrCanvas) {
  // resizable:true forces the mutable texImage2D path instead of
  // texStorage2D(levels=1); only then does generateMipmap actually run
  // (WebGL2 makes texStorage textures immutable at 1 mip level → no mipmaps).
  const texture = vtkTexture.newInstance({ resizable: true });
  texture.setImage(imageOrCanvas);
  texture.setRepeat(true);
  texture.setInterpolate(true); // interpolate=true triggers mipmap generation
  return texture;
}

export async function loadTextureFromScene(scene, texturePath) {
  const img = await loadImageFromScene(scene, texturePath);
  return img ? makeTexture(img) : null;
}

// Build an RGBA vtkTexture whose alpha is the material's opacity map, so VTK.js
// (which does `opacity *= texture1.a` for a 4-channel first texture) renders
// knit gaps / lace / mesh translucently. RGB comes from `colorPath` (e.g. the
// diffuse map) or solid white when null, so a uniform actor color still shows
// through. Returns null when neither a color map nor an opacity map exists.
//   opacity_channel: 0 = luminance of the opacity image, 1 = its alpha channel.
export async function loadOpacityComposite(scene, colorPath, mat) {
  const opPath = mat?.opacityTexturePath || null;
  const colorImg = colorPath ? await loadImageFromScene(scene, colorPath) : null;
  const opImg = opPath ? await loadImageFromScene(scene, opPath) : null;
  if (!colorImg && !opImg) return null;

  const src = colorImg || opImg;
  const w = src.naturalWidth, h = src.naturalHeight;
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  if (colorImg) {
    ctx.drawImage(colorImg, 0, 0, w, h);
  } else {
    ctx.fillStyle = '#ffffff';  // white → preserves a uniform actor color
    ctx.fillRect(0, 0, w, h);
  }
  const id = ctx.getImageData(0, 0, w, h);

  if (opImg) {
    const oc = document.createElement('canvas');
    oc.width = w; oc.height = h;
    const octx = oc.getContext('2d');
    octx.drawImage(opImg, 0, 0, w, h);
    const od = octx.getImageData(0, 0, w, h).data;
    const useAlpha = mat?.opacityChannel === 1;
    for (let i = 0; i < w * h; i++) {
      id.data[i * 4 + 3] = useAlpha
        ? od[i * 4 + 3]
        : (od[i * 4] * 0.299 + od[i * 4 + 1] * 0.587 + od[i * 4 + 2] * 0.114);
    }
  }
  // Upload the RGBA pixels directly via vtkImageData. setImage(canvas) drops the
  // alpha channel on GPU upload (texture renders opaque); setInputData preserves
  // it. Flip Y to match VTK's bottom-up texture origin.
  const flipped = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const src = (h - 1 - y) * w * 4;
    flipped.set(id.data.subarray(src, src + w * 4), y * w * 4);
  }
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(w, h, 1);
  imageData.getPointData().setScalars(vtkDataArray.newInstance({
    numberOfComponents: 4, values: flipped, name: 'rgba',
  }));
  const texture = vtkTexture.newInstance({ resizable: true });
  texture.setInputData(imageData);
  texture.setRepeat(true);
  texture.setInterpolate(true);
  return texture;
}

// Material resolution (matching Python utils_render_vtk.py logic)

function getPatternMaterial(scene, patternIndex) {
  try {
    const nMats = scene.get_fabricMaterials_size();
    const nCw = scene.getColorwaysSize();
    const cwIdx = scene.activeColorwayIndex;
    let mi = scene.get_garmentPatterns(patternIndex).materialIndex;

    if (nCw > 0 && cwIdx >= 0 && cwIdx < nCw) {
      const cw = scene.getColorway(cwIdx);
      if (cw.getPatternFabricIndicesSize() > 0 && patternIndex < cw.getPatternFabricIndicesSize()) {
        mi = cw.getPatternFabricIndex(patternIndex);
      }
    }

    if (mi >= 0 && mi < nMats) {
      return scene.get_fabricMaterials(mi);
    }
  } catch { /* fallback */ }
  return null;
}

function normalizeColor(c) {
  if (!c || c.length < 3) return [0.8, 0.8, 0.8];
  const r = c[0] > 1 ? c[0] / 255 : c[0];
  const g = c[1] > 1 ? c[1] / 255 : c[1];
  const b = c[2] > 1 ? c[2] / 255 : c[2];
  return [r, g, b];
}

// Build mesh data from scene — returns array of { polyData, diffuseColor, roughness, metallic, type, ... }

export async function buildSceneActors(scene) {
  const actors = [];
  const texCache = new Map();
  const cwIdx = scene.activeColorwayIndex;

  // Garment patterns
  const nPatterns = scene.get_garmentPatterns_size();
  for (let i = 0; i < nPatterns; i++) {
    const pattern = scene.get_garmentPatterns(i);
    if (pattern.vertexCount === 0 || pattern.triangleCount === 0) continue;

    const positions = pattern.getPositions();
    const indices = pattern.getIndices();
    const normals = pattern.getNormals();
    const polyData = makePolyData(positions, indices, pattern.vertexCount, pattern.triangleCount, normals);

    const mat = getPatternMaterial(scene, i);
    let diffuseTex = null;
    let roughnessTex = null;
    let opacityTex = null;
    const hasOpacity = !!(mat && mat.opacityTexturePath);

    if (mat && pattern.uvVertexCount === pattern.vertexCount) {
      const diffusePath = mat.diffuseTexturePath || null;
      const roughPath = mat.roughnessTexturePath || null;

      if (hasOpacity) {
        // Merge the opacity map into the alpha of each color texture so VTK.js
        // renders knit gaps translucently (opacity *= texture1.a). The white-RGB
        // composite is reused by the normal/depth shaders via `opacity`.
        const dKey = `diff+op:${diffusePath}`;
        if (!texCache.has(dKey)) texCache.set(dKey, await loadOpacityComposite(scene, diffusePath, mat));
        diffuseTex = texCache.get(dKey);
        const rKey = `rough+op:${roughPath}`;
        if (!texCache.has(rKey)) texCache.set(rKey, await loadOpacityComposite(scene, roughPath, mat));
        roughnessTex = texCache.get(rKey);
        const oKey = 'op:white';
        if (!texCache.has(oKey)) texCache.set(oKey, await loadOpacityComposite(scene, null, mat));
        opacityTex = texCache.get(oKey);
      } else {
        if (diffusePath) {
          if (!texCache.has(diffusePath)) texCache.set(diffusePath, await loadTextureFromScene(scene, diffusePath));
          diffuseTex = texCache.get(diffusePath);
        }
        if (roughPath) {
          const rKey = 'rough:' + roughPath;
          if (!texCache.has(rKey)) texCache.set(rKey, await loadTextureFromScene(scene, roughPath));
          roughnessTex = texCache.get(rKey);
        }
      }

      if (diffuseTex || roughnessTex || opacityTex) {
        const uvs = pattern.getUVs();
        const { tw, th } = textureTileMm(mat);
        setTexCoords(polyData, uvs, pattern.vertexCount, tw, th, mat.diffuseTextureTransform);
      }
    }

    const baseColor = mat ? normalizeColor(mat.getBaseColor()) : [0.8, 0.8, 0.8];
    const isPBR = mat?.useMetalnessRoughnessPBR ?? false;
    const roughness = isPBR ? (mat.roughness ?? 0.5) : 0.5;
    const metallic = isPBR ? (mat.metalness ?? 0.0) : 0.0;
    actors.push({
      polyData, diffuseColor: baseColor, roughness, metallic,
      diffuseTex, roughnessTex, opacityTex, hasOpacity, type: 'garment',
    });
  }

  // Avatar meshes
  const nAvatars = scene.get_avatarMeshes_size();
  for (let i = 0; i < nAvatars; i++) {
    const mesh = scene.get_avatarMeshes(i);
    if (mesh.vertexCount === 0 || mesh.triangleCount === 0) continue;

    let mat = null;
    let diffuseColor = [0.85, 0.75, 0.65];
    if (mesh.hasMaterial) {
      try { mat = mesh.getMaterial(); } catch { /* skip */ }
    }
    if (mat) {
      try {
        const dc = mat.getDiffuseColor();
        if (dc && dc.length >= 4 && dc[3] < 0.001) continue;
        if (dc && dc.length >= 3) diffuseColor = normalizeColor(dc);
      } catch { /* fallback */ }
    }

    const positions = mesh.getPositions();
    const indices = mesh.getIndices();
    const normals = mesh.getNormals();
    const polyData = makePolyData(positions, indices, mesh.vertexCount, mesh.triangleCount, normals);

    try {
      const wm = mesh.getWorldMatrix();
      if (!isIdentity(wm)) applyMatrix(polyData, wm);
    } catch { /* no transform */ }

    let diffuseTex = null;
    let roughnessTex = null;
    const hasUV = mat && mesh.uvVertexCount === mesh.vertexCount;

    if (hasUV) {
      if (mat.diffuseTexturePath) {
        const tp = mat.diffuseTexturePath;
        if (!texCache.has(tp)) texCache.set(tp, await loadTextureFromScene(scene, tp));
        diffuseTex = texCache.get(tp);
      }
      if (mat.roughnessTexturePath) {
        const rp = 'rough:' + mat.roughnessTexturePath;
        if (!texCache.has(rp)) {
          texCache.set(rp, await loadTextureFromScene(scene, mat.roughnessTexturePath));
        }
        roughnessTex = texCache.get(rp);
      }
      if (diffuseTex || roughnessTex) {
        const freshMesh = scene.get_avatarMeshes(i);
        const uvs = freshMesh.getUVs();
        const uvCopy = new Float32Array(uvs.length);
        for (let j = 0; j < uvs.length; j++) uvCopy[j] = uvs[j];
        const tcoords = vtkDataArray.newInstance({
          numberOfComponents: 2, values: uvCopy, name: 'TCoords',
        });
        polyData.getPointData().setTCoords(tcoords);
      }
    }

    actors.push({
      polyData, diffuseColor, roughness: 0.5, metallic: 0.0,
      diffuseTex, roughnessTex, type: 'avatar',
    });
  }

  // Trim objects
  const nTrims = scene.get_trimObjects_size();
  for (let i = 0; i < nTrims; i++) {
    const trim = scene.get_trimObjects(i);
    if (trim.meshVertexCount === 0 || trim.meshTriangleCount === 0) continue;

    const positions = trim.getMeshPositions();
    const indices = trim.getMeshIndices();
    const normals = trim.getMeshNormals();
    const polyData = makePolyData(positions, indices, trim.meshVertexCount, trim.meshTriangleCount, normals);

    try {
      const tm = trim.getTransformMatrix();
      if (!isIdentity(tm)) applyMatrix(polyData, tm);
    } catch { /* no transform */ }

    actors.push({
      polyData, diffuseColor: [0.6, 0.6, 0.7], roughness: 0.5, metallic: 0.0,
      diffuseTex: null, roughnessTex: null, type: 'trim',
    });
  }

  return actors;
}

export function buildFloorDisc(actorsData, fovDeg = 15) {
  if (!actorsData.length) return null;

  let xmin = Infinity, xmax = -Infinity;
  let ymin = Infinity;
  let zmin = Infinity, zmax = -Infinity;
  for (const ad of actorsData) {
    const bounds = ad.polyData.getBounds();
    xmin = Math.min(xmin, bounds[0]); xmax = Math.max(xmax, bounds[1]);
    ymin = Math.min(ymin, bounds[2]);
    zmin = Math.min(zmin, bounds[4]); zmax = Math.max(zmax, bounds[5]);
  }

  const cx = (xmin + xmax) / 2;
  const cz = (zmin + zmax) / 2;
  const spanX = xmax - xmin;
  const spanZ = zmax - zmin;
  const halfFov = (fovDeg * Math.PI / 180) / 2;
  const maxSpan = Math.max(spanX, spanZ);
  const radius = (maxSpan / 2 / Math.tan(halfFov)) * 1.1;
  const nSeg = 64;

  const verts = new Float32Array((nSeg + 1) * 3);
  verts[0] = cx; verts[1] = ymin; verts[2] = cz;
  for (let i = 0; i < nSeg; i++) {
    const a = (i / nSeg) * Math.PI * 2;
    verts[(i + 1) * 3] = cx + radius * Math.cos(a);
    verts[(i + 1) * 3 + 1] = ymin;
    verts[(i + 1) * 3 + 2] = cz + radius * Math.sin(a);
  }
  const indices = new Uint32Array(nSeg * 3);
  let fi = 0;
  for (let i = 0; i < nSeg; i++) {
    const j = (i + 1) % nSeg;
    indices[fi++] = 0; indices[fi++] = j + 1; indices[fi++] = i + 1;
  }
  return makePolyData(verts, indices, nSeg + 1, nSeg, null);
}

export function computeForegroundBounds(actorsData) {
  let minP = [Infinity, Infinity, Infinity];
  let maxP = [-Infinity, -Infinity, -Infinity];
  for (const ad of actorsData) {
    const bounds = ad.polyData.getBounds();
    minP[0] = Math.min(minP[0], bounds[0]);
    minP[1] = Math.min(minP[1], bounds[2]);
    minP[2] = Math.min(minP[2], bounds[4]);
    maxP[0] = Math.max(maxP[0], bounds[1]);
    maxP[1] = Math.max(maxP[1], bounds[3]);
    maxP[2] = Math.max(maxP[2], bounds[5]);
  }
  return { minP, maxP };
}
