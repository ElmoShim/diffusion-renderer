import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkTexture from '@kitware/vtk.js/Rendering/Core/Texture';
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

// ── Image DPI parsing ─────────────────────────────────────────────────────

function _u16be(b, i) { return ((b[i] & 0xFF) << 8) | (b[i + 1] & 0xFF); }
function _u32be(b, i) {
  return (((b[i] & 0xFF) << 24) | ((b[i+1] & 0xFF) << 16) |
          ((b[i+2] & 0xFF) << 8)  |  (b[i+3] & 0xFF)) >>> 0;
}

// Parse JPEG or PNG raw bytes → { widthPx, heightPx, dpiX, dpiY }.
// Defaults to 72 DPI when metadata is absent.
function parseImageMeta(raw) {
  const b = new Uint8Array(raw.length);
  for (let j = 0; j < raw.length; j++) b[j] = raw[j];

  // ── JPEG ──────────────────────────────────────────────────────────────
  if (b[0] === 0xFF && b[1] === 0xD8) {
    let w = 0, h = 0, dpiX = 72, dpiY = 72;
    let i = 2;
    while (i + 3 < b.length) {
      if (b[i] !== 0xFF) break;
      while (i < b.length && b[i] === 0xFF) i++; // skip fill bytes
      const m = b[i++];
      if (m === 0xD9 || m === 0xDA) break;
      const segLen = _u16be(b, i);
      const d = i + 2; // segment data start

      if (m === 0xE0 && b[d] === 0x4A && b[d + 1] === 0x46) {
        // JFIF APP0: d+7=units, d+8..9=Xdensity, d+10..11=Ydensity
        const units = b[d + 7];
        const xd = _u16be(b, d + 8), yd = _u16be(b, d + 10);
        if (xd > 0 && yd > 0) {
          dpiX = units === 1 ? xd : units === 2 ? xd * 2.54 : dpiX;
          dpiY = units === 1 ? yd : units === 2 ? yd * 2.54 : dpiY;
        }
      } else if (m === 0xE1 && b[d] === 0x45 && b[d + 1] === 0x78) {
        // Exif APP1 — minimal TIFF IFD reader for XResolution/YResolution
        const t = d + 6; // TIFF header base (after "Exif\0\0")
        const le = b[t] === 0x49; // 'I' = Intel/little-endian
        const rd16 = le
          ? (o) => (b[t + o] & 0xFF) | ((b[t + o + 1] & 0xFF) << 8)
          : (o) => _u16be(b, t + o);
        const rd32 = le
          ? (o) => (((b[t+o] & 0xFF) | ((b[t+o+1] & 0xFF) << 8) |
                      ((b[t+o+2] & 0xFF) << 16) | ((b[t+o+3] & 0xFF) << 24)) >>> 0)
          : (o) => _u32be(b, t + o);
        const rational = (o) => { const n = rd32(o), den = rd32(o + 4); return den ? n / den : 0; };
        try {
          const ifdOff = rd32(4);
          const cnt = rd16(ifdOff);
          let xr = 0, yr = 0, unit = 2;
          for (let e = 0; e < cnt; e++) {
            const ep = ifdOff + 2 + e * 12;
            const tag = rd16(ep);
            if (tag === 0x011A) xr = rational(rd32(ep + 8));
            else if (tag === 0x011B) yr = rational(rd32(ep + 8));
            else if (tag === 0x0128) unit = rd16(ep + 8);
          }
          if (xr > 0 && yr > 0 && dpiX === 72 && dpiY === 72) {
            dpiX = unit === 3 ? xr * 2.54 : xr;
            dpiY = unit === 3 ? yr * 2.54 : yr;
          }
        } catch { /* ignore malformed Exif */ }
      } else if (m >= 0xC0 && m <= 0xCF && m !== 0xC4 && m !== 0xC8 && m !== 0xCC) {
        // SOF marker → image dimensions
        h = _u16be(b, d + 1);
        w = _u16be(b, d + 3);
      }
      i += segLen;
    }
    return { widthPx: w, heightPx: h, dpiX, dpiY };
  }

  // ── PNG ───────────────────────────────────────────────────────────────
  if (b[0] === 0x89 && b[1] === 0x50) {
    const w = _u32be(b, 16), h = _u32be(b, 20);
    let dpiX = 72, dpiY = 72;
    let ci = 8;
    while (ci + 12 < b.length) {
      const cLen = _u32be(b, ci);
      const cType = String.fromCharCode(b[ci+4], b[ci+5], b[ci+6], b[ci+7]);
      if (cType === 'pHYs' && cLen >= 9) {
        const ppuX = _u32be(b, ci + 8), ppuY = _u32be(b, ci + 12);
        if (b[ci + 16] === 1 && ppuX > 0 && ppuY > 0) { // unit = metre
          dpiX = ppuX / 39.3701;
          dpiY = ppuY / 39.3701;
        }
      }
      if (cType === 'IDAT' || cType === 'IEND') break;
      ci += 12 + cLen;
    }
    return { widthPx: w, heightPx: h, dpiX, dpiY };
  }

  return { widthPx: 0, heightPx: 0, dpiX: 72, dpiY: 72 };
}

// Return { tw, th } tile size in mm derived from image DPI (matches CLO's rule).
// Falls back to mat.tileWidth / mat.tileHeight when the image is unavailable.
function getTextureTileMm(scene, texPath, mat, cache) {
  const fallback = {
    tw: mat && mat.tileWidth > 0 ? mat.tileWidth : 1,
    th: mat && mat.tileHeight > 0 ? mat.tileHeight : 1,
  };
  if (!texPath) return fallback;
  if (cache && cache.has(texPath)) return cache.get(texPath);

  try {
    if (!scene.hasFile(texPath)) {
      if (cache) cache.set(texPath, fallback);
      return fallback;
    }
    const raw = scene.readFile(texPath);
    if (!raw || raw.length === 0) {
      if (cache) cache.set(texPath, fallback);
      return fallback;
    }
    const { widthPx, heightPx, dpiX, dpiY } = parseImageMeta(raw);
    if (widthPx > 0 && heightPx > 0 && dpiX > 0 && dpiY > 0) {
      const result = { tw: (widthPx / dpiX) * 25.4, th: (heightPx / dpiY) * 25.4 };
      if (cache) cache.set(texPath, result);
      return result;
    }
  } catch { /* fall through */ }

  if (cache) cache.set(texPath, fallback);
  return fallback;
}

// ─────────────────────────────────────────────────────────────────────────────

export function setTexCoords(polyData, uvs, vertexCount, tileWidth, tileHeight, texTransform) {
  const tw = tileWidth > 0 ? tileWidth : 1;
  const th = tileHeight > 0 ? tileHeight : 1;
  const scaled = new Float32Array(vertexCount * 2);
  for (let i = 0; i < vertexCount; i++) {
    scaled[i * 2] = uvs[i * 2] / tw;
    scaled[i * 2 + 1] = uvs[i * 2 + 1] / th;
  }
  // scaleU/V encodes the user's texture zoom; 0.001 == 100% (CLO default).
  const SCALE_ONE = 0.001;
  const su = texTransform?.scaleU ?? SCALE_ONE;
  const sv = texTransform?.scaleV ?? SCALE_ONE;
  if (Math.abs(su - SCALE_ONE) > 1e-7 || Math.abs(sv - SCALE_ONE) > 1e-7) {
    const fu = su / SCALE_ONE, fv = sv / SCALE_ONE;
    for (let i = 0; i < vertexCount; i++) {
      scaled[i * 2] *= fu;
      scaled[i * 2 + 1] *= fv;
    }
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
  const ou = texTransform?.offsetU ?? 0;
  const ov = texTransform?.offsetV ?? 0;
  if (ou || ov) {
    for (let i = 0; i < vertexCount; i++) {
      scaled[i * 2] += ou;
      scaled[i * 2 + 1] += ov;
    }
  }
  const tcoords = vtkDataArray.newInstance({
    numberOfComponents: 2,
    values: scaled,
    name: 'TCoords',
  });
  polyData.getPointData().setTCoords(tcoords);
}

export async function loadTextureFromScene(scene, texturePath) {
  try {
    if (!texturePath || !scene.hasFile(texturePath)) return null;
    const raw = scene.readFile(texturePath);
    if (!raw || raw.length === 0) return null;

    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw[i];

    const ext = texturePath.split('.').pop()?.toLowerCase() || '';
    const mime = ext === 'png' ? 'image/png' : 'image/jpeg';
    const blob = new Blob([bytes], { type: mime });
    const url = URL.createObjectURL(blob);

    const img = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error(`Failed to decode texture: ${texturePath}`));
      el.src = url;
    });

    // resizable:true forces the mutable texImage2D path instead of
    // texStorage2D(levels=1); only then does generateMipmap actually run
    // (WebGL2 makes texStorage textures immutable at 1 mip level → no mipmaps).
    const texture = vtkTexture.newInstance({ resizable: true });
    texture.setImage(img);
    texture.setRepeat(true);
    texture.setInterpolate(true); // interpolate=true triggers mipmap generation
    return texture;
  } catch {
    return null;
  }
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
  const tileMmCache = new Map();
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

    if (mat && pattern.uvVertexCount === pattern.vertexCount) {
      const texPath = mat.diffuseTexturePath || null;
      if (texPath) {
        if (!texCache.has(texPath)) {
          texCache.set(texPath, await loadTextureFromScene(scene, texPath));
        }
        diffuseTex = texCache.get(texPath);
      }

      const roughPath = mat.roughnessTexturePath || null;
      if (roughPath) {
        const rKey = 'rough:' + roughPath;
        if (!texCache.has(rKey)) {
          texCache.set(rKey, await loadTextureFromScene(scene, roughPath));
        }
        roughnessTex = texCache.get(rKey);
      }

      if (diffuseTex || roughnessTex) {
        const uvs = pattern.getUVs();
        const { tw, th } = getTextureTileMm(scene, mat.diffuseTexturePath, mat, tileMmCache);
        setTexCoords(polyData, uvs, pattern.vertexCount, tw, th, mat.diffuseTextureTransform);
      }
    }

    const baseColor = mat ? normalizeColor(mat.getBaseColor()) : [0.8, 0.8, 0.8];
    const isPBR = mat?.useMetalnessRoughnessPBR ?? false;
    const roughness = isPBR ? (mat.roughness ?? 0.5) : 0.5;
    const metallic = isPBR ? (mat.metalness ?? 0.0) : 0.0;
    actors.push({
      polyData, diffuseColor: baseColor, roughness, metallic,
      diffuseTex, roughnessTex, type: 'garment',
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
