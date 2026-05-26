// Minimal RGBE (.hdr / Radiance) decoder.
// Returns { width, height, data: Float32Array(width*height*3) } in linear HDR.

function readLine(view, offset) {
  let line = '';
  while (offset < view.byteLength) {
    const c = view.getUint8(offset++);
    if (c === 0x0a) return { line, offset };
    line += String.fromCharCode(c);
  }
  return { line, offset };
}

function rgbeToFloat(r, g, b, e, out, idx) {
  if (e === 0) {
    out[idx] = 0; out[idx + 1] = 0; out[idx + 2] = 0;
    return;
  }
  const f = Math.pow(2, e - 128) / 256;
  out[idx]     = r * f;
  out[idx + 1] = g * f;
  out[idx + 2] = b * f;
}

export function decodeRGBE(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const bytes = new Uint8Array(arrayBuffer);

  // Parse header (ASCII until empty line)
  let offset = 0;
  let header = '';
  // First line should be "#?RADIANCE" or "#?RGBE"
  const first = readLine(view, offset);
  if (!first.line.startsWith('#?')) throw new Error('Not a Radiance HDR file');
  offset = first.offset;

  // Read remaining header until blank line
  while (offset < view.byteLength) {
    const { line, offset: next } = readLine(view, offset);
    offset = next;
    if (line === '') break;
    header += line + '\n';
  }

  // Resolution line, e.g. "-Y 512 +X 1024"
  const { line: resLine, offset: dataStart } = readLine(view, offset);
  offset = dataStart;
  const m = resLine.match(/([+-][XY])\s+(\d+)\s+([+-][XY])\s+(\d+)/);
  if (!m) throw new Error('Bad HDR resolution line: ' + resLine);
  // Second pair is along X (scanline), first is along Y.
  // We treat data as top-down regardless of sign for simplicity.
  const height = parseInt(m[2], 10);
  const width = parseInt(m[4], 10);

  const data = new Float32Array(width * height * 3);

  // Decode scanlines
  let p = offset;
  for (let y = 0; y < height; y++) {
    if (p + 4 > bytes.length) throw new Error('HDR truncated at scanline ' + y);
    const b0 = bytes[p], b1 = bytes[p + 1], b2 = bytes[p + 2], b3 = bytes[p + 3];
    if (b0 === 2 && b1 === 2 && (b2 & 0x80) === 0) {
      // New RLE format
      const scanWidth = (b2 << 8) | b3;
      if (scanWidth !== width) throw new Error('HDR scanline width mismatch');
      p += 4;
      const channels = new Uint8Array(width * 4);
      for (let ch = 0; ch < 4; ch++) {
        let x = 0;
        while (x < width) {
          if (p >= bytes.length) throw new Error('HDR truncated in RLE');
          const count = bytes[p++];
          if (count > 128) {
            // Run: repeat next byte
            const run = count - 128;
            if (p >= bytes.length) throw new Error('HDR truncated in run');
            const val = bytes[p++];
            for (let i = 0; i < run; i++) channels[(x + i) * 4 + ch] = val;
            x += run;
          } else {
            // Non-run: copy `count` literal bytes
            for (let i = 0; i < count; i++) {
              if (p >= bytes.length) throw new Error('HDR truncated in literal');
              channels[(x + i) * 4 + ch] = bytes[p++];
            }
            x += count;
          }
        }
      }
      // Convert RGBE → float
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 3;
        const i4 = x * 4;
        rgbeToFloat(channels[i4], channels[i4 + 1], channels[i4 + 2], channels[i4 + 3], data, idx);
      }
    } else {
      // Old format: uncompressed or simple RLE
      // Fall back: read width pixels of 4 bytes each (no RLE)
      // This branch is rarely needed for typical .hdr files.
      for (let x = 0; x < width; x++) {
        if (p + 4 > bytes.length) throw new Error('HDR truncated in old format');
        const r = bytes[p++], g = bytes[p++], b = bytes[p++], e = bytes[p++];
        rgbeToFloat(r, g, b, e, data, (y * width + x) * 3);
      }
    }
  }
  return { width, height, data };
}

// Tonemap (Reinhard) + gamma → ImageData (sRGB 8-bit).
export function tonemapToImageData(hdr, exposure = 1.0) {
  const { width, height, data } = hdr;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
    let r = data[i] * exposure;
    let g = data[i + 1] * exposure;
    let b = data[i + 2] * exposure;
    // Reinhard
    r = r / (1 + r);
    g = g / (1 + g);
    b = b / (1 + b);
    // Gamma 2.2 (approx sRGB)
    out[j]     = Math.round(Math.pow(r, 1 / 2.2) * 255);
    out[j + 1] = Math.round(Math.pow(g, 1 / 2.2) * 255);
    out[j + 2] = Math.round(Math.pow(b, 1 / 2.2) * 255);
    out[j + 3] = 255;
  }
  return new ImageData(out, width, height);
}
