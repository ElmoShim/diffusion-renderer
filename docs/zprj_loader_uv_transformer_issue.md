# Expose per-fabric UV transformer (texture tile width / angle / position)

## Summary

`ColorwayUVMatrix.uv_transformer_index` references a "UV transformer" object, but the transformer itself is not exposed by the Python bindings. As a result, downstream renderers cannot reproduce the texture tile size that CLO displays in the Fabric panel (the "width", "angle", "position X/Y" fields of the per-fabric texture transformation).

For fabrics displayed at 100% scale, the texture tile size can be recovered from the image's embedded DPI (`(pixels / dpi) * 25.4`). For any other percentage, the necessary scale factor is missing.

## Reproduction

Using two sample files: `garment.zprj` (one fabric at 100%) and `outfit.zprj` (three fabrics, one of which is at 11.3% and one at 226.1%).

```
Tartan check  (garment.zprj): CLO shows 118.9 mm × 128.1 mm  (100%)
top           (outfit.zprj):  CLO shows  40.8 mm ×  40.8 mm  (11.3%)
Default_Fabric(outfit.zprj):  CLO shows 229.7 mm × ...        (226.1%)
```

### What the Python bindings expose

| Source | Field | Value (Tartan) | Value (top) | Value (Default) |
|---|---|---|---|---|
| `FabricMaterial` | `tile_width` | 1117.60 mm | 1117.60 mm | 1117.60 mm |
| `FabricMaterial` | `tile_height` | 746.55 mm | 321.86 mm | 325.48 mm |
| `FabricMaterial.diffuse_texture_transform` | `scale_u` / `scale_v` | 0.001 | 0.001 | 0.001 |
| `FabricMaterial.diffuse_texture_transform` | `offset_u`/`offset_v`/`rotation` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| `ColorwayUVMatrix.local_to_uv` (2×2 scale) | — | 0.294 | 0.224 | 0.371 |
| Image DPI tile = `(px / dpi) × 25.4` | — | 118.89 mm | 361.24 mm | 101.60 mm |
| **CLO displayed width (target)** | — | **118.9 mm** | **40.8 mm** | **229.7 mm** |

Observations:

- `tile_width = 1117.60 mm` for every fabric. That's exactly 44 inches — the standard fabric bolt width. It is not the displayed texture tile.
- `diffuse_texture_transform.scale_u` is `0.001` whenever a diffuse texture exists, and `1.0` when it does not. It behaves like an on/off flag, not a user-set scale.
- `ColorwayUVMatrix.local_to_uv` carries a per-pattern affine transform. The 2×2 scale is similar within a fabric (most "top" patterns are 0.224, most "Default_Fabric" patterns are 0.371) but does not yield the CLO width through any consistent formula:

  ```
  scale × DPI_tile     →  34.96 (Tartan)  /  80.92 (top)  /  37.69 (Default)
  DPI_tile / scale     →  404.5 / 1611.6 /  273.9
  ```

  None of these match the targets (118.9 / 40.8 / 229.7). And `local_to_uv` already contains rotation and translation, suggesting it encodes the pattern's grain line — not the fabric's texture scale.

### What the bindings hint at

From `zprj_loader.pyi`:

```python
class ColorwayUVMatrix:
    """Per-colorway UV transformation matrix assignment for a fabric element."""
    @property
    def item_uid(self) -> int: ...
    @property
    def local_to_uv(self) -> numpy.ndarray[numpy.float32]: ...
    @property
    def uv_transformer_index(self) -> int:
        """Index of the UV transformer."""
```

`uv_transformer_index` is documented as an index — but there is no `UVTransformer` class in `__all__`, no `uv_transformers` collection on `Scene` or `Colorway`, and nothing on `FabricMaterial` that exposes width/angle/position for the fabric's texture mapping. The underlying object referenced by this index is invisible to Python.

In the sample files, `uv_transformer_index` takes values 0 and 1; index 1 only appears on a couple of identity-matrix entries that look like graphic placeholders.

## Why this matters

CLO's Fabric panel shows a "transformation" group per fabric with `angle`, `width`, `position X`, `position Y`. The `width` field is what determines the displayed tile size in millimeters. Without it, an offline renderer can only get the texture scale right for fabrics that happen to be at 100%; everything else is rendered at the wrong scale (e.g. the `top` sweater in `outfit.zprj` renders ~8.85× larger than CLO).

## Proposed change

Expose the UV transformer data referenced by `ColorwayUVMatrix.uv_transformer_index`. A minimal shape:

```python
class UVTransformer:
    """Per-fabric (or per-item) UV transformation set in CLO's Fabric panel."""
    @property
    def width(self) -> float:
        """Displayed texture tile width in mm."""
    @property
    def height(self) -> float:
        """Displayed texture tile height in mm."""
    @property
    def angle(self) -> float:
        """Texture rotation in degrees."""
    @property
    def position(self) -> numpy.ndarray[numpy.float32]:
        """Texture offset (X, Y) in mm."""
```

And expose the collection at the scope where it lives (likely `Colorway` or `Scene`):

```python
class Colorway:
    @property
    def uv_transformers(self) -> list[UVTransformer]: ...
```

so `ColorwayUVMatrix.uv_transformer_index` can be resolved into the actual transformer.

## Workaround

For fabrics displayed at 100%, the displayed tile size matches the image's embedded DPI:

```python
tile_mm = (image_px / dpi) * 25.4
```

There is no reliable workaround for fabrics displayed at other percentages without the UV transformer data.
