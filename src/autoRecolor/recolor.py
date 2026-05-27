from __future__ import annotations
import numpy as np
from PIL import Image
from autoRecolor.palette import Palette
from autoRecolor.utils import rgb_to_oklab_batch, oklab_to_rgb_batch

# ── OKLCH helpers ─────────────────────────────────────────────────────────────

def _to_lch(lab: np.ndarray) -> np.ndarray:
    """(N,3) OKLAB → (N,3) OKLCH  [L, C, h_rad]"""
    C = np.hypot(lab[:, 1], lab[:, 2])
    h = np.arctan2(lab[:, 2], lab[:, 1])
    return np.stack([lab[:, 0], C, h], axis=1)


def _to_lab(lch: np.ndarray) -> np.ndarray:
    """(N,3) OKLCH [L, C, h_rad] → (N,3) OKLAB"""
    a = lch[:, 1] * np.cos(lch[:, 2])
    b = lch[:, 1] * np.sin(lch[:, 2])
    return np.stack([lch[:, 0], a, b], axis=1)


# ── Gamut clamping ────────────────────────────────────────────────────────────

# Approximate max chroma for OKLAB within sRGB at a given L.
# Derived empirically: the sRGB gamut boundary in OKLAB is roughly
# a tent function peaking around L≈0.55 with C_max≈0.4.
# We use a conservative polynomial fit so we never over-saturate.
def _max_chroma(L: np.ndarray) -> np.ndarray:
    """Approximate max sRGB-safe chroma for each L value."""
    L = np.clip(L, 0.0, 1.0)
    # Tent: rises linearly to ~0.37 at L=0.55, falls to 0 at 0 and 1
    return np.where(
        L < 0.55,
        L * (0.37 / 0.55),
        (1.0 - L) * (0.37 / 0.45),
    )


def _clamp_chroma(lch: np.ndarray) -> np.ndarray:
    """Reduce chroma so it stays inside the approximate sRGB gamut."""
    out = lch.copy()
    max_C = _max_chroma(lch[:, 0])
    out[:, 1] = np.minimum(lch[:, 1], max_C)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    return out


# ── Lightness remap ───────────────────────────────────────────────────────────

def _remap_lightness(
    L_pixels: np.ndarray,
    orig_L_cent: float,
    new_L_cent: float,
) -> np.ndarray:
    """
    Remap pixel lightness values so the centroid moves from orig_L_cent to
    new_L_cent while preserving tonal relationships within the cluster.

    Strategy: shift + soft compress toward the destination pole so that
    pixels already near the gamut boundary don't blow out.
      - Pure additive delta works well for small shifts.
      - For large shifts we blend toward a proportional remap so shadows
        and highlights move in the right direction without clipping.
    """
    dL = new_L_cent - orig_L_cent
    if abs(dL) < 1e-5:
        return L_pixels

    L_out = L_pixels + dL

    # Soft-compress pixels that overshoot [0, 1]
    # Use a smooth tanh-based toe/shoulder so detail is preserved.
    def _soft_clamp(x: np.ndarray) -> np.ndarray:
        # Map to [-1,1] range then expand back — preserves interior, rolls off edges
        margin = 0.05
        lo, hi = margin, 1.0 - margin
        below = x < lo
        above = x > hi
        x = x.copy()
        if below.any():
            x[below] = lo * np.tanh(x[below] / lo)
        if above.any():
            excess = x[above] - hi
            x[above] = hi + (1 - hi) * np.tanh(excess / (1 - hi))
        return x

    return _soft_clamp(np.clip(L_out, -0.1, 1.1))


# ── Main recolor ──────────────────────────────────────────────────────────────

def recolor_image(
    img: Image.Image,
    original_palette: Palette,
    modified_palette: Palette,
    label_map: np.ndarray,
) -> Image.Image:
    """
    Recolour img by transforming each cluster's pixels in OKLCH space.

    Per-cluster transform:
      L  — shifted additively + soft-clamped to preserve shadows/highlights
      C  — scaled multiplicatively (vivid pixels stay vivid proportionally);
           hard cap at sRGB gamut boundary to prevent blowout;
           achromatic source clusters use additive nudge instead
      h  — rotated by the centroid's hue rotation (Δh);
           achromatic source clusters (C<threshold) adopt target hue directly,
           blending in proportion to how chromatic each pixel actually is
    """
    pixels = np.array(img)          # (H, W, 3) uint8
    output = pixels.copy()

    orig_map = {c.id: c for c in original_palette.palette}
    mod_map  = {c.id: c for c in modified_palette.palette}

    for cid, orig_entry in orig_map.items():
        mod_entry = mod_map.get(cid, orig_entry)

        if orig_entry.hex == mod_entry.hex:
            continue

        mask = label_map == cid
        region_pixels = pixels[mask]
        if len(region_pixels) == 0:
            continue

        # ── Centroid deltas ───────────────────────────────────────────────
        orig_lab = rgb_to_oklab_batch(np.array([orig_entry.rgb], dtype=np.uint8))
        new_lab  = rgb_to_oklab_batch(np.array([mod_entry.rgb],  dtype=np.uint8))
        o_lch = _to_lch(orig_lab)[0]   # (L, C, h)
        n_lch = _to_lch(new_lab)[0]

        orig_C_cent = o_lch[1]
        new_C_cent  = n_lch[1]
        dH          = n_lch[2] - o_lch[2]   # hue rotation (radians)

        # Chroma strategy
        achromatic_source = orig_C_cent < 0.03   # centroid was basically gray

        if not achromatic_source:
            chroma_scale = new_C_cent / orig_C_cent
        else:
            # Can't scale from near-zero — nudge additively and set hue directly
            chroma_scale = None

        # ── Convert region to OKLCH ───────────────────────────────────────
        region_lab = rgb_to_oklab_batch(region_pixels)
        region_lch = _to_lch(region_lab)          # (N, 3)

        # ── Lightness ─────────────────────────────────────────────────────
        region_lch[:, 0] = _remap_lightness(
            region_lch[:, 0], o_lch[0], n_lch[0]
        )

        # ── Hue rotation ──────────────────────────────────────────────────
        if achromatic_source:
            # Source centroid was achromatic — set hue directly to target,
            # blended by how chromatic each pixel is (gray pixels unaffected)
            px_C = region_lch[:, 1]
            blend = np.clip(px_C / 0.05, 0, 1)   # 0=gray stays, 1=vivid rotates
            target_h = n_lch[2]
            region_lch[:, 2] = (1 - blend) * region_lch[:, 2] + blend * target_h
        else:
            region_lch[:, 2] += dH

        # ── Chroma ────────────────────────────────────────────────────────
        if achromatic_source:
            # Add chroma in proportion to pixel's original chroma
            px_C = region_lch[:, 1]
            blend = np.clip(px_C / 0.05, 0, 1)
            region_lch[:, 1] = px_C + blend * (new_C_cent - orig_C_cent)
        else:
            region_lch[:, 1] *= chroma_scale

        # Clamp chroma to sRGB gamut boundary + non-negative
        region_lch = _clamp_chroma(region_lch)

        # ── Back to RGB ───────────────────────────────────────────────────
        region_lab_out = _to_lab(region_lch)
        output[mask] = oklab_to_rgb_batch(region_lab_out)

    return Image.fromarray(output, mode="RGB")
