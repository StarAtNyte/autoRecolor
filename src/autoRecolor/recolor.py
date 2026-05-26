from __future__ import annotations
import numpy as np
from PIL import Image
from autoRecolor.palette import Palette
from autoRecolor.utils import rgb_to_oklab_batch, oklab_to_rgb_batch


def recolor_image(
    img: Image.Image,
    original_palette: Palette,
    modified_palette: Palette,
    label_map: np.ndarray,
) -> Image.Image:
    """
    Recolour img by shifting each cluster's pixels in OKLAB space.

    The agent specifies changes via HSL/hex (intuitive for reasoning).
    We compute the centroid-to-centroid delta in perceptually-uniform OKLAB
    and apply it to every pixel in the cluster — so equal shifts look equal
    across hues and lightness levels.
    """
    pixels = np.array(img)          # (H, W, 3) uint8
    output = pixels.copy()

    orig_map = {c.id: c for c in original_palette.palette}
    mod_map  = {c.id: c for c in modified_palette.palette}

    for cid, orig_entry in orig_map.items():
        mod_entry = mod_map.get(cid, orig_entry)

        # Skip clusters with no change
        if orig_entry.hex == mod_entry.hex:
            continue

        mask = label_map == cid
        region_pixels = pixels[mask]        # (N, 3) uint8
        if len(region_pixels) == 0:
            continue

        # Delta = target centroid − source centroid, both in OKLAB
        orig_lab = rgb_to_oklab_batch(np.array([orig_entry.rgb], dtype=np.uint8))[0]
        new_lab  = rgb_to_oklab_batch(np.array([mod_entry.rgb],  dtype=np.uint8))[0]
        delta    = new_lab - orig_lab       # (3,) float64

        # Shift every pixel in the cluster by the same OKLAB delta
        region_lab = rgb_to_oklab_batch(region_pixels)  # (N, 3) float64
        region_lab += delta

        output[mask] = oklab_to_rgb_batch(region_lab)   # (N, 3) uint8

    return Image.fromarray(output, mode="RGB")
