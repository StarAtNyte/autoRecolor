from __future__ import annotations
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from autoRecolor.palette import Palette, ColorEntry
from autoRecolor.utils import rgb_to_hsl, rgb_to_hex, rgb_to_hsl_batch


# Images with this many unique colors or fewer skip K-means entirely —
# their exact palette is extracted directly for pixel-perfect mapping.
EXACT_COLOR_THRESHOLD = 64


class ClusterStats:
    def __init__(self, cluster_id: int, pixels_hsl: np.ndarray):
        self.cluster_id = cluster_id
        self.mean_hsl = pixels_hsl.mean(axis=0).tolist()
        self.std_hsl = pixels_hsl.std(axis=0).tolist()
        self.count = len(pixels_hsl)


def analyze_image(
    image_path: str,
    n_colors: int = 6,
    resize_max: int = 512,
) -> tuple[Palette, Image.Image, np.ndarray, list[ClusterStats]]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    pixels_full = np.array(img).reshape(-1, 3)  # (N, 3) uint8

    # Check how many unique colours the image actually contains
    unique_rgb, inverse = np.unique(pixels_full, axis=0, return_inverse=True)
    n_unique = len(unique_rgb)

    if n_unique <= EXACT_COLOR_THRESHOLD:
        print(f"  ↳ {n_unique} unique colours — using exact extraction (no clustering)")
        return _analyze_exact(img, pixels_full, unique_rgb, inverse, image_path)
    else:
        print(f"  ↳ {n_unique} unique colours — clustering to {n_colors}")
        return _analyze_clustered(img, pixels_full, n_colors, resize_max, image_path)


# ── Exact extraction (quantized / flat-colour images) ─────────────────────────

def _analyze_exact(
    img: Image.Image,
    pixels_full: np.ndarray,
    unique_rgb: np.ndarray,
    inverse: np.ndarray,
    image_path: str,
) -> tuple[Palette, Image.Image, np.ndarray, list[ClusterStats]]:
    w, h = img.size
    n_colors = len(unique_rgb)
    counts = np.bincount(inverse, minlength=n_colors)
    total = counts.sum()
    order = np.argsort(-counts)   # most-frequent first

    pixels_hsl_full = rgb_to_hsl_batch(pixels_full)
    palette_colors: list[ColorEntry] = []
    cluster_stats_list: list[ClusterStats] = []

    for rank, idx in enumerate(order):
        rgb = unique_rgb[idx].tolist()
        hex_color = rgb_to_hex(*rgb)
        hsl = rgb_to_hsl(*rgb)
        pct = round(counts[idx] / total * 100, 1)

        palette_colors.append(ColorEntry(
            id=int(idx),
            label=_guess_label(rgb),
            hex=hex_color,
            rgb=rgb,
            hsl=hsl,
            pixel_percent=pct,
        ))

        mask = inverse == idx
        cluster_pixels_hsl = pixels_hsl_full[mask]
        if len(cluster_pixels_hsl) > 0:
            cluster_stats_list.append(ClusterStats(int(idx), cluster_pixels_hsl))

    dedupe_labels(palette_colors)

    # label_map: each pixel → index into unique_rgb (== ColorEntry.id)
    label_map = inverse.reshape(img.size[1], img.size[0])

    palette = Palette(
        image_path=image_path,
        image_width=w,
        image_height=h,
        palette=palette_colors,
    )
    return palette, img, label_map, cluster_stats_list


# ── K-means clustering (photos / complex illustrations) ───────────────────────

def _analyze_clustered(
    img: Image.Image,
    pixels_full: np.ndarray,
    n_colors: int,
    resize_max: int,
    image_path: str,
) -> tuple[Palette, Image.Image, np.ndarray, list[ClusterStats]]:
    w, h = img.size

    # Fit K-means on a downsampled version for speed
    if max(w, h) > resize_max:
        ratio = resize_max / max(w, h)
        img_small = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        pixels_small = np.array(img_small).reshape(-1, 3).astype(np.float32)
    else:
        pixels_small = pixels_full.astype(np.float32)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels_small)
    centroids = kmeans.cluster_centers_.astype(int)

    # Predict on full-resolution pixels
    labels_full = kmeans.predict(pixels_full.astype(np.float32))

    unique, counts = np.unique(labels_full, return_counts=True)
    total = counts.sum()
    order = np.argsort(-counts)

    pixels_hsl_full = rgb_to_hsl_batch(pixels_full)
    palette_colors: list[ColorEntry] = []
    cluster_stats_list: list[ClusterStats] = []

    for rank, idx in enumerate(order):
        centroid = centroids[idx].tolist()
        hex_color = rgb_to_hex(*centroid)
        hsl = rgb_to_hsl(*centroid)
        pct = round(counts[idx] / total * 100, 1)

        palette_colors.append(ColorEntry(
            id=int(idx),
            label=_guess_label(centroid),
            hex=hex_color,
            rgb=centroid,
            hsl=hsl,
            pixel_percent=pct,
        ))

        mask = labels_full == idx
        cluster_pixels_hsl = pixels_hsl_full[mask]
        if len(cluster_pixels_hsl) > 0:
            cluster_stats_list.append(ClusterStats(int(idx), cluster_pixels_hsl))

    dedupe_labels(palette_colors)
    label_map = labels_full.reshape(img.size[1], img.size[0])

    palette = Palette(
        image_path=image_path,
        image_width=w,
        image_height=h,
        palette=palette_colors,
    )
    return palette, img, label_map, cluster_stats_list


# ── Color labelling ───────────────────────────────────────────────────────────

def _guess_label(rgb: list[int]) -> str:
    """HSL-based colour name — covers grays, browns, and the full hue wheel."""
    h, s, l = rgb_to_hsl(*rgb)

    if s < 12:
        if l < 12:  return "black"
        if l < 30:  return "dark_gray"
        if l < 60:  return "gray"
        if l < 85:  return "light_gray"
        return "white"

    # Only call it black/white if truly achromatic-looking, not just very dark/light
    if l < 15 and s < 40:  return "black"
    if l > 88 and s < 20:  return "white"

    if s < 30 and 20 < h < 50:
        return "brown" if l < 50 else "tan"

    if h < 15 or h >= 345:  hue_name = "red"
    elif h < 40:             hue_name = "orange"
    elif h < 70:             hue_name = "yellow"
    elif h < 155:            hue_name = "green"
    elif h < 195:            hue_name = "cyan"
    elif h < 255:            hue_name = "blue"
    elif h < 290:            hue_name = "purple"
    elif h < 345:            hue_name = "pink"
    else:                    hue_name = "color"

    # Prefix with lightness tier so duplicate hue names stay distinct
    if l < 25:   return f"dark_{hue_name}"
    if l > 75:   return f"light_{hue_name}"
    return hue_name


def dedupe_labels(palette_colors: list) -> list:
    """Append a numeric suffix to any labels that appear more than once."""
    from collections import Counter
    counts = Counter(c.label for c in palette_colors)
    seen: dict[str, int] = {}
    for c in palette_colors:
        if counts[c.label] > 1:
            seen[c.label] = seen.get(c.label, 0) + 1
            c.label = f"{c.label}_{seen[c.label]}"
    return palette_colors
