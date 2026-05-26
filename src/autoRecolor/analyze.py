from __future__ import annotations
import numpy as np
from PIL import Image
from autoRecolor.palette import Palette, ColorEntry
from autoRecolor.utils import (
    rgb_to_hsl, rgb_to_hex, rgb_to_hsl_batch,
    rgb_to_oklab_batch, hyab_dist_matrix,
)


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

    unique_rgb, inverse = np.unique(pixels_full, axis=0, return_inverse=True)
    n_unique = len(unique_rgb)

    if n_unique <= EXACT_COLOR_THRESHOLD:
        print(f"  ↳ {n_unique} unique colours — using exact extraction (no clustering)")
        return _analyze_exact(img, pixels_full, unique_rgb, inverse, image_path)
    else:
        print(f"  ↳ {n_unique} unique colours — clustering to {n_colors} (OKLAB + HyAB)")
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
            # oklab is auto-computed in __post_init__
        ))

        mask = inverse == idx
        cluster_pixels_hsl = pixels_hsl_full[mask]
        if len(cluster_pixels_hsl) > 0:
            cluster_stats_list.append(ClusterStats(int(idx), cluster_pixels_hsl))

    dedupe_labels(palette_colors)

    label_map = inverse.reshape(img.size[1], img.size[0])

    palette = Palette(
        image_path=image_path,
        image_width=w,
        image_height=h,
        palette=palette_colors,
    )
    return palette, img, label_map, cluster_stats_list


# ── HyAB K-Means in OKLAB space ───────────────────────────────────────────────

def _hyab_kmeans(
    pixels_oklab: np.ndarray,
    k: int,
    max_iter: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Means++ initialised K-Means using HyAB distance in OKLAB space.
    Returns (centers, labels) where centers is (k, 3) float64 and labels is (N,) int.
    """
    n = len(pixels_oklab)
    rng = np.random.default_rng(seed)

    # ── K-Means++ initialisation ──────────────────────────────────────────────
    first_idx = int(rng.integers(n))
    centers = [pixels_oklab[first_idx].copy()]

    for _ in range(k - 1):
        cents = np.array(centers)                          # (K_so_far, 3)
        dists = hyab_dist_matrix(pixels_oklab, cents)      # (N, K_so_far)
        min_dists = dists.min(axis=1)                      # (N,)
        probs = min_dists ** 2
        total = probs.sum()
        if total == 0:
            centers.append(pixels_oklab[int(rng.integers(n))].copy())
        else:
            probs /= total
            idx = int(rng.choice(n, p=probs))
            centers.append(pixels_oklab[idx].copy())

    centers = np.array(centers)   # (k, 3)
    labels = np.zeros(n, dtype=np.int32)

    # ── Lloyd iterations ──────────────────────────────────────────────────────
    for _ in range(max_iter):
        dists   = hyab_dist_matrix(pixels_oklab, centers)  # (N, k)
        new_labels = dists.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ki in range(k):
            mask = labels == ki
            if mask.sum() > 0:
                centers[ki] = pixels_oklab[mask].mean(axis=0)

    return centers, labels


# ── K-Means clustering (photos / complex illustrations) ───────────────────────

def _analyze_clustered(
    img: Image.Image,
    pixels_full: np.ndarray,
    n_colors: int,
    resize_max: int,
    image_path: str,
) -> tuple[Palette, Image.Image, np.ndarray, list[ClusterStats]]:
    w, h = img.size

    # Downsample for speed when fitting
    if max(w, h) > resize_max:
        ratio = resize_max / max(w, h)
        img_small = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        pixels_small = np.array(img_small).reshape(-1, 3).astype(np.uint8)
    else:
        pixels_small = pixels_full.copy()

    # Convert small pixels to OKLAB, cluster with HyAB
    small_oklab = rgb_to_oklab_batch(pixels_small)          # (N_small, 3)
    centers_oklab, _ = _hyab_kmeans(small_oklab, n_colors)  # fit on small

    # Assign full-res pixels using HyAB (chunked for memory safety)
    full_oklab   = rgb_to_oklab_batch(pixels_full)                      # (N, 3)
    dists_full   = hyab_dist_matrix(full_oklab, centers_oklab)          # (N, K)
    labels_full  = dists_full.argmin(axis=1).astype(np.int32)           # (N,)

    # Centroid RGB: round mean of pixels in each cluster (done in RGB not OKLAB
    # so that saved hex looks natural on screen)
    unique, counts = np.unique(labels_full, return_counts=True)
    total = counts.sum()
    order = np.argsort(-counts)   # most-frequent first

    pixels_hsl_full = rgb_to_hsl_batch(pixels_full)
    palette_colors: list[ColorEntry] = []
    cluster_stats_list: list[ClusterStats] = []

    for rank, idx in enumerate(order):
        # Centroid in OKLAB → convert back to RGB for the palette entry
        import numpy as _np
        oklab_c = centers_oklab[idx]
        from autoRecolor.utils import oklab_to_rgb_batch
        centroid_rgb = oklab_to_rgb_batch(oklab_c[_np.newaxis])[0].tolist()

        hex_color = rgb_to_hex(*centroid_rgb)
        hsl = rgb_to_hsl(*centroid_rgb)
        pct = round(counts[idx] / total * 100, 1)

        palette_colors.append(ColorEntry(
            id=int(idx),
            label=_guess_label(centroid_rgb),
            hex=hex_color,
            rgb=centroid_rgb,
            hsl=hsl,
            pixel_percent=pct,
            oklab=[round(float(oklab_c[0]), 5),
                   round(float(oklab_c[1]), 5),
                   round(float(oklab_c[2]), 5)],
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
