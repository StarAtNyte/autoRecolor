from __future__ import annotations
import math
import numpy as np


def hex_to_rgb(hex_color: str) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_to_hsl(r: int, g: int, b: int) -> list[float]:
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rn, gn, bn), min(rn, gn, bn)
    l = (mx + mn) / 2
    if mx == mn:
        return [0.0, 0.0, round(l * 100, 1)]
    d = mx - mn
    s = d / (1 - abs(2 * l - 1))
    if mx == rn:
        h = (gn - bn) / d + (6 if gn < bn else 0)
    elif mx == gn:
        h = (bn - rn) / d + 2
    else:
        h = (rn - gn) / d + 4
    h /= 6
    return [round(h * 360, 1), round(s * 100, 1), round(l * 100, 1)]


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    h, s, l = h / 360.0, s / 100.0, l / 100.0
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = l - c / 2
    if h < 1 / 6:
        rn, gn, bn = c, x, 0
    elif h < 2 / 6:
        rn, gn, bn = x, c, 0
    elif h < 3 / 6:
        rn, gn, bn = 0, c, x
    elif h < 4 / 6:
        rn, gn, bn = 0, x, c
    elif h < 5 / 6:
        rn, gn, bn = x, 0, c
    else:
        rn, gn, bn = c, 0, x
    return (
        round((rn + m) * 255),
        round((gn + m) * 255),
        round((bn + m) * 255),
    )


# ── Vectorized batch conversions (numpy) ─────────────────────────────────────

def rgb_to_hsl_batch(pixels: np.ndarray) -> np.ndarray:
    """Convert (N, 3) uint8 RGB array → (N, 3) float32 HSL [0-360, 0-100, 0-100]."""
    p = pixels.astype(np.float32) / 255.0
    r, g, b = p[:, 0], p[:, 1], p[:, 2]

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn

    l = (mx + mn) / 2.0

    denom = np.maximum(1.0 - np.abs(2.0 * l - 1.0), 1e-10)
    s = np.where(d == 0, 0.0, d / denom)

    h = np.zeros_like(r)
    nz = d > 0
    mr = nz & (mx == r)
    mg = nz & (mx == g)
    mb = nz & (mx == b)
    h[mr] = ((g[mr] - b[mr]) / d[mr]) % 6.0
    h[mg] = (b[mg] - r[mg]) / d[mg] + 2.0
    h[mb] = (r[mb] - g[mb]) / d[mb] + 4.0
    h /= 6.0

    return np.stack([h * 360.0, s * 100.0, l * 100.0], axis=1).astype(np.float32)


def hsl_to_rgb_batch(hsl: np.ndarray) -> np.ndarray:
    """Convert (N, 3) float HSL [0-360, 0-100, 0-100] → (N, 3) uint8 RGB."""
    h = hsl[:, 0] / 360.0
    s = hsl[:, 1] / 100.0
    l = hsl[:, 2] / 100.0

    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    h6 = h * 6.0
    x = c * (1.0 - np.abs(h6 % 2.0 - 1.0))
    m = l - c / 2.0

    hi = np.floor(h6).astype(np.int32) % 6
    z = np.zeros_like(c)

    rr = np.where((hi == 0) | (hi == 5), c, np.where((hi == 1) | (hi == 4), x, z))
    gg = np.where((hi == 1) | (hi == 2), c, np.where((hi == 0) | (hi == 3), x, z))
    bb = np.where((hi == 3) | (hi == 4), c, np.where((hi == 2) | (hi == 5), x, z))

    return np.stack([
        np.clip((rr + m) * 255.0, 0, 255),
        np.clip((gg + m) * 255.0, 0, 255),
        np.clip((bb + m) * 255.0, 0, 255),
    ], axis=1).astype(np.uint8)


def color_distance(c1: list[int], c2: list[int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


# ── OKLAB (perceptually uniform) ──────────────────────────────────────────────
# Matrices from Björn Ottosson  https://bottosson.github.io/posts/oklab/

_M1 = np.array([                        # linear RGB → LMS
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float64)

_M2 = np.array([                        # LMS^(1/3) → OKLAB
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float64)

_M1_INV = np.array([                    # LMS → linear RGB
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
], dtype=np.float64)

_M2_INV = np.array([                    # OKLAB → LMS^(1/3)
    [1.0000000000,  0.3963377774,  0.2158037573],
    [1.0000000000, -0.1055613458, -0.0638541728],
    [1.0000000000, -0.0894841775, -1.2914855480],
], dtype=np.float64)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def rgb_to_oklab_batch(pixels: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 RGB → (N, 3) float64 OKLAB  [L≈0-1, a≈±0.4, b≈±0.4]."""
    lin = _srgb_to_linear(pixels.astype(np.float64) / 255.0)  # (N,3)
    lms = lin @ _M1.T                                          # (N,3)
    lms_cbrt = np.cbrt(lms)                                    # (N,3)
    return lms_cbrt @ _M2.T                                    # (N,3)


def oklab_to_rgb_batch(lab: np.ndarray) -> np.ndarray:
    """(N, 3) float64 OKLAB → (N, 3) uint8 RGB  (out-of-gamut values clamped)."""
    lms_cbrt = lab @ _M2_INV.T
    lms = lms_cbrt ** 3
    lin = lms @ _M1_INV.T
    srgb = _linear_to_srgb(lin)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


# ── Scalar OKLAB helpers (for palette-level use) ──────────────────────────────

def hex_to_oklab(hex_color: str) -> tuple[float, float, float]:
    """Hex string → (L, a, b) OKLAB floats."""
    rgb = hex_to_rgb(hex_color)
    arr = rgb_to_oklab_batch(np.array([rgb], dtype=np.uint8))[0]
    return float(arr[0]), float(arr[1]), float(arr[2])


def oklab_to_hex(L: float, a: float, b: float) -> str:
    """(L, a, b) OKLAB → hex string (clamped to sRGB gamut)."""
    arr = oklab_to_rgb_batch(np.array([[L, a, b]], dtype=np.float64))[0]
    return rgb_to_hex(int(arr[0]), int(arr[1]), int(arr[2]))


# ── HyAB distance ─────────────────────────────────────────────────────────────
# Manhattan on L + Euclidean on ab  — better than Euclidean for OKLAB clustering
# (notes.md: "OKLAB ko space my HyAB ko distance")

def hyab_dist_matrix(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Vectorized HyAB distance: (N, 3) points × (K, 3) centers → (N, K) distances.
    Memory-safe: processes in chunks to avoid allocating giant arrays.
    """
    N, K = len(points), len(centers)
    out = np.empty((N, K), dtype=np.float64)
    CHUNK = 50_000
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk = points[start:end]                              # (C, 3)
        dL   = np.abs(chunk[:, 0:1] - centers[:, 0])          # (C, K)
        diff_ab = chunk[:, np.newaxis, 1:] - centers[np.newaxis, :, 1:]  # (C, K, 2)
        dab  = np.linalg.norm(diff_ab, axis=2)                # (C, K)
        out[start:end] = dL + dab
    return out


# ── OKLAB perceptual axis shifts ──────────────────────────────────────────────
# Ported from ColorWay/Mycolor.studio/lib/colorways.jsx — applyOklabAxisPalette
#
# Axes and their semantics:
#   lightness   — additive L shift  (+0.1 = noticeably brighter)
#   warmth      — along blackbody locus D65→A  (+0.1 = warmer amber/orange)
#   chroma      — radial ab push, hue angle preserved exactly  (+0.1 = more vivid)
#   hue         — ab-plane rotation in degrees  (+30 = 30° CCW shift)
#   exposure    — multiplicative L in photographic stops  (+1 = 2× brighter)
#   saturation  — proportional chroma scale  (+0.5 = 50% more saturated)
#   purity      — jewel-tone: darken + boost chroma (+) / lighten + desaturate (-)
#   mute        — pull toward gray/mid-L (+) / boost contrast+chroma (-)
#   green_red   — direct a-channel shift  (+ = redder, − = greener)
#   blue_yellow — direct b-channel shift  (+ = yellower, − = bluer)
#   contrast    — expand/compress L around palette mean (palette-level only)

_WARM_A = 0.216   # blackbody locus direction in ab-plane (validated)
_WARM_B = 0.976


def oklab_axis_shift(
    L: float, a: float, b: float,
    axis: str, delta: float,
    mean_L: float = 0.5,
) -> tuple[float, float, float]:
    """
    Apply a single named perceptual axis shift in OKLAB space.
    mean_L is only used for the 'contrast' axis (pass the palette mean L).
    Returns new (L, a, b).
    """
    if axis == "lightness":
        return L + delta, a, b

    elif axis == "warmth":
        return L, a + delta * _WARM_A, b + delta * _WARM_B

    elif axis == "chroma":
        C = math.hypot(a, b)
        safe_C = max(C, 1e-10)
        scale = max((C + delta) / safe_C, 0.0)
        return L, a * scale, b * scale

    elif axis == "hue":
        t = delta * math.pi / 180.0
        cos_t, sin_t = math.cos(t), math.sin(t)
        return L, a * cos_t - b * sin_t, a * sin_t + b * cos_t

    elif axis == "exposure":
        return L * (2.0 ** delta), a, b

    elif axis == "saturation":
        f = max(1.0 + delta, 0.0)
        return L, a * f, b * f

    elif axis == "purity":
        # +delta → darker AND more chromatic (jewel/ink)
        # -delta → lighter AND less chromatic (chalk/pastel)
        return L * (1.0 - delta), a * (1.0 + delta), b * (1.0 + delta)

    elif axis == "mute":
        # +delta → less chromatic, L pulled toward 0.5 (corporate/neutral)
        return L + 0.3 * delta * (0.5 - L), a * (1.0 - delta), b * (1.0 - delta)

    elif axis == "green_red":
        return L, a + delta, b

    elif axis == "blue_yellow":
        return L, a, b + delta

    elif axis == "contrast":
        # Expand/compress around the palette's mean L
        return mean_L + (L - mean_L) * (1.0 + delta), a, b

    else:
        return L, a, b


def apply_oklab_axes(
    L: float, a: float, b: float,
    axes: dict[str, float],
    mean_L: float = 0.5,
) -> tuple[float, float, float]:
    """Apply multiple named axis shifts sequentially. Returns clamped (L, a, b)."""
    for axis, delta in axes.items():
        if delta == 0:
            continue
        L, a, b = oklab_axis_shift(L, a, b, axis, delta, mean_L=mean_L)
    # Clamp L to [0, 1]; a/b are unclamped (oklab_to_rgb handles gamut mapping)
    L = max(0.0, min(1.0, L))
    return L, a, b


# ── Palette harmony score (Ou et al. 2006) ───────────────────────────────────
# Ported from ColorWay/Mycolor.studio/lib/colorways.jsx — harmonyPalette
# The Ou formula was calibrated for CIELAB scale, so we scale OKLAB first:
#   L_ou = L_oklab × 100,  a_ou = a_oklab × 150,  b_ou = b_oklab × 150

def _harmony_pair(l1: float, a1: float, b1: float,
                  l2: float, a2: float, b2: float) -> float:
    """Ou et al. pairwise harmony score in CIELAB-scale coordinates."""
    dL = l2 - l1
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    dC = C2 - C1
    dE = math.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)
    dH = math.sqrt(max(0.0, dE*dE - dL*dL - dC*dC))
    Lsum = l1 + l2
    return (
        -0.7 * math.tanh(-0.7  + 0.04  * dH)
        - 0.3 * math.tanh(-1.1  + 0.05  * dC)
        + 0.4 * math.tanh(-0.8  + 0.05  * dL)
        + 0.3 + 0.6 * math.tanh(-4.2 + 0.028 * Lsum)
    )


def harmony_score(oklab_colors: list[tuple[float, float, float]]) -> float:
    """
    Ou et al. palette harmony score.
    Input: list of (L, a, b) in OKLAB native scale (L≈0-1, a/b≈±0.4).
    Returns a float; higher is more harmonious (typical range -1 to +1).
    """
    # Scale to CIELAB-like range as ColorWay does
    scaled = [(L * 100, a * 150, b * 150) for L, a, b in oklab_colors]
    n = len(scaled)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _harmony_pair(*scaled[i], *scaled[j])
            count += 1
    return round(total / count, 4) if count > 0 else 0.0
