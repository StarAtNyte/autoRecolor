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

    # Saturation (0 where achromatic).
    # Guard the denominator to avoid divide-by-zero for l≈0 or l≈1 pixels;
    # those are achromatic anyway so the np.where picks 0.0 regardless.
    denom = np.maximum(1.0 - np.abs(2.0 * l - 1.0), 1e-10)
    s = np.where(d == 0, 0.0, d / denom)

    # Hue
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

    # Sector index 0-5
    hi = np.floor(h6).astype(np.int32) % 6
    z = np.zeros_like(c)

    # (r, g, b) coefficients per sector
    # 0:(c,x,0)  1:(x,c,0)  2:(0,c,x)  3:(0,x,c)  4:(x,0,c)  5:(c,0,x)
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
