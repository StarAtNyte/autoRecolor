from __future__ import annotations
import math
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ColorEntry:
    id: int
    label: str
    hex: str
    rgb: list[int]
    hsl: list[float]
    pixel_percent: float
    oklab: list[float] = field(default_factory=list)

    def __post_init__(self):
        """Auto-compute OKLAB from RGB if not provided (e.g. when loading from JSON)."""
        if not self.oklab and self.rgb:
            self._sync_oklab()

    # ── Internal sync helpers ─────────────────────────────────────────────────

    def _sync_oklab(self) -> None:
        """Recompute stored OKLAB from current RGB."""
        import numpy as np
        from autoRecolor.utils import rgb_to_oklab_batch
        arr = rgb_to_oklab_batch(np.array([self.rgb], dtype=np.uint8))[0]
        self.oklab = [round(float(arr[0]), 5), round(float(arr[1]), 5), round(float(arr[2]), 5)]

    def _sync_from_oklab(self, L: float, a: float, b: float) -> None:
        """Apply new OKLAB values → update hex, rgb, hsl, and oklab fields."""
        import numpy as np
        from autoRecolor.utils import oklab_to_rgb_batch, rgb_to_hsl, rgb_to_hex
        arr = oklab_to_rgb_batch(np.array([[L, a, b]], dtype=np.float64))[0]
        r, g, b_val = int(arr[0]), int(arr[1]), int(arr[2])
        self.rgb = [r, g, b_val]
        self.hex = rgb_to_hex(r, g, b_val)
        self.hsl = rgb_to_hsl(r, g, b_val)
        self.oklab = [round(float(L), 5), round(float(a), 5), round(float(b), 5)]

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def chroma(self) -> float:
        """Colorfulness: distance from the achromatic axis in the ab-plane (0=gray, ~0.4=vivid)."""
        if len(self.oklab) < 3:
            return 0.0
        return round(math.hypot(self.oklab[1], self.oklab[2]), 5)

    @property
    def hue_angle(self) -> float:
        """Hue direction in the ab-plane in degrees (0=red, 90=yellow, 180=green/cyan, 270=blue)."""
        if len(self.oklab) < 3:
            return 0.0
        angle = math.degrees(math.atan2(self.oklab[2], self.oklab[1]))
        return round(angle % 360, 1)

    def to_info_dict(self) -> dict:
        """Full info dict including derived fields — used by get_palette tool."""
        d = asdict(self)
        d["chroma"] = self.chroma
        d["hue_angle"] = self.hue_angle
        return d


@dataclass
class Palette:
    image_path: str
    image_width: int
    image_height: int
    palette: list[ColorEntry] = field(default_factory=list)
    modifications: list[dict] = field(default_factory=list)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_index(self) -> dict[int, ColorEntry]:
        return {c.id: c for c in self.palette}

    def get_entry(self, color_id: int) -> Optional[ColorEntry]:
        return self._build_index().get(color_id)

    def _mean_L(self) -> float:
        """Mean lightness across all palette colors (used by contrast axis)."""
        if not self.palette:
            return 0.5
        return sum(c.oklab[0] for c in self.palette if c.oklab) / len(self.palette)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "palette": [c.to_info_dict() for c in self.palette],
            "modifications": self.modifications,
            "harmony_score": self.harmony_score(),
        }

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> Palette:
        p = cls(
            image_path=d["image_path"],
            image_width=d["image_width"],
            image_height=d["image_height"],
            modifications=d.get("modifications", []),
        )
        p.palette = [ColorEntry(**{k: v for k, v in c.items()
                                   if k in ColorEntry.__dataclass_fields__})
                     for c in d["palette"]]
        return p

    # ── Harmony scoring ───────────────────────────────────────────────────────

    def harmony_score(self) -> float:
        """Ou et al. palette harmony score (higher = more harmonious, typical range -1 to +1)."""
        from autoRecolor.utils import harmony_score as _hs
        labs = [tuple(c.oklab) for c in self.palette if len(c.oklab) == 3]
        return _hs(labs)

    # ── Mutations — single color ──────────────────────────────────────────────

    def update_hex(self, color_id: int, new_hex: str) -> None:
        """Set a color to an exact hex value."""
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        from autoRecolor.utils import hex_to_rgb, rgb_to_hsl, hex_to_oklab
        new_hex = "#" + new_hex.lstrip("#").upper()
        entry.hex = new_hex
        entry.rgb = hex_to_rgb(new_hex)
        entry.hsl = rgb_to_hsl(*entry.rgb)
        L, a, b = hex_to_oklab(new_hex)
        entry.oklab = [round(L, 5), round(a, 5), round(b, 5)]
        self.modifications.append({"color_id": color_id, "action": "set_hex", "value": new_hex})

    def adjust_oklab_axes(self, color_id: int, axes: dict[str, float]) -> None:
        """
        Adjust a single color using named perceptual OKLAB axes.

        axes dict keys and their semantics:
          lightness   — additive L shift  (useful range -0.5 to +0.5)
          warmth      — along blackbody locus  (useful range -0.2 to +0.2; + = warmer amber)
          chroma      — radial ab push, hue angle unchanged  (-0.3 to +0.3; + = more vivid)
          hue         — ab-plane rotation in degrees  (-180 to +180)
          exposure    — multiplicative L in photographic stops  (-2 to +2; +1 = 2× brighter)
          saturation  — proportional chroma scale  (-1.0 to +1.0; +0.5 = 50% more saturated)
          purity      — jewel-tone axis  (-0.5 to +0.5; + = darker+more chromatic)
          mute        — pull toward mid-gray  (-0.5 to +0.5; + = more neutral/corporate)
          green_red   — direct a-channel shift  (-0.2 to +0.2; + = redder)
          blue_yellow — direct b-channel shift  (-0.2 to +0.2; + = yellower)
        """
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        from autoRecolor.utils import apply_oklab_axes
        L, a, b = entry.oklab[0], entry.oklab[1], entry.oklab[2]
        L, a, b = apply_oklab_axes(L, a, b, axes, mean_L=self._mean_L())
        entry._sync_from_oklab(L, a, b)
        self.modifications.append({"color_id": color_id, "action": "adjust_oklab_axes", "value": axes})

    def set_lightness(self, color_id: int, L: float) -> None:
        """Set the absolute lightness of a color (0.0 = black, 1.0 = white)."""
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        L = max(0.0, min(1.0, L))
        a, b = entry.oklab[1], entry.oklab[2]
        entry._sync_from_oklab(L, a, b)
        self.modifications.append({"color_id": color_id, "action": "set_lightness", "value": L})

    def set_chroma(self, color_id: int, C: float) -> None:
        """
        Set the absolute chroma (colorfulness) of a color.
        C = 0 → achromatic gray at the same lightness.
        C ≈ 0.2 → moderately saturated, C ≈ 0.4 → very vivid.
        Hue angle is preserved.
        """
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        C = max(0.0, C)
        L = entry.oklab[0]
        a0, b0 = entry.oklab[1], entry.oklab[2]
        cur_C = math.hypot(a0, b0)
        if cur_C < 1e-10:
            # Achromatic — arbitrary direction (toward red)
            a_new, b_new = C, 0.0
        else:
            scale = C / cur_C
            a_new, b_new = a0 * scale, b0 * scale
        entry._sync_from_oklab(L, a_new, b_new)
        self.modifications.append({"color_id": color_id, "action": "set_chroma", "value": C})

    def set_hue(self, color_id: int, target_hue: float) -> None:
        """Set the absolute hue angle of a color (0–360°), preserving L and C."""
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        L = entry.oklab[0]
        C = entry.chroma
        if C < 1e-10:
            C = 0.05  # give achromatic colors a small chroma so hue takes effect
        h_rad = target_hue * math.pi / 180.0
        a_new = C * math.cos(h_rad)
        b_new = C * math.sin(h_rad)
        entry._sync_from_oklab(L, a_new, b_new)
        self.modifications.append({"color_id": color_id, "action": "set_hue", "value": target_hue})

    def match_lightness(self, source_id: int, target_id: int) -> None:
        """Copy the lightness (L) of source color onto target, keeping target's hue/chroma."""
        src = self.get_entry(source_id)
        tgt = self.get_entry(target_id)
        if src is None:
            raise ValueError(f"Source color id {source_id} not found")
        if tgt is None:
            raise ValueError(f"Target color id {target_id} not found")
        L = src.oklab[0]
        a, b = tgt.oklab[1], tgt.oklab[2]
        tgt._sync_from_oklab(L, a, b)
        self.modifications.append({
            "action": "match_lightness",
            "source_id": source_id, "target_id": target_id,
        })

    def relabel(self, color_id: int, new_label: str) -> None:
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        entry.label = new_label
        self.modifications.append({"color_id": color_id, "action": "relabel", "value": new_label})

    # ── Mutations — palette-wide ───────────────────────────────────────────────

    def adjust_palette_axes(
        self,
        axes: dict[str, float],
        exclude_ids: list[int] | None = None,
    ) -> None:
        """
        Apply named OKLAB axis shifts to all colors (except those in exclude_ids).
        Supports all per-color axes PLUS:
          contrast — expand/compress all L values around the palette mean L
                     (useful range -0.5 to +1.0; + = more contrast)
        """
        from autoRecolor.utils import apply_oklab_axes
        exclude = set(exclude_ids or [])
        mean_L = self._mean_L()
        for entry in self.palette:
            if entry.id in exclude:
                continue
            L, a, b = entry.oklab[0], entry.oklab[1], entry.oklab[2]
            L, a, b = apply_oklab_axes(L, a, b, axes, mean_L=mean_L)
            entry._sync_from_oklab(L, a, b)
        self.modifications.append({
            "action": "adjust_palette_axes",
            "axes": axes,
            "exclude_ids": list(exclude),
        })

    # ── Legacy HSL adjust (kept for backward compatibility) ───────────────────

    def adjust_hsl(self, color_id: int, h_shift: float, s_shift: float, l_shift: float) -> None:
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        from autoRecolor.utils import hsl_to_rgb, rgb_to_hex
        h, s, l = entry.hsl
        new_h = (h + h_shift) % 360
        new_s = max(0.0, min(100.0, s + s_shift))
        new_l = max(0.0, min(100.0, l + l_shift))
        entry.hsl = [new_h, new_s, new_l]
        r, g, b = hsl_to_rgb(new_h, new_s, new_l)
        entry.rgb = [r, g, b]
        entry.hex = rgb_to_hex(r, g, b)
        entry._sync_oklab()
        self.modifications.append({
            "color_id": color_id,
            "action": "adjust_hsl",
            "value": {"h_shift": h_shift, "s_shift": s_shift, "l_shift": l_shift},
        })
