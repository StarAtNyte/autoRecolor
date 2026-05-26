from __future__ import annotations
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


@dataclass
class Palette:
    image_path: str
    image_width: int
    image_height: int
    palette: list[ColorEntry] = field(default_factory=list)
    modifications: list[dict] = field(default_factory=list)

    # ── Internal id→entry index (kept in sync by all mutating methods) ───────
    def _build_index(self) -> dict[int, ColorEntry]:
        return {c.id: c for c in self.palette}

    def get_entry(self, color_id: int) -> Optional[ColorEntry]:
        return self._build_index().get(color_id)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "palette": [asdict(c) for c in self.palette],
            "modifications": self.modifications,
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
        p.palette = [ColorEntry(**c) for c in d["palette"]]
        return p

    # ── Mutations ─────────────────────────────────────────────────────────────

    def update_hex(self, color_id: int, new_hex: str) -> None:
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        from autoRecolor.utils import hex_to_rgb, rgb_to_hsl
        # Normalise to uppercase
        new_hex = "#" + new_hex.lstrip("#").upper()
        entry.hex = new_hex
        entry.rgb = hex_to_rgb(new_hex)
        entry.hsl = rgb_to_hsl(*entry.rgb)
        self.modifications.append({
            "color_id": color_id,
            "action": "set_hex",
            "value": new_hex,
        })

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
        self.modifications.append({
            "color_id": color_id,
            "action": "adjust_hsl",
            "value": {"h_shift": h_shift, "s_shift": s_shift, "l_shift": l_shift},
        })

    def relabel(self, color_id: int, new_label: str) -> None:
        entry = self.get_entry(color_id)
        if entry is None:
            raise ValueError(f"Color id {color_id} not found")
        entry.label = new_label
        self.modifications.append({
            "color_id": color_id,
            "action": "relabel",
            "value": new_label,
        })
