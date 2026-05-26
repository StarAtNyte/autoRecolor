from __future__ import annotations

from autoRecolor.palette import Palette, ColorEntry
from autoRecolor.analyze import analyze_image
from autoRecolor.recolor import recolor_image
from autoRecolor.agent import Agent
from autoRecolor.cli import main

__all__ = [
    "Palette",
    "ColorEntry",
    "analyze_image",
    "recolor_image",
    "Agent",
    "main",
]
