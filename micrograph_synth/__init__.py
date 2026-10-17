"""MicrographSynth — synthetic micrographs with pixel-perfect ground truth."""

from .presets import PRESETS, make_scene
from .render import render
from .stats import measure

__all__ = ["PRESETS", "make_scene", "render", "measure"]
__version__ = "0.1.0"
