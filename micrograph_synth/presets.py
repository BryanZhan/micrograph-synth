"""Named specimen presets."""

from __future__ import annotations

import numpy as np

from . import scene as S

PRESETS = {
    "grains": "等轴晶粒（腐蚀金相 / SEM 二次电子像）",
    "spheres-mono": "单分散微球（密排单层）",
    "spheres-poly": "多分散微球（宽粒径分布）",
    "needles": "针状晶体（两端尖锐）",
    "plates": "片状晶体（六方板状、带倾角）",
}


def make_scene(preset: str, rng: np.random.Generator, width: int, height: int, **kw) -> S.Scene:
    if preset == "grains":
        return S.grains(rng, width, height, **kw)
    if preset == "spheres-mono":
        return S.spheres(rng, width, height, mono=True, **kw)
    if preset == "spheres-poly":
        return S.spheres(rng, width, height, mono=False, **kw)
    if preset == "needles":
        return S.needles(rng, width, height, **kw)
    if preset == "plates":
        return S.plates(rng, width, height, **kw)
    raise ValueError(f"unknown preset: {preset} (choose from {', '.join(PRESETS)})")
