"""Turn a :class:`~micrograph_synth.scene.Scene` into SEM-like images.

The mask is the authoritative ground truth; the grey image is a rendering of it,
with secondary-electron style edge brightening, etched boundary grooves, pores
and detector noise.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .scene import Scene, fbm

MASK_COLORS = [
    (86, 208, 226), (242, 166, 90), (127, 179, 240), (240, 127, 168),
    (154, 224, 107), (199, 155, 242), (242, 193, 78), (104, 224, 176),
    (236, 122, 122), (126, 210, 240), (200, 224, 88), (176, 148, 232),
]


def _edge_mask(ids: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(ids, bool)
    edge[1:, :] |= ids[1:, :] != ids[:-1, :]
    edge[:-1, :] |= ids[:-1, :] != ids[1:, :]
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    return edge


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _blur3(a: np.ndarray) -> np.ndarray:
    return (
        a * 4
        + np.roll(a, 1, 0) + np.roll(a, -1, 0)
        + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    ) / 8.0


def render(scene: Scene, rng: np.random.Generator, noise: float = 0.05, pits: int = 90) -> dict:
    """Render ``scene`` and return ``{"gray": uint8, "ids": int32, "overlay": uint8}``."""
    h, w = scene.ids.shape
    ids = scene.ids
    mask = ids >= 0

    brightness = np.array([i.brightness for i in scene.instances] + [0.2], np.float32)
    b = brightness[np.where(mask, ids, len(scene.instances))]
    dn = np.clip(scene.dnorm, 0.0, 1.0)
    glow = np.clip(1.0 - dn / 0.06, 0.0, 1.0)

    if scene.kind == "spheres":
        nz = np.sqrt(np.clip(1.0 - (1.0 - dn) ** 2, 0.0, 1.0))
        v = b * (0.42 + 0.72 * nz) + 0.26 * glow
    else:
        v = b * (0.82 + 0.30 * np.clip(dn / 0.25, 0.0, 1.0)) + 0.28 * glow

    detail = 0.10 if scene.kind == "grains" else 0.05
    v += (fbm(rng, h, w, cells=12, octaves=3) - 0.5) * detail

    # per-instance texture amplitude: neighbouring grains shouldn't look identical
    if scene.n_instances:
        amp = (rng.random(scene.n_instances).astype(np.float32) * (0.10 if scene.kind == "grains" else 0.05)).astype(np.float32)
        amp_field = amp[np.where(mask, ids, 0)]
        v += (fbm(rng, h, w, cells=22, octaves=2) - 0.5) * amp_field

    substrate = 0.22 + 0.10 * (fbm(rng, h, w, cells=6, octaves=3) - 0.5)
    v = np.where(mask, v, substrate)

    edge = _edge_mask(ids)
    shoulder = _dilate(edge)
    v = np.where(shoulder & ~edge, v - 0.05, v)
    v = np.where(edge, v - 0.16, v)

    # pores and etch pits (drawn inside their own bounding box)
    def _spot(cx: float, cy: float, r: float, depth: float) -> None:
        x0, x1 = max(0, int(cx - r - 1)), min(w, int(cx + r + 2))
        y0, y1 = max(0, int(cy - r - 1)), min(h, int(cy + r + 2))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        v[y0:y1, x0:x1] -= depth * np.clip(1.0 - d / r, 0.0, 1.0)

    for _ in range(max(1, pits // 30)):
        _spot(rng.random() * w, rng.random() * h, 12 + rng.random() * 18, 0.22)
    for _ in range(pits):
        _spot(rng.random() * w, rng.random() * h, 1.2 + rng.random() * 2.0, 0.14)

    v = _blur3(v)

    yy, xx = np.mgrid[0:h, 0:w]
    vx = (xx / w - 0.5) * 2
    vy = (yy / h - 0.5) * 2
    vig = 1.0 - 0.15 * (vx * vx + vy * vy)
    v = v * vig + (rng.random((h, w)).astype(np.float32) - 0.5) * noise
    v = np.clip(v, 0.0, 1.0)
    v = np.clip((v - 0.10) / 0.84, 0.0, 1.0)
    v = v * v * (3 - 2 * v) * 0.92 + 0.05
    gray = np.round(v * 255).astype(np.uint8)

    overlay = np.zeros((h, w, 3), np.uint8)
    palette = np.array(MASK_COLORS, np.uint8)
    safe = np.where(mask, ids % len(MASK_COLORS), 0)
    base = gray[..., None].astype(np.float32) * 0.55
    color = base + (palette[safe].astype(np.float32) - base) * 0.52
    color[~mask] *= 0.55
    color[edge] = 250
    overlay = np.clip(color, 0, 255).astype(np.uint8)

    return {"gray": gray, "ids": ids, "overlay": overlay}


def gray_image(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray, mode="L")


def mask_image(ids: np.ndarray) -> Image.Image:
    """Palette PNG: index 0 = background, index ``k+1`` = instance ``k``."""
    h, w = ids.shape
    idx = np.where(ids >= 0, ids % 255 + 1, 0).astype(np.uint8)
    img = Image.fromarray(idx, mode="P")
    pal = [0, 0, 0]
    for c in MASK_COLORS:
        pal.extend(c)
    while len(pal) < 768:
        pal.extend((40, 40, 40))
    img.putpalette(pal[:768])
    return img


def overlay_image(overlay: np.ndarray) -> Image.Image:
    return Image.fromarray(overlay, mode="RGB")
