"""Instance geometry for synthetic micrographs.

Every scene is an integer label map (``ids``) plus a normalised distance-to-edge
map (``dnorm``).  Instances are drawn in painter order, exactly like a real
sample where later objects sit on top of earlier ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instance:
    idx: int
    kind: str  # grain | sphere | needle | plate
    cx: float
    cy: float
    size: float  # characteristic size (radius / half-length)
    brightness: float
    angle: float = 0.0
    extent: float = 1.0  # major/minor ratio for needles and plates


@dataclass
class Scene:
    width: int
    height: int
    ids: np.ndarray
    dnorm: np.ndarray
    instances: list[Instance] = field(default_factory=list)
    kind: str = "grains"

    @property
    def n_instances(self) -> int:
        return len(self.instances)


def fbm(rng: np.random.Generator, height: int, width: int, cells: int = 4, octaves: int = 4) -> np.ndarray:
    """Vectorised value-noise fBm in ``[0, 1]``."""
    out = np.zeros((height, width), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        c = cells * 2**o
        grid = rng.random((c + 1, c + 1)).astype(np.float32)
        ys = np.linspace(0, c, height, endpoint=False)
        xs = np.linspace(0, c, width, endpoint=False)
        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        sy = fy * fy * (3 - 2 * fy)
        sx = fx * fx * (3 - 2 * fx)
        a = grid[np.ix_(y0, x0)]
        b = grid[np.ix_(y0, x0 + 1)]
        c1 = grid[np.ix_(y0 + 1, x0)]
        d = grid[np.ix_(y0 + 1, x0 + 1)]
        top = a * (1 - sx) + b * sx
        bot = c1 * (1 - sx) + d * sx
        out += amp * (top * (1 - sy) + bot * sy)
        norm += amp
        amp *= 0.5
    return out / norm


def _new_scene(width: int, height: int, kind: str) -> Scene:
    return Scene(
        width=width,
        height=height,
        ids=np.full((height, width), -1, np.int32),
        dnorm=np.zeros((height, width), np.float32),
        kind=kind,
    )


def _stamp_disc(scene: Scene, cx: float, cy: float, r: float, idx: int) -> None:
    h, w = scene.ids.shape
    x0, x1 = max(0, int(cx - r - 2)), min(w, int(cx + r + 3))
    y0, y1 = max(0, int(cy - r - 2)), min(h, int(cy + r + 3))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    d = np.hypot(xx - cx, yy - cy)
    m = d <= r
    sub_ids = scene.ids[y0:y1, x0:x1]
    sub_dn = scene.dnorm[y0:y1, x0:x1]
    sub_ids[m] = idx
    sub_dn[m] = (r - d[m]) / max(r, 1e-6)


def _dist_to_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    len2 = max(vx * vx + vy * vy, 1e-9)
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / len2, 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _stamp_poly(scene: Scene, pts: np.ndarray, idx: int, size_hint: float) -> None:
    pts = np.asarray(pts, np.float32)
    area2 = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area2 < 0:
        pts = pts[::-1]
    h, w = scene.ids.shape
    x0, x1 = max(0, int(pts[:, 0].min()) - 2), min(w, int(pts[:, 0].max()) + 3)
    y0, y1 = max(0, int(pts[:, 1].min()) - 2), min(h, int(pts[:, 1].max()) + 3)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    inside = np.ones(xx.shape, bool)
    for i in range(len(pts)):
        ax, ay = pts[i - 1]
        bx, by = pts[i]
        inside &= (bx - ax) * (yy - ay) - (by - ay) * (xx - ax) >= 0
    if not inside.any():
        return
    dist = np.full(xx.shape, np.inf, np.float32)
    for i in range(len(pts)):
        ax, ay = pts[i - 1]
        bx, by = pts[i]
        dist = np.minimum(dist, _dist_to_segment(xx, yy, ax, ay, bx, by))
    sub_ids = scene.ids[y0:y1, x0:x1]
    sub_dn = scene.dnorm[y0:y1, x0:x1]
    sub_ids[inside] = idx
    sub_dn[inside] = np.clip(dist[inside] / max(size_hint, 1e-6), 0.0, 1.0)


def grains(rng, width, height, cols=9, rows=7, jitter=0.4, warp=0.9) -> Scene:
    """Equiaxed grains: warped Voronoi tessellation."""
    scene = _new_scene(width, height, "grains")
    cw, ch = width / cols, height / rows
    seeds = []
    for r in range(rows):
        for c in range(cols):
            seeds.append(
                (
                    cw * (c + 0.5) + (rng.random() - 0.5) * cw * jitter,
                    ch * (r + 0.5) + (rng.random() - 0.5) * ch * jitter,
                )
            )
    seeds = np.asarray(seeds, np.float32)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    jitter_field = (fbm(rng, height, width, cells=4, octaves=3) - 0.5) * warp * (min(cw, ch) ** 2) * 1.6

    best = np.full((height, width), np.inf, np.float32)
    ids = np.full((height, width), -1, np.int32)
    for i, (sx, sy) in enumerate(seeds):
        d = (xs - sx) ** 2 + (ys - sy) ** 2 + jitter_field
        m = d < best
        best[m] = d[m]
        ids[m] = i

    # distance to the nearest different label, normalised by grain size
    edge = np.zeros_like(ids, bool)
    edge[1:, :] |= ids[1:, :] != ids[:-1, :]
    edge[:-1, :] |= ids[:-1, :] != ids[1:, :]
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]

    dist = np.where(edge, 0.0, np.inf).astype(np.float32)
    # cheap distance transform by repeated dilation
    step = 1.0
    growing = edge.copy()
    for _ in range(6):
        grown = growing.copy()
        grown[1:, :] |= growing[:-1, :]
        grown[:-1, :] |= growing[1:, :]
        grown[:, 1:] |= growing[:, :-1]
        grown[:, :-1] |= growing[:, 1:]
        newly = grown & ~growing
        dist[newly] = step
        growing = grown
        step += 1.0
    scale = float(min(cw, ch)) * 0.5
    dnorm = np.clip(dist / scale, 0.0, 1.0)

    scene.ids = ids
    scene.dnorm = dnorm
    scene.instances = [
        Instance(i, "grain", float(sx), float(sy), float(min(cw, ch) * 0.5), float(0.36 + rng.random() * 0.4))
        for i, (sx, sy) in enumerate(seeds)
    ]
    return scene


def spheres(rng, width, height, mono=True, count=None, r_min=14.0, r_max=64.0) -> Scene:
    """Microspheres: one tightly packed size, or a broad size distribution."""
    scene = _new_scene(width, height, "spheres")
    if mono:
        r = min(width, height) * 0.085
        step = r * 2.05
        idx = 0
        row = 0
        while row * step * 0.87 - r < height:
            cy = row * step * 0.87
            col = 0
            while col * step - r < width:
                cx = col * step + (step / 2 if row % 2 else 0)
                cx += (rng.random() - 0.5) * step * 0.12
                cy2 = cy + (rng.random() - 0.5) * step * 0.12
                rr = r * (0.96 + rng.random() * 0.08)
                _stamp_disc(scene, cx, cy2, rr, idx)
                scene.instances.append(
                    Instance(idx, "sphere", float(cx), float(cy2), float(rr), float(0.38 + rng.random() * 0.22))
                )
                idx += 1
                col += 1
            row += 1
    else:
        n = count or 78
        for i in range(n):
            t = rng.random()
            rr = r_min + (t**2) * (r_max - r_min)
            cx = rng.random() * (width + 60) - 30
            cy = rng.random() * (height + 60) - 30
            _stamp_disc(scene, cx, cy, rr, i)
            scene.instances.append(
                Instance(i, "sphere", float(cx), float(cy), float(rr), float(0.36 + rng.random() * 0.28))
            )
    return scene


def needles(rng, width, height, count=70, length=(0.18, 0.42), width_range=(0.008, 0.02)) -> Scene:
    """Acicular crystals: lens-shaped, sharp at both ends."""
    scene = _new_scene(width, height, "needles")
    for i in range(count):
        ang = (rng.random() - 0.5) * 1.25 + (0.35 if rng.random() < 0.5 else -0.35)
        length_px = (length[0] + rng.random() * (length[1] - length[0])) * max(width, height)
        half_w = (width_range[0] + rng.random() * (width_range[1] - width_range[0])) * max(width, height)
        cx = rng.random() * (width + 120) - 60
        cy = rng.random() * (height + 120) - 60
        ux, uy = np.cos(ang), np.sin(ang)
        nx, ny = -uy, ux
        ax, ay = cx - ux * length_px / 2, cy - uy * length_px / 2
        bx, by = cx + ux * length_px / 2, cy + uy * length_px / 2
        pts = np.array(
            [[ax, ay], [cx + nx * half_w, cy + ny * half_w], [bx, by], [cx - nx * half_w, cy - ny * half_w]],
            np.float32,
        )
        _stamp_poly(scene, pts, i, half_w)
        scene.instances.append(
            Instance(i, "needle", float(cx), float(cy), float(half_w), float(0.4 + rng.random() * 0.28), float(ang), float(length_px / (2 * half_w)))
        )
    return scene


def plates(rng, width, height, count=42, size=(0.05, 0.14)) -> Scene:
    """Plate-like crystals: hexagonal plates, randomly tilted."""
    scene = _new_scene(width, height, "plates")
    for i in range(count):
        radius = (size[0] + rng.random() * (size[1] - size[0])) * max(width, height)
        rot = rng.random() * np.pi * 2
        squash = 0.38 + rng.random() * 0.62
        cx = rng.random() * (width + 80) - 40
        cy = rng.random() * (height + 80) - 40
        pts = []
        for k in range(6):
            a = k / 6 * np.pi * 2 + (rng.random() - 0.5) * 0.16
            r = radius * (0.82 + rng.random() * 0.36)
            lx, ly = np.cos(a) * r, np.sin(a) * r * squash
            pts.append(
                [
                    cx + lx * np.cos(rot) - ly * np.sin(rot),
                    cy + lx * np.sin(rot) + ly * np.cos(rot),
                ]
            )
        _stamp_poly(scene, np.array(pts, np.float32), i, radius * 0.5)
        scene.instances.append(
            Instance(i, "plate", float(cx), float(cy), float(radius * 0.5), float(0.42 + rng.random() * 0.26), float(rot), float(1.0 / max(squash, 0.05)))
        )
    return scene
