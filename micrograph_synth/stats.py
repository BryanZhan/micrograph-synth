"""Per-instance measurements computed straight from the ground-truth labels."""

from __future__ import annotations

import numpy as np


def measure(ids: np.ndarray, pixel_size_um: float = 1.0) -> dict:
    """Return per-instance geometry plus whole-field summary statistics."""
    labels = np.unique(ids)
    labels = labels[labels >= 0]
    rows = []
    h, w = ids.shape
    total = float(h * w)
    for lab in labels:
        m = ids == lab
        area = int(m.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(m)
        cy, cx = float(ys.mean()), float(xs.mean())
        dy, dx = ys - cy, xs - cx
        cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
        evals, evecs = np.linalg.eigh(cov)
        major = float(np.sqrt(max(evals[-1], 0.0)) * 4)
        minor = float(np.sqrt(max(evals[0], 1e-9)) * 4)
        rows.append(
            {
                "label": int(lab),
                "area_px": area,
                "area_frac": area / total,
                "eq_diameter_px": float(2 * np.sqrt(area / np.pi)),
                "eq_diameter_um": float(2 * np.sqrt(area / np.pi) * pixel_size_um),
                "major_px": major,
                "minor_px": minor,
                "aspect_ratio": float(major / max(minor, 1e-6)),
                "orientation_deg": float(np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1])) % 180.0),
                "centroid_x": cx,
                "centroid_y": cy,
            }
        )
    d = np.array([r["eq_diameter_px"] for r in rows], float) if rows else np.array([], float)
    return {
        "instances": rows,
        "count": len(rows),
        "area_fraction": float(sum(r["area_px"] for r in rows) / total),
        "mean_eq_diameter_um": float(d.mean() * pixel_size_um) if len(d) else 0.0,
        "std_eq_diameter_um": float(d.std() * pixel_size_um) if len(d) else 0.0,
        "d10_um": float(np.percentile(d, 10) * pixel_size_um) if len(d) else 0.0,
        "d50_um": float(np.percentile(d, 50) * pixel_size_um) if len(d) else 0.0,
        "d90_um": float(np.percentile(d, 90) * pixel_size_um) if len(d) else 0.0,
    }
