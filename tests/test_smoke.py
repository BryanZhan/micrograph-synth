"""Smoke tests: determinism, shapes, and measurement sanity."""

import numpy as np

from micrograph_synth import PRESETS, make_scene, measure, render


def test_all_presets_are_deterministic():
    for preset in PRESETS:
        a = make_scene(preset, np.random.default_rng(7), 160, 128)
        b = make_scene(preset, np.random.default_rng(7), 160, 128)
        assert a.ids.shape == (128, 160)
        assert np.array_equal(a.ids, b.ids), f"{preset} is not deterministic"
        assert a.n_instances > 0


def test_render_and_measure():
    for preset in PRESETS:
        scene = make_scene(preset, np.random.default_rng(3), 192, 160)
        out = render(scene, np.random.default_rng(4), pits=8)
        assert out["gray"].dtype == np.uint8
        assert out["overlay"].shape == (160, 192, 3)
        stats = measure(out["ids"], pixel_size_um=0.5)
        assert stats["count"] >= 1
        assert 0.0 < stats["area_fraction"] <= 1.0
        assert stats["d50_um"] > 0.0
