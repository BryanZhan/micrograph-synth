"""Command line entry point: generate a small dataset in one call."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .presets import PRESETS, make_scene
from .render import gray_image, mask_image, overlay_image, render
from .stats import measure


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="micrograph-synth",
        description="Generate synthetic micrographs with pixel-perfect instance masks.",
    )
    p.add_argument("--preset", default="grains", choices=sorted(PRESETS), help="specimen type")
    p.add_argument("--count", type=int, default=8, help="number of images")
    p.add_argument("--size", type=int, default=512, help="image edge length in pixels")
    p.add_argument("--seed", type=int, default=0, help="base random seed")
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--pixel-size-um", type=float, default=0.2, help="physical size of one pixel")
    p.add_argument("--noise", type=float, default=0.05, help="detector noise amplitude")
    p.add_argument("--pits", type=int, default=90, help="etch pits / speckles per image")
    p.add_argument("--thumb", type=int, default=256, help="preview thumbnail edge in pixels")
    return p


def _contact_sheet(tiles, cols: int, thumb: int) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    pad = 10
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + pad + 18) + pad), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(tiles):
        c, r = i % cols, i // cols
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + pad + 18)
        sheet.paste(img.resize((thumb, thumb), Image.BILINEAR), (x, y))
        draw.text((x, y + thumb + 4), name, fill=(40, 40, 40))
    return sheet


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    tiles = []
    for i in range(args.count):
        rng = np.random.default_rng(args.seed + i)
        scene = make_scene(args.preset, rng, args.size, args.size)
        result = render(scene, rng, noise=args.noise, pits=args.pits)

        stem = f"{args.preset}_{i:03d}"
        gray_image(result["gray"]).save(out / f"{stem}.png")
        mask_image(result["ids"]).save(out / f"{stem}_mask.png")
        overlay_image(result["overlay"]).save(out / f"{stem}_overlay.png")

        stats = measure(result["ids"], pixel_size_um=args.pixel_size_um)
        for row in stats["instances"]:
            rows.append({"sample": stem, **row})
        tiles.append(
            (
                f"{stem}  n={stats['count']}  d50={stats['d50_um']:.2f}µm",
                overlay_image(result["overlay"]),
            )
        )
        print(
            f"[{i + 1}/{args.count}] {stem}: {stats['count']} instances, "
            f"area {stats['area_fraction'] * 100:.1f}%, d50 {stats['d50_um']:.2f} µm"
        )

    if rows:
        with open(out / "instances.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    cols = min(4, max(1, args.count))
    sheet = _contact_sheet(tiles, cols, args.thumb)
    sheet.save(out / "preview.png")

    meta = {
        "preset": args.preset,
        "description": PRESETS[args.preset],
        "count": args.count,
        "size": args.size,
        "seed": args.seed,
        "pixel_size_um": args.pixel_size_um,
        "files_per_sample": ["{stem}.png", "{stem}_mask.png", "{stem}_overlay.png"],
        "notes": "Mask index 0 is background; instance i uses palette index i % 255 + 1.",
    }
    (out / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done -> {out}  (preview: {out / 'preview.png'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
