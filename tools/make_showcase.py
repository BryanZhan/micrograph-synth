"""Build docs/showcase.png from the generated examples (numpy + Pillow only)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ["grains", "spheres-mono", "spheres-poly", "needles", "plates"]
TITLES = {
    "grains": "等轴晶粒",
    "spheres-mono": "单分散微球",
    "spheres-poly": "多分散微球",
    "needles": "针状晶体",
    "plates": "片状晶体",
}

CELL = 300
GAP = 16
LEFT = 130
TOP = 96
LABEL_H = 44
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> int:
    for preset in PRESETS:
        if not (ROOT / "examples" / preset / f"{preset}_000.png").exists():
            print(f"missing examples/{preset}; run the CLI first", file=sys.stderr)
            return 1

    width = LEFT + len(PRESETS) * CELL + (len(PRESETS) - 1) * GAP + 40
    height = TOP + 2 * (CELL + LABEL_H) + 30
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    draw.text((40, 26), "MicrographSynth — 合成显微图像与像素级真值", fill=(27, 36, 48), font=font(30))
    draw.text((40, 66), "上排：SEM 风格灰度图　下排：实例掩膜叠加（颜色即真值标签）", fill=(91, 107, 122), font=font(19))

    for row, (suffix, label) in enumerate([("", "灰度图"), ("_overlay", "真值掩膜")]):
        y = TOP + row * (CELL + LABEL_H)
        draw.text((40, y + CELL // 2 - 12), label, fill=(27, 36, 48), font=font(24))
        for col, preset in enumerate(PRESETS):
            x = LEFT + col * (CELL + GAP)
            img = Image.open(ROOT / "examples" / preset / f"{preset}_000{suffix}.png").convert("RGB")
            sheet.paste(img.resize((CELL, CELL), Image.BILINEAR), (x, y))
            if row == 0:
                draw.text((x, y + CELL + 10), TITLES[preset], fill=(27, 36, 48), font=font(22))

    out = ROOT / "docs" / "showcase.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if sheet.width > 1200:
        ratio = 1200 / sheet.width
        sheet = sheet.resize((1200, round(sheet.height * ratio)), Image.LANCZOS)
    sheet.save(out, optimize=True)
    print("wrote", out, sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
