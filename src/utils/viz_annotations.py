"""Visualize auto-generated annotations: draw boxes from a COCO json onto the images.

Great for (a) sanity-checking Florence-2's box quality and (b) slide material.

    python src/utils/viz_annotations.py --ann data/processed/train.json --n 12
    python src/utils/viz_annotations.py --ann data/annotations/annotations.json --n 9
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import CLASSES  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "figures"
COLORS = {0: (220, 40, 40), 1: (40, 120, 220)}  # person=red, vehicle=blue (category_id space)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(PROJECT_ROOT / "data" / "processed" / "train.json"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "annotation_samples.png"))
    args = ap.parse_args()

    coco = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    by_img = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        if a["image_id"] in by_img:
            by_img[a["image_id"]].append(a)
    imgs = [im for im in coco["images"] if by_img[im["id"]]]
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[: args.n]
    if not imgs:
        raise SystemExit(f"No annotated images in {args.ann}")

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()

    # build a grid montage
    cell = 360
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))

    for i, info in enumerate(imgs):
        img = Image.open(info["file_name"]).convert("RGB")
        d = ImageDraw.Draw(img)
        for a in by_img[info["id"]]:
            x, y, w, h = a["bbox"]
            c = COLORS.get(a["category_id"], (0, 200, 0))
            d.rectangle([x, y, x + w, y + h], outline=c, width=3)
            d.text((x + 2, max(0, y - 20)), CLASSES[a["category_id"]], fill=c, font=font)
        img.thumbnail((cell, cell))
        gx, gy = (i % cols) * cell, (i // cols) * cell
        grid.paste(img, (gx, gy))

    OUT.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"Saved {len(imgs)}-image montage -> {args.out}")


if __name__ == "__main__":
    main()
