"""Stage 1c — Turn raw Florence-2 annotations into clean train/val COCO splits.

Reads data/annotations/annotations.json, drops images with no boxes, then splits into
train/val (default 85/15) ensuring each split has enough of BOTH classes. Writes:
    data/processed/train.json
    data/processed/val.json

Also prints per-class image counts so we can confirm the >=500 train / >=100 val per class
requirement from the brief.

Usage:
    python src/dataset/build_dataset.py --val-frac 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANN = PROJECT_ROOT / "data" / "annotations" / "annotations.json"
OUT = PROJECT_ROOT / "data" / "processed"


def images_per_class(coco) -> dict[int, set]:
    """category_id -> set of image_ids that contain that class."""
    per = defaultdict(set)
    for a in coco["annotations"]:
        per[a["category_id"]].add(a["image_id"])
    return per


def subset(coco, image_ids: set) -> dict:
    image_ids = set(image_ids)
    imgs = [im for im in coco["images"] if im["id"] in image_ids]
    anns = [a for a in coco["annotations"] if a["image_id"] in image_ids]
    return {"images": imgs, "annotations": anns, "categories": coco["categories"]}


def select_per_class(coco, target: int, seed: int) -> set:
    """Greedily pick a MINIMAL set of images so each class has ~`target` images.
    The rare class (vehicle) drives selection; common classes (person) are covered
    along the way. Returns a set of image_ids."""
    img_cats = defaultdict(set)  # image_id -> set of category_ids it contains
    for a in coco["annotations"]:
        img_cats[a["image_id"]].add(a["category_id"])
    cats = [c["id"] for c in coco["categories"]]
    counts = {c: 0 for c in cats}
    ids = [im["id"] for im in coco["images"]]
    random.Random(seed).shuffle(ids)
    chosen = set()
    for img_id in ids:
        cs = img_cats.get(img_id)
        if not cs:
            continue
        if any(counts[c] < target for c in cs):   # this image helps an under-target class
            chosen.add(img_id)
            for c in cs:
                counts[c] += 1
        if all(counts[c] >= target for c in cats):
            break
    return chosen


def report(name, coco):
    per = images_per_class(coco)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    print(f"  {name}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")
    for cid, ids in sorted(per.items()):
        print(f"      {cat_name.get(cid, cid):<8} -> {len(ids)} images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(ANN))
    ap.add_argument("--out-dir", default=str(OUT),
                    help="output dir for train.json/val.json (e.g. data/processed/ensemble)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--per-class", type=int, default=None,
                    help="build a SMALL subset with ~N images per class (fast training). "
                         "N=750 -> ~637 train/113 val per class after the split.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    coco = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    print(f"Loaded {len(coco['images'])} annotated images, "
          f"{len(coco['annotations'])} boxes.")

    if args.per_class:
        keep = select_per_class(coco, args.per_class, args.seed)
        coco = subset(coco, keep)
        print(f"Subset to ~{args.per_class}/class -> {len(coco['images'])} images, "
              f"{len(coco['annotations'])} boxes.")

    img_ids = [im["id"] for im in coco["images"]]
    random.Random(args.seed).shuffle(img_ids)
    n_val = int(len(img_ids) * args.val_frac)
    val_ids, train_ids = set(img_ids[:n_val]), set(img_ids[n_val:])

    train, val = subset(coco, train_ids), subset(coco, val_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (out_dir / "val.json").write_text(json.dumps(val), encoding="utf-8")

    print(f"\nSplit written to {out_dir}:")
    report("train", train)
    report("val", val)
    print("\nRequirement check (>=500 train / >=100 val images per class):")
    tr_per, va_per = images_per_class(train), images_per_class(val)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    for c in coco["categories"]:
        t, v = len(tr_per.get(c["id"], [])), len(va_per.get(c["id"], []))
        ok = "OK" if (t >= 500 and v >= 100) else "SHORT"
        print(f"  {cat_name[c['id']]:<8} train={t:<5} val={v:<5} [{ok}]")


if __name__ == "__main__":
    main()
