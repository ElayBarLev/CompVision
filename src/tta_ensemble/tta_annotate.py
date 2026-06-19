"""BONUS — TTA-based annotation quality check / cleanup.

Idea (see docs/05): run Florence-2 on several transformed views of each image (original,
horizontal flip, up/down scale), map every box back to original coordinates, then fuse with
Weighted Boxes Fusion (WBF). Boxes that appear in MANY views are trustworthy; boxes from a
single view are likely spurious. We keep boxes whose view-agreement >= --min-agreement.

Output: a cleaned COCO json + a per-box "agreement" score, so we can compare a model trained
on raw labels vs. TTA-cleaned labels.

    python src/tta_ensemble/tta_annotate.py --limit 500 --min-agreement 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.annotate import (detect, iter_images, load_florence,  # noqa: E402
                                  map_label, passes_filters, resolve_images_root)
from src.utils.config import CLASS_TO_ID, CLASSES                       # noqa: E402

ANN_DIR = PROJECT_ROOT / "data" / "annotations"


def views(image: Image.Image):
    """Yield (name, transformed_image, untransform_fn). Boxes are normalized [0,1]."""
    w, h = image.width, image.height
    yield "orig", image, lambda b: b
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    # flip x: for normalized xyxy, x' = 1 - x (and swap x1,x2)
    yield "flip", flipped, lambda b: [1 - b[2], b[1], 1 - b[0], b[3]]
    big = image.resize((int(w * 1.3), int(h * 1.3)))
    yield "up", big, lambda b: b  # normalized coords are scale-invariant


def norm(box, w, h):
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]


def run_tta_on_image(model, processor, dtype, device, image, min_agreement):
    w, h = image.width, image.height
    boxes_list, scores_list, labels_list = [], [], []
    for _name, view_img, untransform in views(image):
        dets = detect(model, processor, dtype, device, view_img)
        vb, vs, vl = [], [], []
        for bbox, raw_label in dets:
            cls = map_label(raw_label)
            if cls is None:
                continue
            nb = untransform(norm(bbox, view_img.width, view_img.height))
            nb = [min(max(c, 0.0), 1.0) for c in nb]
            if nb[2] <= nb[0] or nb[3] <= nb[1]:
                continue
            vb.append(nb); vs.append(1.0); vl.append(CLASS_TO_ID[cls])
        boxes_list.append(vb); scores_list.append(vs); labels_list.append(vl)

    if not any(boxes_list):
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.0,
    )
    # WBF fused score with all-ones inputs approximates (n_views_agreeing / n_views).
    n_views = sum(1 for bl in boxes_list if bl)
    out = []
    for nb, sc, lb in zip(fused_boxes, fused_scores, fused_labels):
        agreement = round(sc * n_views)          # how many views supported this box
        if agreement < min_agreement:
            continue
        bbox = [nb[0] * w, nb[1] * h, nb[2] * w, nb[3] * h]
        if not passes_filters(bbox, w, h):
            continue
        out.append((bbox, int(lb), agreement))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Florence-2-base")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-agreement", type=int, default=2,
                    help="min number of TTA views a box must appear in (1-3)")
    ap.add_argument("--out", default=str(ANN_DIR / "annotations_tta.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_root = resolve_images_root()
    model, processor, dtype = load_florence(args.model, device)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": CLASS_TO_ID[c], "name": c} for c in CLASSES]}
    ann_id = 0
    kept = {c: 0 for c in CLASSES}
    for img_id, img_path in enumerate(iter_images(images_root, args.limit)):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        dets = run_tta_on_image(model, processor, dtype, device, image, args.min_agreement)
        if not dets:
            continue
        coco["images"].append({"id": img_id, "file_name": str(img_path),
                               "width": image.width, "height": image.height})
        for bbox, cid, agreement in dets:
            x1, y1, x2, y2 = bbox
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0, "tta_agreement": agreement,
            })
            ann_id += 1
            kept[CLASSES[cid]] += 1
        if (img_id + 1) % 50 == 0:
            print(f"  {img_id+1} images | kept {kept}")

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(coco), encoding="utf-8")
    print(f"\nTTA done. images={len(coco['images'])} boxes={len(coco['annotations'])} "
          f"per-class={kept}\nSaved: {args.out}")


if __name__ == "__main__":
    main()
