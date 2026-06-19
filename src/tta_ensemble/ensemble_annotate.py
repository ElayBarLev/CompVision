"""BONUS — Ensemble-based annotation quality check.

Idea (see docs/05): annotate each image with TWO models of different biases —
Florence-2 (open-vocab VLM) and a COCO-pretrained torchvision Faster R-CNN — then fuse with
Weighted Boxes Fusion. Boxes where BOTH models agree are high-confidence; boxes only one
model produces are flagged as uncertain. We keep boxes with fused score >= --min-score.

    python src/tta_ensemble/ensemble_annotate.py --limit 500 --min-score 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.annotate import (detect, iter_images, load_florence,  # noqa: E402
                                  map_label, passes_filters, resolve_images_root)
from src.utils.config import CLASS_TO_ID, CLASSES                       # noqa: E402

ANN_DIR = PROJECT_ROOT / "data" / "annotations"

# COCO label id -> our class (torchvision COCO ids)
COCO_TO_CLASS = {1: "person", 2: "vehicle", 3: "vehicle", 4: "vehicle",
                 6: "vehicle", 8: "vehicle"}  # person, bicycle, car, motorcycle, bus, truck


def load_coco_detector(device):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    return m.to(device).eval()


@torch.inference_mode()
def coco_detect(model, device, image, score_thr=0.3):
    t = torch.from_numpy(_np(image)).permute(2, 0, 1).float().div(255.0).to(device)
    out = model([t])[0]
    res = []
    for box, score, lab in zip(out["boxes"].cpu().tolist(),
                               out["scores"].cpu().tolist(),
                               out["labels"].cpu().tolist()):
        if score < score_thr or lab not in COCO_TO_CLASS:
            continue
        res.append((box, COCO_TO_CLASS[lab], score))
    return res


def _np(image):
    import numpy as np
    return np.asarray(image.convert("RGB"))


def norm(box, w, h):
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]


def fuse_image(flo, processor, dtype, device, coco_model, image, min_score):
    w, h = image.width, image.height
    # model 1: Florence-2 (no scores -> assign 1.0)
    fb, fs, fl = [], [], []
    for bbox, raw in detect(flo, processor, dtype, device, image):
        cls = map_label(raw)
        if cls is None:
            continue
        fb.append(norm(bbox, w, h)); fs.append(1.0); fl.append(CLASS_TO_ID[cls])
    # model 2: COCO detector (real scores)
    cb, cs, cl = [], [], []
    for box, cls, score in coco_detect(coco_model, device, image):
        cb.append(norm(box, w, h)); cs.append(score); cl.append(CLASS_TO_ID[cls])

    if not fb and not cb:
        return []
    boxes, scores, labels = weighted_boxes_fusion(
        [fb, cb], [fs, cs], [fl, cl], weights=[1, 1], iou_thr=0.5, skip_box_thr=0.0,
    )
    out = []
    for nb, sc, lb in zip(boxes, scores, labels):
        if sc < min_score:
            continue
        bbox = [nb[0] * w, nb[1] * h, nb[2] * w, nb[3] * h]
        if not passes_filters(bbox, w, h):
            continue
        out.append((bbox, int(lb), float(sc)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Florence-2-base")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--out", default=str(ANN_DIR / "annotations_ensemble.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_root = resolve_images_root()
    flo, processor, dtype = load_florence(args.model, device)
    coco_model = load_coco_detector(device)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": CLASS_TO_ID[c], "name": c} for c in CLASSES]}
    ann_id = 0
    kept = {c: 0 for c in CLASSES}
    for img_id, img_path in enumerate(iter_images(images_root, args.limit)):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        dets = fuse_image(flo, processor, dtype, device, coco_model, image, args.min_score)
        if not dets:
            continue
        coco["images"].append({"id": img_id, "file_name": str(img_path),
                               "width": image.width, "height": image.height})
        for bbox, cid, score in dets:
            x1, y1, x2, y2 = bbox
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0, "ensemble_score": round(score, 3)})
            ann_id += 1
            kept[CLASSES[cid]] += 1
        if (img_id + 1) % 50 == 0:
            print(f"  {img_id+1} images | kept {kept}")

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(coco), encoding="utf-8")
    print(f"\nEnsemble done. images={len(coco['images'])} boxes={len(coco['annotations'])} "
          f"per-class={kept}\nSaved: {args.out}")


if __name__ == "__main__":
    main()
