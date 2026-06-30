"""Shared COCO-format helpers for the annotation pipeline.

The three annotators (annotate.py, tta_annotate.py, ensemble_annotate.py) all build the same
COCO-style structures. Keeping the construction here means the JSON shape is defined once.
"""
from __future__ import annotations


def new_coco(classes, class_to_id, info=None) -> dict:
    """A COCO scaffold with our categories. `info` is attached only when given."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": class_to_id[c], "name": c} for c in classes],
    }
    if info is not None:
        coco["info"] = info
    return coco


def coco_annotation(ann_id: int, img_id: int, cat_id: int, bbox_xyxy, **extra) -> dict:
    """One COCO annotation dict from an xyxy box (converted to xywh + area).

    `extra` carries optional per-annotator fields (e.g. tta_agreement, ensemble_score).
    """
    x1, y1, x2, y2 = bbox_xyxy
    ann = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": cat_id,
        "bbox": [x1, y1, x2 - x1, y2 - y1],   # COCO xywh
        "area": (x2 - x1) * (y2 - y1),
        "iscrowd": 0,
    }
    ann.update(extra)
    return ann


def normalize_box(box, w: float, h: float) -> list:
    """xyxy pixel box -> xyxy normalized to [0, 1] by image size (for WBF)."""
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
