"""Shared training / evaluation loop for the torchvision detectors.

Used by train_fasterrcnn.py and train_retinanet.py so both go through identical code
(fair comparison). Features: mixed precision (AMP) for the 8 GB GPU, COCO mAP via
torchmetrics, and a returned history dict for plotting.
"""
from __future__ import annotations

import math

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_one_epoch(model, optimizer, loader, device, scaler, epoch, log_every=20):
    model.train()
    running = 0.0
    n = 0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)        # detectors return a dict of losses
            loss = sum(loss_dict.values())

        if not math.isfinite(loss.item()):
            print(f"  WARNING non-finite loss ({loss.item()}); skipping batch")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        n += 1
        if (i + 1) % log_every == 0:
            print(f"  epoch {epoch} | batch {i+1}/{len(loader)} | loss {running/n:.4f}")

    return running / max(n, 1)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    # class_metrics=True also returns per-class AP, so we can watch the weak vehicle class
    # (it's the imbalance bottleneck) — not just the person-dominated mean.
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds = [{"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(),
                  "labels": o["labels"].cpu()} for o in outputs]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, gts)
    res = metric.compute()
    out = {"map": float(res["map"]), "map_50": float(res["map_50"]),
           "map_75": float(res["map_75"])}
    out.update(_per_class(res))
    return out


# label id (background-shifted) -> metrics-key name; person=1, vehicle=2 (see coco_dataset.py)
_CLASS_NAMES = {1: "map_person", 2: "map_vehicle"}


def _per_class(res) -> dict:
    """Pull per-class AP out of torchmetrics' map_per_class / classes tensors.

    With class_metrics=True both are 1-D tensors aligned by index; with a single class
    present torchmetrics returns a scalar. Missing classes report -1.
    """
    out = {name: -1.0 for name in _CLASS_NAMES.values()}
    per_class = res.get("map_per_class")
    classes = res.get("classes")
    if per_class is None or classes is None:
        return out
    aps = per_class.reshape(-1).tolist()
    ids = classes.reshape(-1).tolist()
    for cid, ap in zip(ids, aps):
        if int(cid) in _CLASS_NAMES:
            out[_CLASS_NAMES[int(cid)]] = float(ap)
    return out
