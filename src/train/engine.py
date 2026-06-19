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
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
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
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds = [{"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(),
                  "labels": o["labels"].cpu()} for o in outputs]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, gts)
    res = metric.compute()
    return {"map": float(res["map"]), "map_50": float(res["map_50"]),
            "map_75": float(res["map_75"])}
