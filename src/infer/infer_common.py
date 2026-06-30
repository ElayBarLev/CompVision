"""Shared inference helpers used by infer_image.py and infer_folder.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.build_model import build_model           # noqa: E402
from src.utils.config import CLASSES                    # noqa: E402

# label id (after +1 background shift) -> name / colour
ID_TO_NAME = {i + 1: name for i, name in enumerate(CLASSES)}
COLORS = {1: (220, 40, 40), 2: (40, 120, 220)}  # person=red, vehicle=blue


def load_model(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict(model, image: Image.Image, device: str, score_thresh: float):
    img_t = torch.from_numpy(_np(image)).permute(2, 0, 1).float().div(255.0).to(device)
    out = model([img_t])[0]
    keep = out["scores"] >= score_thresh
    return {
        "boxes": out["boxes"][keep].cpu().tolist(),
        "scores": out["scores"][keep].cpu().tolist(),
        "labels": out["labels"][keep].cpu().tolist(),
    }


def _np(image: Image.Image):
    return np.asarray(image.convert("RGB"))


def draw(image: Image.Image, pred) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        x1, y1, x2, y2 = box
        color = COLORS.get(label, (0, 200, 0))
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"{ID_TO_NAME.get(label, label)} {score:.2f}"
        d.text((x1 + 2, max(0, y1 - 18)), tag, fill=color, font=font)
    return img
