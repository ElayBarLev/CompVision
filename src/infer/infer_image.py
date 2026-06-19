"""Deliverable — inference on a SINGLE image.

    python src/infer/infer_image.py --weights weights/retinanet_resnet50_aug_best.pt \
                                    --image path/to/img.jpg --out out.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.infer.infer_common import draw, load_model, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None, help="output path (default: <image>_pred.jpg)")
    ap.add_argument("--score", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    image = Image.open(args.image).convert("RGB")
    pred = predict(model, image, device, args.score)
    print(f"Detections (score>={args.score}): {len(pred['boxes'])}")
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        print(f"  label={label} score={score:.3f} box={[round(v, 1) for v in box]}")

    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_pred.jpg"))
    draw(image, pred).save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
