"""Deliverable — inference on every image in a FOLDER.

    python src/infer/infer_folder.py --weights weights/..._best.pt \
                                     --folder data/raw/some_dir --out-dir outputs/preds
Writes annotated images to --out-dir and a predictions.json with all detections.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.infer.infer_common import draw, load_model, predict  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out-dir", default="outputs/preds")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--save-images", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(Path(args.folder).rglob("*")) if p.suffix.lower() in EXTS]
    print(f"Found {len(files)} images in {args.folder}")

    all_preds = {}
    for i, p in enumerate(files):
        try:
            image = Image.open(p).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        pred = predict(model, image, device, args.score)
        all_preds[p.name] = pred
        if args.save_images:
            draw(image, pred).save(out_dir / f"{p.stem}_pred.jpg")
        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{len(files)}")

    (out_dir / "predictions.json").write_text(json.dumps(all_preds, indent=2), encoding="utf-8")
    total = sum(len(v["boxes"]) for v in all_preds.values())
    print(f"Done. {len(all_preds)} images, {total} detections. Output -> {out_dir}")


if __name__ == "__main__":
    main()
