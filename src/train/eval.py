"""Evaluate a trained detector checkpoint at a chosen resolution + postprocess.

Why this exists
---------------
train.py only reports mAP at the *training* resolution, and the old 800px "final" number
was produced ad-hoc. For the phone story we care about mAP at the **deployment** resolution
(the ONNX export uses min_size 512). This script re-scores any weights/*_best.pt on val.json
at a fixed --min-size/--max-size and prints mean + per-class (person/vehicle) AP, and lets us
sweep --nms-thresh WITHOUT retraining.

Usage
-----
    python src/train/eval.py --weights weights/mobilenet_512_balveh_best.pt
    python src/train/eval.py --weights ... --min-size 512 --max-size 640 --nms-thresh 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.engine import evaluate                               # noqa: E402
from src.utils.build_model import build_model                       # noqa: E402
from src.utils.coco_dataset import (CocoDetectionDataset,           # noqa: E402
                                     NUM_CLASSES, collate_fn)
from src.utils.transforms import build_transform                    # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="path to a weights/*_best.pt checkpoint")
    ap.add_argument("--data-dir", default=str(PROC / "ensemble"),
                    help="dir holding val.json")
    ap.add_argument("--min-size", type=int, default=512,
                    help="eval shorter-side size (match the phone export: 512)")
    ap.add_argument("--max-size", type=int, default=640)
    ap.add_argument("--nms-thresh", type=float, default=None,
                    help="box NMS IoU threshold (None = torchvision default 0.5)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location="cpu")
    pp = {} if args.nms_thresh is None else {"box_nms_thresh": args.nms_thresh}
    model = build_model(ckpt["arch"], ckpt.get("num_classes", NUM_CLASSES), pretrained=False,
                        min_size=args.min_size, max_size=args.max_size, **pp)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    val_ds = CocoDetectionDataset(str(Path(args.data_dir) / "val.json"), build_transform(False))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    print(f"=== eval {Path(args.weights).name} @ min{args.min_size}/max{args.max_size}"
          f"{'' if args.nms_thresh is None else f' nms={args.nms_thresh}'} on {device} ===")
    m = evaluate(model, val_loader, device)
    print(f"  mAP@[.5:.95] : {m['map']:.4f}")
    print(f"  mAP@.5       : {m['map_50']:.4f}")
    print(f"  mAP@.75      : {m['map_75']:.4f}")
    print(f"  AP person    : {m['map_person']:.4f}")
    print(f"  AP vehicle   : {m['map_vehicle']:.4f}")


if __name__ == "__main__":
    main()
