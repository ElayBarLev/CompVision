"""Stage 2 — Train a torchvision detector on our auto-generated dataset.

Handles both detectors and the raw-vs-augmented comparison via flags. Saves: best weights
(by val mAP), a loss/mAP curve PNG, and a metrics JSON — the materials the slides need.

Examples:
    python src/train/train.py --arch fasterrcnn_mobilenet --epochs 20
    python src/train/train.py --arch retinanet_resnet50 --augment --epochs 20

Run each arch twice (with and without --augment) to produce the raw-vs-augmented result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.engine import evaluate, train_one_epoch          # noqa: E402
from src.utils.build_model import build_model                    # noqa: E402
from src.utils.coco_dataset import (CocoDetectionDataset,        # noqa: E402
                                     NUM_CLASSES, collate_fn)
from src.utils.transforms import build_transform                 # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
WEIGHTS = PROJECT_ROOT / "weights"
FIGS = PROJECT_ROOT / "outputs" / "figures"
METRICS = PROJECT_ROOT / "outputs" / "metrics"


def make_loaders(batch_size, augment, workers):
    train_ds = CocoDetectionDataset(str(PROC / "train.json"), build_transform(augment))
    val_ds = CocoDetectionDataset(str(PROC / "val.json"), build_transform(False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader


def plot_history(history, tag):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], "b-o", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["map"] for h in history], "r-s", label="val mAP")
    ax2.plot(epochs, [h["map_50"] for h in history], "g-^", label="val mAP@.5")
    ax2.set_ylabel("val mAP", color="r")
    fig.suptitle(tag)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{tag}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved curve: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["fasterrcnn_mobilenet", "fasterrcnn_resnet50", "retinanet_resnet50"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)   # 8 GB VRAM -> keep small
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.arch}{'_aug' if args.augment else '_raw'}"
    print(f"=== Training {tag} on {device} ===")

    train_loader, val_loader = make_loaders(args.batch_size, args.augment, args.workers)
    print(f"train batches: {len(train_loader)} | val images: {len(val_loader)}")

    model = build_model(args.arch, NUM_CLASSES, pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not args.no_amp))

    history, best_map = [], -1.0
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(f"epoch {epoch}: loss={train_loss:.4f} "
              f"mAP={metrics['map']:.4f} mAP@.5={metrics['map_50']:.4f}")

        if metrics["map"] > best_map:
            best_map = metrics["map"]
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "num_classes": NUM_CLASSES, "epoch": epoch, "map": best_map},
                       WEIGHTS / f"{tag}_best.pt")
            print(f"  ** new best mAP {best_map:.4f} -> saved {tag}_best.pt")

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"{tag}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, tag)
    print(f"\nDone. Best val mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
