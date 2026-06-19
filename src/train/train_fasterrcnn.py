"""Convenience wrapper: train Faster R-CNN.

Defaults to the edge-friendly MobileNetV3-FPN backbone. Pass extra flags through, e.g.:
    python src/train/train_fasterrcnn.py --augment --epochs 25
    python src/train/train_fasterrcnn.py --arch fasterrcnn_resnet50
"""
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent / "train.py"


def main():
    args = sys.argv[1:]
    if "--arch" not in args:
        args = ["--arch", "fasterrcnn_mobilenet", *args]
    raise SystemExit(subprocess.call([sys.executable, str(THIS), *args]))


if __name__ == "__main__":
    main()
