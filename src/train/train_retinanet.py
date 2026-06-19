"""Convenience wrapper: train RetinaNet (ResNet50-FPN, one-stage / Focal Loss).

    python src/train/train_retinanet.py --augment --epochs 25
"""
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent / "train.py"


def main():
    args = sys.argv[1:]
    if "--arch" not in args:
        args = ["--arch", "retinanet_resnet50", *args]
    raise SystemExit(subprocess.call([sys.executable, str(THIS), *args]))


if __name__ == "__main__":
    main()
