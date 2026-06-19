"""Stage 1a — Download the Flickr image dataset via kagglehub.

The Kaggle dataset `hsankesara/flickr-image-dataset` is the "Flickr30k" image set:
~31,000 everyday photos (people, vehicles, scenes) — a good source of unlabeled images
for our semi-supervised pipeline.

kagglehub downloads to its own cache (usually ~/.cache/kagglehub/...). We then create a
symlink/copy reference under data/raw/ so the rest of the pipeline has a stable path.

Requires Kaggle credentials: ~/.kaggle/kaggle.json  (Kaggle -> Settings -> Create New Token)
or an interactive login.

Usage:
    python src/dataset/download.py
"""
from __future__ import annotations

import os
from pathlib import Path

import kagglehub

DATASET = "hsankesara/flickr-image-dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    print(f"Downloading '{DATASET}' via kagglehub (this can take a while)...")
    # Download latest version
    path = kagglehub.dataset_download(DATASET)
    path = Path(path)
    print(f"Downloaded to cache: {path}")

    # Find the directory that actually holds the .jpg images.
    image_dirs = _find_image_dirs(path)
    if image_dirs:
        print("\nImage directories found:")
        for d, n in image_dirs:
            print(f"  {n:>7d} images  <-  {d}")
    else:
        print("WARNING: no image directories found; inspect the cache path above.")

    # Record where the data lives so other scripts can find it without re-downloading.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pointer = RAW_DIR / "DATASET_PATH.txt"
    pointer.write_text(str(path), encoding="utf-8")
    print(f"\nWrote dataset cache path to: {pointer}")
    print("Other scripts read this file to locate the images.")


def _find_image_dirs(root: Path, exts=(".jpg", ".jpeg", ".png")) -> list[tuple[Path, int]]:
    """Return (directory, image_count) for dirs containing images, biggest first."""
    counts: dict[Path, int] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        n = sum(1 for f in filenames if f.lower().endswith(exts))
        if n:
            counts[Path(dirpath)] = n
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


if __name__ == "__main__":
    main()
