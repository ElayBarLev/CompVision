"""A minimal COCO-style detection Dataset for torchvision detectors.

Our annotation JSON stores category_id in {0: person, 1: vehicle}. Torchvision detection
models reserve label 0 for *background*, so we shift labels by +1 here:
    background=0, person=1, vehicle=2  ->  num_classes = 3
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 3  # background + person + vehicle


class CocoDetectionDataset(Dataset):
    def __init__(self, ann_file: str, transforms=None):
        coco = json.loads(Path(ann_file).read_text(encoding="utf-8"))
        self.images = coco["images"]
        self.transforms = transforms

        # group annotations by image_id
        self._by_image: dict[int, list] = {im["id"]: [] for im in self.images}
        for a in coco["annotations"]:
            if a["image_id"] in self._by_image:
                self._by_image[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = Image.open(info["file_name"]).convert("RGB")

        boxes, labels = [], []
        for a in self._by_image[info["id"]]:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])      # xyxy
            labels.append(a["category_id"] + 1)      # +1 for background shift

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([info["id"]]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def collate_fn(batch):
    """Detection batches are lists of (image, target) — keep them as tuples."""
    return tuple(zip(*batch))
