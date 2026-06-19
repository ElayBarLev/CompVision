"""Transforms for detection. Two pipelines so we can compare raw vs. augmented:

  - build_transform(augment=False): just PIL -> float tensor in [0,1].
    (torchvision detectors normalize + resize internally, so nothing else is needed.)
  - build_transform(augment=True): albumentations augmentations that also transform the
    bounding boxes (flip, scale/crop, brightness/contrast, blur).

The raw-vs-augmented comparison is a required result in the brief.
"""
from __future__ import annotations

import numpy as np
import torch


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [0,1]."""
    return torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0)


class DetectionTransform:
    def __init__(self, augment: bool):
        self.augment = augment
        self.aug = None
        if augment:
            import albumentations as A  # imported lazily so raw path needs no albumentations

            self.aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.HueSaturationValue(p=0.2),
                    A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-8, 8), p=0.5),
                    A.MotionBlur(blur_limit=5, p=0.1),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",          # xyxy
                    label_fields=["labels"],
                    min_visibility=0.3,            # drop boxes mostly cropped away
                ),
            )

    def __call__(self, img_pil, target):
        img_np = np.asarray(img_pil)  # HWC uint8 RGB
        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()

        if self.aug is not None and len(boxes) > 0:
            out = self.aug(image=img_np, bboxes=boxes, labels=labels)
            img_np = out["image"]
            boxes = out["bboxes"]
            labels = out["labels"]

        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        if len(boxes):
            b = target["boxes"]
            target["area"] = (b[:, 3] - b[:, 1]) * (b[:, 2] - b[:, 0])
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        return _to_tensor(img_np), target


def build_transform(augment: bool = False) -> DetectionTransform:
    return DetectionTransform(augment=augment)
