"""Model factory for the two detectors we compare (torchvision).

We expose a small set of architectures. For the *edge* story we default to the
MobileNetV3-FPN Faster R-CNN (small, fast) but also offer the ResNet50-FPN variants for a
higher-accuracy comparison.

    fasterrcnn_mobilenet   - lightest, best edge candidate
    fasterrcnn_resnet50    - stronger two-stage baseline
    retinanet_resnet50     - one-stage baseline (Focal Loss)
"""
from __future__ import annotations

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(arch: str, num_classes: int, pretrained: bool = True,
                min_size: int | None = None, max_size: int | None = None):
    """Return a torchvision detection model with the head resized to num_classes
    (num_classes INCLUDES background).

    min_size/max_size override the internal resize transform. Smaller = less VRAM +
    faster, and closer to edge-inference resolution. None keeps torchvision defaults (800).
    """
    weights = "DEFAULT" if pretrained else None
    size = {}
    if min_size is not None:
        size["min_size"] = min_size
    if max_size is not None:
        size["max_size"] = max_size

    if arch == "fasterrcnn_mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, **size)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif arch == "fasterrcnn_resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights, **size)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif arch == "retinanet_resnet50":
        # num_classes here is foreground+background; torchvision's retinanet head counts
        # classes directly, so pass num_classes and let it build the head.
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=weights, num_classes=None, **size
        )
        _replace_retinanet_head(model, num_classes)

    else:
        raise ValueError(f"Unknown arch: {arch}")

    return model


def _replace_retinanet_head(model, num_classes: int):
    """Swap RetinaNet's classification head to our class count."""
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes
    )


ARCHS = ["fasterrcnn_mobilenet", "fasterrcnn_resnet50", "retinanet_resnet50"]
