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


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """Return a torchvision detection model with the head resized to num_classes
    (num_classes INCLUDES background)."""
    weights = "DEFAULT" if pretrained else None

    if arch == "fasterrcnn_mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif arch == "fasterrcnn_resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif arch == "retinanet_resnet50":
        # num_classes here is foreground+background; torchvision's retinanet head counts
        # classes directly, so pass num_classes and let it build the head.
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=weights, num_classes=None
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
