# Decision Doc — Detector Architecture

The brief asks us to pick **one** architecture, justify it, and say why we rejected the
others. We will **train both Faster R-CNN and RetinaNet** first, then choose with evidence.
This doc explains the candidates so the decision (and the slide) is grounded.

## The candidates (and why we keep / drop each)

### ✅ Faster R-CNN — *kept, will train*
- **Two-stage** detector: a Region Proposal Network proposes boxes, then a head classifies
  + refines them.
- **Pros:** historically high accuracy, very well documented, first-class in torchvision
  (`fasterrcnn_resnet50_fpn`, and a mobile-friendly `fasterrcnn_mobilenet_v3_large_fpn`).
- **Cons:** slower, heavier than one-stage detectors — but the mobilenet variant is a
  legitimate edge candidate.
- **Why kept:** the canonical two-stage baseline; pairs perfectly against a one-stage model
  for an honest comparison, and torchvision gives us a clean training path.

### ✅ RetinaNet — *kept, will train*
- **One-stage** detector with **Focal Loss** (its key idea: fixes the foreground/background
  class imbalance that hurt earlier one-stage models).
- **Pros:** simpler/faster than two-stage, strong accuracy, native in torchvision
  (`retinanet_resnet50_fpn`).
- **Cons:** anchor tuning matters; can lag two-stage on small objects.
- **Why kept:** the clean one-stage counterpart to Faster R-CNN — comparing the two is
  exactly the "understand the trade-off" exercise we want.

### ❌ YOLO family — *dropped*
- Excellent speed/accuracy and edge-export, **but** it lives in Ultralytics, not torchvision.
  We chose torchvision/PyTorch to learn the fundamentals and build the head ourselves; YOLO
  would mean adopting a different framework. (Also YOLO-World / YOLOE are explicitly banned.)

### ❌ EfficientDet — *dropped*
- Strong accuracy/efficiency (BiFPN), **but** not in torchvision; best supported via
  MMDetection. Off our chosen stack.

### ❌ MobileNetV4 (custom backbone + head) — *dropped*
- The brief flags this as advanced (integrate a MobileNetV4 backbone via `timm` with a
  detection head, best in MMDetection). High effort/risk for the timeline. *Note:* we still
  get an edge-friendly option through torchvision's **MobileNetV3-FPN** Faster R-CNN variant.

### ❌ "Another model" — *dropped*
- Requires prior staff approval; no reason to take that path.

## How we will decide between the two we keep
Train both on the *same* auto-generated dataset and compare:
1. **Accuracy** — COCO mAP (mAP@[.5:.95]) and mAP@.5 on the validation split.
2. **Edge-fitness** — params, model size (MB), and inference latency (the project's whole
   point is edge deployment).
3. **Training behavior** — convergence speed, stability on 8 GB VRAM.

Likely backbone choice for the *edge* story: a **MobileNetV3-FPN** or ResNet50-FPN backbone;
we'll note the accuracy-vs-size trade-off in the slides.

## Final decision
- **Will train:** Faster R-CNN **and** RetinaNet (torchvision).
- **Final pick:** _[after we see val mAP, size, and latency — fill in]_
- **One-line rationale for the slide:** _[fill in]_
