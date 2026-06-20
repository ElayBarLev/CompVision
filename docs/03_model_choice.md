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

## Results (750/class subset, 15 epochs, 512px) — raw vs augmented
| arch | mode | mAP@[.5:.95] | mAP@.5 | params | size | CPU lat | GPU lat |
|---|---|---|---|---|---|---|---|
| **Faster R-CNN MobileNet** | raw | **0.399** | **0.703** | 18.9 M | 72 MB | **77 ms** | 40 ms |
| Faster R-CNN MobileNet | aug | 0.397 | 0.696 | 18.9 M | 72 MB | 56 ms | 30 ms |
| RetinaNet ResNet50 | raw | 0.404 | 0.673 | 32.2 M | 123 MB | 466 ms | 344 ms |
| RetinaNet ResNet50 | aug | 0.207 | 0.387 | 32.2 M | 123 MB | 523 ms | 299 ms |

(Full numbers: `outputs/metrics/summary.md`; chart: `outputs/figures/model_comparison.png`.)

### Reading the results
- **Accuracy is ~tied** at mAP@[.5:.95] (0.40 vs 0.40); MobileNet is actually **better at
  mAP@.5 (0.70 vs 0.67)**.
- **Edge fitness is not close:** MobileNet is **~6–8× faster on CPU (77 vs 466 ms)** and
  **40% smaller (72 vs 123 MB)**. For an edge target, that's decisive.
- **Augmentation didn't help here** — neutral for Faster R-CNN (0.399→0.397), and it made
  RetinaNet *underfit*: its from-scratch classification head + focal loss sat at 0 mAP for 3
  epochs and was still rising at epoch 15 (0.21, not converged). Faster R-CNN only retrains a
  small box-predictor, so it was robust. **Insight:** RetinaNet needs more epochs (and/or
  warmup / milder aug) under augmentation — a clear "future work" + retrain item.

## Final decision
- **Will train:** Faster R-CNN **and** RetinaNet (torchvision). ✅ done.
- **Final pick:** **Faster R-CNN with the MobileNetV3-FPN backbone** (`fasterrcnn_mobilenet`).
- **One-line rationale for the slide:** *Equal-or-better accuracy than RetinaNet (mAP@.5 0.70
  vs 0.67) at ~1/8 the CPU latency and ~60% the size — the right model for an edge device.*
- **Next:** retrain this winner on the **full ensemble dataset** with more epochs (and revisit
  augmentation with warmup), per the plan.
