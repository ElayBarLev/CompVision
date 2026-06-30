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

## Final-model investigation (full-dataset retrain)
Retraining the winner on the full ensemble set surfaced two important lessons:

1. **LR was too high, not "needs few epochs".** A 30-epoch full run at **LR 0.01** peaked at
   epoch 3 (mAP@.5 0.66) then *declined* to 0.59 — the high LR was degrading the pretrained
   features. Re-running at **LR 0.0025** the curve climbs then **plateaus ~0.655 with no
   decline**. So training longer is safe at the right LR; the plateau is the *noisy-label
   ceiling* (our val labels are auto-generated too).
2. **Data-leakage catch.** The subset and full datasets were split independently, so
   **85% of the full-val images had been in the subset model's training set** — inflating the
   subset model to a fake 0.464. Always keep one canonical, disjoint split across experiments.

Leakage-free comparison on the full val set (690 imgs, 800px eval):
| model | mAP@[.5:.95] | mAP@.5 |
|---|---|---|
| full / 30 ep / LR 0.01 | 0.427 | 0.699 |
| **full / 15 ep / LR 0.0025** | **0.437** | **0.700** |

## Final decision
- **Architecture:** **Faster R-CNN, MobileNetV3-FPN backbone** (chosen on the valid same-val
  comparison: ≈RetinaNet accuracy at ~1/8 CPU latency, ~60% the size).
- **Final weights:** `weights/fasterrcnn_mobilenet_full_lr0025_best.pt` — trained on the full
  ensemble dataset, LR 0.0025, 15 epochs. **mAP 0.437 / mAP@.5 0.700** (clean full val),
  72 MB, ~73 ms CPU / 26 ms GPU per image at 512px.
- **One-line rationale for the slide:** *MobileNet Faster R-CNN — edge-sized (72 MB, ~14 fps
  CPU) with mAP@.5 ≈ 0.70, after fixing the LR (0.01→0.0025) that was degrading training.*
- **Ceiling note:** further gains are limited by auto-label quality, not the model — better
  labels (more TTA/ensemble cleanup) is the highest-leverage future work.

## Phone-demo tuning round (2026-06-30) — resolution beats training tricks
The phone runs the model as ONNX in-browser at a small input size for speed, so its *honest*
mAP is at ~512px, **not** the 0.437 @800px headline. Goal: raise the 512px number while
keeping it phone-fast. We added per-class AP, a vehicle-oversampling sampler, postprocess
knobs, and an eval-only script (`src/train/eval.py`), then measured everything on the same
690-img val set.

**Per-class eval grid (final weights `…_lr0025`, evaluated at each resolution):**
| eval res | mAP@[.5:.95] | AP person | AP vehicle |
|---|---|---|---|
| 512 (phone default) | 0.368 | 0.451 | 0.285 |
| **640 (sweet spot)** | **0.426** | **0.523** | **0.330** |
| 800 | 0.437 | 0.541 | 0.334 |

**Training experiments (both *underperformed* the existing weights — kept as honest negatives):**
| run | config | @512 | @640 |
|---|---|---|---|
| `mobilenet_512_base` | retrain *at* 512, raw, 15 ep | 0.342 | — |
| `mobilenet_640_balveh` | 640 + vehicle ×4 oversample, 10 ep | 0.334 | 0.406 |

Findings: (1) **train-low loses** — retraining at 512 (0.342) is worse than the 640-trained
weights run at 512 (0.368); higher train-resolution learns better features that stay robust
when run smaller. (2) **Vehicle oversampling hurt** at every resolution, even on vehicle AP —
×4 weighting (plus the 10-epoch cut) cost person-scene diversity for no vehicle gain.
(3) **NMS threshold** is neutral (0.5 optimal). So the only real lever is the **resolution
the same weights run at**: 512→640 buys **+0.058 mAP / +0.045 vehicle AP for free**, 640→800
adds only +0.011 (diminishing returns).

**Recommendation:** keep `…_lr0025` as the model; run the phone demo at **640**. That needs two
coordinated changes (both prepared): export ONNX at `--min-size 640 --max-size 1024`
(`outputs/edge/model_640.onnx`, ready) **and** raise `procMax` in `web/app.js` (line 18) from
512 toward ~853–1024 so the browser actually feeds 640-shortest-side detail. Cost: ~1.5–2×
in-browser inference time — verify it stays smooth on the presenter's phone before swapping
`web/model.onnx`.

### Follow-up: a dedicated 800px "all-in" run — and why it lost (2026-06-30)
To try to *beat* `lr0025`, we trained one definitive model **at 800px** (`FINAL_mobilenet_800`:
raw, LR 0.0025, 15 ep, full ensemble). Two operational notes: (1) **AMP is pathological for
Faster R-CNN at 800px** — mixed precision crawled at >1.8 s/batch (the fp16 box-op path,
cf. decision #12); `--no-amp` ran at **0.117 s/batch** (~1 h total). (2) Its best val mAP
landed at **epoch 2** then plateaued — the noisy-label ceiling again.

Per-class eval, new 800-trained model vs the incumbent, at each resolution:
| eval res | `FINAL_mobilenet_800` | `…_lr0025` (incumbent) |
|---|---|---|
| 512 | 0.319 / v0.240 | **0.368 / v0.285** |
| **640 (deploy)** | 0.402 / v0.314 | **0.426 / v0.330** |
| 800 | 0.435 / v0.340 | **0.437 / v0.334** |

**The 800-trained model is worse at 640 (0.402 < 0.426) and at 512, and only ties at 800.**
Training *at* a high resolution over-specializes to it and degrades more at the lower deploy
resolution — the same train↔deploy-gap penalty seen when training at 512. A model trained
around the deploy resolution (≈640) generalizes *upward* to 800 better than one trained at 800
generalizes *downward* to 640. **Conclusion: `lr0025` remains the best deploy-at-640 model; no
further training is warranted (the ceiling is auto-label quality, not resolution or epochs).**
The only outstanding win is deploying `lr0025` itself at 640 (artifact ready), a user-gated
decision because it ~1.5–2×s the on-phone latency.
