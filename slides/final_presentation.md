---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 26px; }
  h1 { color: #1a3a6b; }
  h2 { color: #2a6df4; }
  table { font-size: 22px; }
  section.lead h1 { font-size: 46px; }
  section.lead { text-align: center; }
  .small { font-size: 20px; color: #555; }
  .win { color: #137333; font-weight: 700; }
  .bad { color: #b3261e; font-weight: 700; }
---

<!-- _class: lead -->

# Building an Edge Detector — Without Human Labels

### Semi-supervised dataset creation + a small, edge-deployable Person/Vehicle detector

**Course 10224 — Introduction to Computer Vision · Final Project 2026**

<span class="small">Elay Barlev</span>

---

## The task, in one slide

**Goal:** train a *small* object detector for **Person** and **Vehicle** — with **no human annotation**.

Three-stage pipeline:

1. **Auto-annotate** Flickr images with a large VLM (Florence-2) in inference mode → bounding boxes.
2. **Train** a small detector (torchvision Faster R-CNN & RetinaNet) on those auto-labels.
3. **Compare** raw vs. augmented, report **mAP** + edge-fitness, pick a winner.

**Hard requirements met:** classes Person + Vehicle · ≥ 500 train / 100 val **per class** · trained weights, training code, dataset code, single-image + folder inference.

<span class="small">Framework: Torchvision / native PyTorch. Hardware: RTX 3080 Laptop, 8 GB VRAM, Windows.</span>

---

## Why these choices

- **Torchvision, not YOLO/EfficientDet** — the point is to *learn the fundamentals* and build the detection head ourselves, on one clean stack. (YOLO-World / YOLOE are also banned by the brief.)
- **Train BOTH Faster R-CNN (two-stage) and RetinaNet (one-stage)** — then decide with evidence, not vibes. An honest one-stage vs. two-stage comparison is exactly the trade-off exercise.
- **8 GB VRAM is the real constraint** — small batches, mixed precision (AMP), free GPU memory between stages.
- **"Edge-deployable" is a first-class metric**, not an afterthought: we measure params, model size, and CPU/GPU latency for every model.

---

## Stage 1 — Auto-annotation: the key dataset challenge

First instinct: Florence-2 **`<CAPTION_TO_PHRASE_GROUNDING>`** with `"person. car. bicycle…"`.
→ It **hallucinated vehicles**: blue "vehicle" boxes on indoor scenes, a wedding, a guitarist.

**Why:** grounding answers *"where are these phrases?"* — it grounds **every** phrase, even absent ones. Great recall, but it fabricates objects → that noise would poison the student model.

| Method (same 60 images) | person | vehicle | quality |
|---|---|---|---|
| `<CAPTION_TO_PHRASE_GROUNDING>` | 117 | 92 | many **false** vehicles |
| **`<OD>`** (generic detection) | 202 | 22 | clean — vehicles only where real |
| **`<OD>` + COCO ensemble** | 295 | 38 | clean **and** higher recall |

---

## Seeing the difference

![w:560](../outputs/figures/compare_grounding.png) ![w:560](../outputs/figures/compare_od.png)

<span class="small">Left: phrase-grounding invents vehicles. Right: <code>&lt;OD&gt;</code> reports only what's really there.
Full qualitative samples: <code>outputs/figures/annotation_samples.png</code>.</span>

---

## The dataset gap — and the ensemble fix

`<OD>` is clean but **vehicles are genuinely sparse** in Flickr30k → only **98 vehicle images in val**, *short of the 100 minimum*. We can't fake labels, so we **add a second model**.

- Fuse Florence-2 `<OD>` with a **COCO-pretrained Faster R-CNN** (strong on vehicles) via **Weighted Boxes Fusion**.
- Key detail: **`conf_type="max"`** so a box found by only one model is **not** suppressed — the COCO detector *contributes* real vehicles instead of being penalized.

**Result: +52% vehicles, +43% people, still clean → requirement now passes.**

| | Person | Vehicle |
|---|---|---|
| Train images | 3,893 | 758 |
| **Val images** | 685 | **138** ✅ (was 98) |

![bg right:30% w:340](../outputs/figures/compare_ensemble.png)

---

## Stage 2 — Training setup

- Same auto-generated dataset for **both** architectures (fair comparison).
- **SGD + Cosine LR** schedule, **AMP / mixed precision** to fit 8 GB VRAM.
- Save best checkpoint by **val mAP**; export metrics JSON + training curves.
- Measure **edge-fitness** for every model: params, size (MB), CPU & GPU latency.
- **Raw vs. augmented** toggled by one `--augment` flag (albumentations: flips, color jitter, scale — bbox-aware).

![bg right:38% w:440](../outputs/figures/fasterrcnn_mobilenet_full_lr0025.png)

<span class="small">Curve: final model — loss falls, val mAP climbs then plateaus cleanly.</span>

---

## Model comparison — accuracy is a near-tie, edge isn't

| model | mAP@[.5:.95] | mAP@.5 | size | CPU ms | GPU fps |
|---|---|---|---|---|---|
| **Faster R-CNN MobileNet (FINAL)** | 0.399 | <span class="win">0.703</span> | <span class="win">72 MB</span> | <span class="win">65</span> | 38 |
| Faster R-CNN MobileNet (aug) | 0.397 | 0.696 | 72 MB | 56 | 33 |
| RetinaNet ResNet50 (raw) | 0.404 | 0.673 | <span class="bad">123 MB</span> | <span class="bad">466</span> | 3 |

- Accuracy **~tied** on mAP@[.5:.95]; MobileNet actually **wins mAP@.5** (0.70 vs 0.67).
- Edge fitness **not close**: MobileNet is **~7× faster on CPU** and **40% smaller**.

> ⚠️ Our auto-generated `summary.md` blindly argmaxes raw mAP and labels RetinaNet "best" — that **ignores the edge goal**. The honest winner for *this project* is **MobileNet Faster R-CNN**.

![bg right:30% w:340](../outputs/figures/model_comparison.png)

---

## Raw vs. augmented — reported honestly

| Faster R-CNN MobileNet | mAP@[.5:.95] | mAP@.5 |
|---|---|---|
| Raw | 0.399 | **0.703** |
| Augmented | 0.397 | 0.696 |

- **Augmentation was neutral** here for Faster R-CNN — *within noise*, not a win. We report that rather than overclaim.
- It made **RetinaNet underfit**: its from-scratch focal-loss head sat at **0 mAP for 3 epochs**, still rising at epoch 15 (0.21, not converged).
- **Insight:** RetinaNet needs more epochs / warmup / milder aug. Faster R-CNN only retrains a small box-predictor, so it's robust. → clear future-work item.

---

## The two lessons that mattered most

**1. It was the learning rate, not "needs few epochs."**
A 30-epoch run at **LR 0.01** peaked at epoch 3 then *declined* — the high LR was degrading pretrained features. At **LR 0.0025** the curve climbs and **plateaus with no decline**. The plateau is the **noisy auto-label ceiling**, not model capacity.

**2. We caught data leakage.**
Subset and full datasets were split *independently* → **85% of full-val images had been in the subset model's training set**, inflating it to a fake 0.464. Fix: **one canonical, disjoint split** across all experiments.

<span class="small">Clean, leakage-free full-val result: <b>mAP 0.437 / mAP@.5 0.700</b> (690 imgs, 800px eval).</span>

---

## Verifying labels with no ground truth — TTA & Ensembles

The brief asks *how* we'd verify auto-labels. Core idea: **predictions that survive perturbation, or that multiple models agree on, are more trustworthy.**

- **TTA** — run the annotator on flips/scales of the same image, map boxes back, fuse with WBF.
  Survives all views → keep (high agreement); one view only → likely spurious → drop.
- **Ensembles** — annotate with *two different models*; boxes both agree on are high-confidence; solo boxes are flagged or fill a recall gap. Class disagreement flags ambiguous cases.

**We didn't just write it up — we implemented both** (`src/tta_ensemble/`). The ensemble is exactly what rescued our vehicle count. <span class="small">(bonus, beyond the required write-up)</span>

---

## Bonus — it really runs on a phone, on-device

![bg right:34% w:380](../outputs/figures/final_demo_1.jpg)

Open a URL on a phone → rear camera → live person/vehicle boxes, **computed on the phone itself** (no server).

- Export to **ONNX**; the torchvision model bakes normalize + resize + NMS into `forward`, so the browser just feeds a raw RGB tensor → boxes.
- Runs via **onnxruntime-web (WASM)** — covers RoiAlign / NMS / TopK reliably. Lowered export `min_size` 800 → **512** for phone speed.
- Hosted on **GitHub Pages** (HTTPS, required for the camera API).

🔗 **`https://elaybarlev.github.io/CompVision/web/`**

---

## Results recap & future work

**Final model:** Faster R-CNN, MobileNetV3-FPN — `weights/FINAL_fasterrcnn_mobilenet.pt`
**mAP 0.437 · mAP@.5 0.703 · 72 MB · ~65 ms CPU (~14 fps) · 38 fps GPU**

**What we'd do next**
- **Better labels = the highest-leverage lever** — the mAP plateau is the auto-label ceiling, not the model. More TTA/ensemble cleanup (or SAM3) raises the whole curve.
- **Retrain RetinaNet properly** under augmentation (warmup + more epochs).
- **Native Android (ExecuTorch / NNAPI)** for real-time, accelerator-backed inference.

---

<!-- _class: lead -->

# Thank you

**Built without a single human label** — Florence-2 auto-annotation → ensemble cleanup →
torchvision training → a 72 MB detector that runs live on a phone.

<span class="small">Code, docs, and the live demo: github.com/ElayBarLev/CompVision</span>
