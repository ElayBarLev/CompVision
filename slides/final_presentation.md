---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 27px; padding: 50px 60px; }
  h1 { color: #1a3a6b; font-size: 40px; }
  h2 { color: #2a6df4; font-size: 34px; }
  strong { color: #14306b; }
  table { font-size: 22px; }
  section.lead { text-align: center; }
  section.lead h1 { font-size: 52px; line-height: 1.1; }
  .big { font-size: 33px; font-weight: 700; color: #14306b; }
  .mut { color: #667; }
  .small { font-size: 19px; color: #667; }
  .win { color: #137333; font-weight: 800; }
  .bad { color: #b3261e; font-weight: 800; }
  .url { font-family: ui-monospace, monospace; font-size: 22px; color: #2a6df4; }
  .pill { background:#eef3fb; border:2px solid #2a6df4; border-radius:99px; padding:3px 14px; font-size:20px; font-weight:700; color:#14306b; display:inline-block; margin:3px; }
  /* pipeline flow */
  .flow { display:flex; flex-direction:column; align-items:center; gap:3px; }
  .row { display:flex; align-items:center; justify-content:center; gap:9px; flex-wrap:wrap; }
  .box { background:#eef3fb; border:2px solid #2a6df4; border-radius:10px; padding:6px 11px; font-size:17px; font-weight:600; color:#14306b; }
  .box.final { background:#e6f4ea; border-color:#137333; color:#0b5a26; }
  .box.data  { background:#f4f4f4; border-color:#888; color:#333; }
  .box.ens   { background:#fff4e5; border-color:#e08a00; color:#8a5300; }
  /* denser slide for the pipeline + graph */
  section.tight { font-size: 24px; padding: 34px 50px; }
  section.tight h2 { margin: 0 0 8px; }
  section.tight ul { margin: 8px 0; }
  section.tight .big { font-size: 27px; }
  section.tight img { display:block; margin: 6px auto 0; }
  .arrow { font-size:21px; color:#2a6df4; }
  .down  { font-size:21px; color:#2a6df4; line-height:1; }
  .note  { font-size:13px; color:#666; }
---

<!-- _class: lead -->

# Built without a single human label

### An edge person/vehicle detector that runs live — **on this phone**

<span class="pill">📲 open it now</span>

<span class="url">elaybarlev.github.io/CompVision/web/</span>

![w:230](assets/qr_demo.png)

**Elay Barlev** · Intro to Computer Vision · Final Project 2026

![bg right:33%](../outputs/figures/final_demo_1.jpg)

---

<!-- _class: tight -->

## Choices & the pipeline

<div class="flow"><div class="row">
  <span class="box data">Flickr<br/><span class="note">no labels</span></span>
  <span class="arrow">→</span>
  <span class="box">Florence-2 <code>&lt;OD&gt;</code><br/><span class="note">VLM annotator</span></span>
  <span class="arrow">→</span>
  <span class="box ens">+ COCO R-CNN<br/><span class="note">ensemble vote</span></span>
  <span class="arrow">→</span>
  <span class="box data">dataset<br/><span class="note">raw + aug</span></span>
  <span class="arrow">→</span>
  <span class="box final">MobileNet FRCNN<br/><span class="note">72 MB · ONNX · phone</span></span>
</div></div>

- **COCO R-CNN** = a 2nd, COCO-trained detector that **votes on each box** → confirms labels + adds vehicles.
- **8 GB RTX 3080 laptop** → **CUDA + torchvision** · trained **BOTH** detectors, compared on **size + CPU ms**:

<span class="big">FRCNN MobileNet 72 MB · 65 ms&nbsp;&nbsp;|&nbsp;&nbsp;RetinaNet 123 MB · 466 ms</span>

![w:980](../outputs/figures/model_comparison.png)

---

## Edge = the phone in your pocket

<span class="big">No server. No cloud. The model runs on the phone's own CPU.</span>

- **Constraint:** train on 8 GB, but it must *run* on a phone → **size + latency are the score.**
- **Choices:** MobileNet backbone (small + fast) · resolution **switchable live** (512 / 640 / 800 px) in the demo.

**Same model, three resolutions — accuracy vs. on-device speed** (full 690-img val):

| input size | accuracy · mAP@[.5:.95] | precision · mAP@.5 | size | on-device |
|---|---|---|---|---|
| **512 px** (edge fast) | 0.368 | 0.630 | 72 MB | fastest |
| **640 px** (balanced) | 0.426 | 0.697 | 72 MB | ~1.6× cost |
| **800 px** (max accuracy) | <span class="win">0.437</span> | <span class="win">0.700</span> | 72 MB | ~2.4× cost |

<span class="small">Resolution at <b>inference</b> — not more training — is the lever: <b>+0.058 mAP / +0.045 vehicle AP</b> just from 512→640. A dedicated 800 px "all-in" retrain <b>didn't beat</b> it.</span>

![bg right:32%](../outputs/figures/final_demo_2.jpg)

---

## What we learned (the honest part)

- **Raw vs. augmented → basically no effect** (0.703 vs 0.696 — within noise). We report it straight.
- **Data leakage, caught:** two splits made independently → **85% of val had been trained on** → earlier scores were *inflated*. Fix: **one canonical, disjoint split** everywhere.
- **Ensemble earns its keep:** vehicles in val **<span class="bad">98</span> → <span class="win">138</span>** ✅ → clears the per-class minimum.

![bg right:33% w:360](../outputs/figures/compare_ensemble.png)

<span class="small">The mAP plateau is the <b>noisy auto-label ceiling</b>, not the model — better labels is the real lever.</span>

---

## How it runs on a phone — & what's next

<div class="flow"><div class="row">
  <span class="box">📷 camera frame</span><span class="arrow">→</span>
  <span class="box">RGB tensor</span><span class="arrow">→</span>
  <span class="box final">ONNX model<br/><span class="note">normalize+resize+NMS baked in</span></span><span class="arrow">→</span>
  <span class="box">boxes drawn</span>
</div></div>

- **ONNX** ships the *whole* pipeline → the browser just feeds pixels, gets boxes back.
- Runs via **onnxruntime-web (WASM)** — fully **on-device**, no install.

**Future — what else fits on the edge:** int8/fp16 **quantization** · native **NNAPI/GPU** · **tiny, localized models** personalized on-device.

<span class="pill">📲 elaybarlev.github.io/CompVision/web/</span> &nbsp; ![w:120](assets/qr_demo.png)

---

<!-- _class: lead -->

# Thank you

**No human labels → a 72 MB detector that runs live in your hand.**

<span class="small">Code + live demo: github.com/ElayBarLev/CompVision · backup slides follow ↓</span>

---

## Backup — full model comparison

| model | mAP@[.5:.95] | mAP@.5 | size | CPU ms | GPU fps |
|---|---|---|---|---|---|
| **Faster R-CNN MobileNet (FINAL)** | 0.399 | <span class="win">0.703</span> | <span class="win">72 MB</span> | <span class="win">65</span> | 38 |
| Faster R-CNN MobileNet (aug) | 0.397 | 0.696 | 72 MB | 56 | 33 |
| RetinaNet ResNet50 (raw) | 0.404 | 0.673 | <span class="bad">123 MB</span> | <span class="bad">466</span> | 3 |

- Accuracy ~tied; MobileNet wins mAP@.5 **and** is ~7× faster on CPU, 40% smaller → edge winner.

---

## Backup — annotation: `<OD>` vs phrase grounding

| Method (same 60 images) | person | vehicle | quality |
|---|---|---|---|
| `<CAPTION_TO_PHRASE_GROUNDING>` | 117 | 92 | many **false** vehicles |
| **`<OD>`** (generic detection) | 202 | 22 | clean — vehicles only where real |
| **`<OD>` + COCO ensemble** | 295 | 38 | clean **and** higher recall |

Phrase grounding grounds *every* phrase → hallucinates vehicles on indoor scenes. `<OD>` reports only what's there; the COCO ensemble then *adds* the sparse real vehicles.

![bg right:30%](../outputs/figures/compare_grounding.png)

---

## Backup — Why COCO Faster R-CNN as the ensemble partner, not SAM3?

**The brief's "Florence-2 *or* SAM3" picks the *annotator*** — we chose Florence-2 (`docs/02`). The COCO detector is a **different role**: the different-bias second model in the **ensemble cleanup**.

- **Its classes *are* ours** — person / car / bus / truck / bike / motorcycle → a **direct vehicle booster**.
- **WBF needs box scores** — COCO R-CNN emits boxes + scores natively; SAM3 emits **masks** (→ derive boxes), heavier (848 M), newer, Ultralytics-leaning.
- **Same torchvision stack**, cheap on 8 GB, one pass with Florence-2.

> **SAM3 = a legitimate *future-work* partner** (tighter mask boxes, max recall) — deferred for weight/novelty.

---

## Backup — the LR fix

- 30 epochs @ **LR 0.01** → peaked at epoch 3 (mAP@.5 0.66) then **declined** — high LR was degrading pretrained features.
- @ **LR 0.0025** → climbs then **plateaus, no decline.** Plateau = noisy-auto-label ceiling.
- Clean, leakage-free full-val: **mAP 0.437 / mAP@.5 0.700** (690 imgs, 800 px eval).

![w:760](../outputs/figures/fasterrcnn_mobilenet_full_lr0025.png)
