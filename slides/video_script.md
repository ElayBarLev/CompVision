# Individual Video Script — Edge Person/Vehicle Detector

**Target length:** ~4 minutes (aim 3:30–4:30). First-person. Read this while screen-recording.
**What to have open before recording:**
- The slide deck (`slides/final_presentation.md` exported to PDF/HTML).
- A terminal in the project root with the `.venv` active.
- A sample image or two in `data/raw/` for the live inference demo.
- (Optional) the live phone demo URL open on your phone: `https://elaybarlev.github.io/CompVision/web/`

> Tip: the bracketed **[SHOW: …]** notes tell you what to have on screen. Speak naturally —
> these are talking points, not a word-for-word teleprompter.

---

### 0:00 — Intro (≈20s)
> "Hi, I'm Elay. This is my final project for Intro to Computer Vision. The goal was to build a
> small, edge-deployable object detector for two classes — **person** and **vehicle** — but with
> a twist: **no human labels**. The entire training set is annotated automatically by a large
> vision-language model. I'll walk through how I created the dataset, trained the detector,
> and what I learned."

**[SHOW: title slide]**

---

### 0:20 — Stage 1: auto-annotation & the main challenge (≈45s)
> "Stage one is labeling the Flickr dataset with **Florence-2** running in inference mode.
> My first attempt used phrase-grounding — I gave it a list like 'person, car, bicycle' — but it
> **hallucinated vehicles**: it drew vehicle boxes on indoor scenes and portraits, because
> grounding tries to locate *every* phrase whether or not the object is there. That noise would
> directly poison the model I'm training."
>
> "The fix was switching to Florence's generic **`<OD>`** task, which only reports objects it
> actually finds. Clean person boxes, realistic vehicles."

**[SHOW: `compare_grounding.png` vs `compare_od.png` — the annotation slide]**

---

### 1:05 — The dataset gap & the ensemble (≈40s)
> "But there was a catch: vehicles are genuinely rare in this dataset, so I ended up with only
> **98 vehicle images** in validation — just short of the required 100 per class. I couldn't
> invent labels, so I added a **second model** — a COCO-pretrained Faster R-CNN that's strong on
> vehicles — and fused the two with **Weighted Boxes Fusion**. The important detail is that a box
> found by only one model isn't suppressed, so the COCO model *contributes* real vehicles. That
> pushed me to 138 vehicle val images — requirement met — and it's also a concrete example of
> using an **ensemble to verify and improve** labels when you have no ground truth."

**[SHOW: ensemble slide with the counts table]**

---

### 1:45 — Stage 2: training both detectors (≈30s)
> "Stage two: I trained **both** a two-stage Faster R-CNN and a one-stage RetinaNet on the exact
> same data — mixed precision to fit my 8 GB laptop GPU, cosine learning-rate schedule, and I
> logged mAP plus edge metrics for every run."

**[SHOW: training-curve slide, `fasterrcnn_mobilenet_full_lr0025.png`]**

---

### 2:15 — Live inference demo (≈40s)
> "Here it is actually running. I'll point it at a sample image."

**[DO: run this in the terminal and show the output image]**
```
.venv/Scripts/python.exe src/infer/infer_image.py --weights weights/FINAL_fasterrcnn_mobilenet.pt --image data/raw/<your-sample>.jpg --out outputs/figures/demo.jpg
```
> "It loads the final weights, runs detection, and draws person boxes in red and vehicle boxes
> in blue. There's also a folder-inference script that does this over a whole directory and dumps
> a predictions JSON."

**[SHOW: the saved `outputs/figures/demo.jpg`]**

---

### 2:55 — Results & the model choice (≈40s)
> "On accuracy the two detectors are basically tied — but for an **edge** target it isn't close.
> The MobileNet Faster R-CNN actually wins mAP-at-0.5, at **0.70**, while being **~7× faster on
> CPU** and **40% smaller** than RetinaNet. So my final model is the MobileNet Faster R-CNN:
> about **72 megabytes**, 65 milliseconds per image on CPU, mAP-at-0.5 of 0.70."
>
> "One honest note: augmentation came out **neutral** here — I'm reporting that rather than
> overclaiming a gain that isn't there."

**[SHOW: model-comparison table slide]**

---

### 3:35 — The lessons (≈30s)
> "Two things I'm glad I caught. First, a training plateau that looked like overfitting was
> actually a **learning rate that was too high** — dropping it from 0.01 to 0.0025 fixed it, and
> the remaining plateau is just the ceiling of noisy auto-labels. Second, I caught **data
> leakage**: two experiments were split independently, so most of my validation images had been
> in the other model's training set. The fix was one canonical, disjoint split everywhere."

**[SHOW: 'lessons' slide]**

---

### 4:05 — Bonus phone demo & close (≈25s)
> "As a bonus, the model is exported to ONNX and runs **fully on a phone**, in the browser, via
> WebAssembly — no server. Here it is detecting live."

**[SHOW: phone demo if available, or `final_demo_1.jpg`]**
> "Everything — the dataset code, training, inference, and this live demo — is in the repo.
> The biggest takeaway: with no human labels at all, the quality of the auto-annotation is the
> real ceiling, and a good ensemble is what raises it. Thanks for watching."

**[SHOW: closing slide]**

---

## Quick pre-flight checklist
- [ ] `.venv` active; `infer_image.py` runs and produces a clean demo image.
- [ ] Slide PDF open and ready to page through.
- [ ] Phone demo URL loads (or fallback screenshot ready).
- [ ] Screen recorder capturing the right window + mic working.
- [ ] Keep it under ~5 minutes.
