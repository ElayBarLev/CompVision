# Decisions Log

A running record of every meaningful choice and *why*, so we can explain it on the slides
and to each other. Newest at the bottom.

| # | Date | Decision | Why | Doc |
|---|------|----------|-----|-----|
| 1 | 2026-06-19 | Framework = **Torchvision / native PyTorch** | User's experience; a fundamental, transferable ML skill; gives full control over the detector head | — |
| 2 | 2026-06-19 | Train **both Faster R-CNN and RetinaNet** first | Decide empirically (mAP, speed, size) once we have data, rather than guessing | `03_model_choice.md` |
| 3 | 2026-06-19 | Hardware = local **RTX 3080 Laptop, 8 GB** | Available GPU; drives batch-size / mixed-precision constraints | — |
| 4 | 2026-06-19 | Dataset = Kaggle **flickr-image-dataset** via kagglehub | Specified by user; large, varied real-world photos with people + vehicles | — |
| 5 | 2026-06-19 | Implement **TTA + ensemble** (beyond the required write-up) | To *show* measurable annotation/detection improvement on slides | `05_tta_and_ensemble.md` |
| 6 | 2026-06-19 | Annotator = **Florence-2** (base) | Emits boxes directly; lightest on 8 GB; mature + course baseline; fits PyTorch stack. SAM3's masks/video power is overkill | `02_florence2_vs_sam3.md` |
| 7 | 2026-06-19 | Annotation task = **`<OD>`**, NOT `<CAPTION_TO_PHRASE_GROUNDING>` | Visual+count comparison on 60 imgs: grounding *hallucinated* ~4× vehicles (forces every phrase to a region) → would poison training. `<OD>` reports only what it detects: precise person boxes, vehicles only where real | `02_florence2_vs_sam3.md`, `07_annotation_method.md` |
| 8 | 2026-06-19 | Final labels = **OD + ensemble cleanup** (Florence `<OD>` + COCO Faster R-CNN, fused via WBF `conf_type=max`) | COCO detector is strong on vehicles → recovers real vehicles OD missed (+52% on sample) while staying precise. We keep the OD-only set too, to quantify the gain | `05_tta_and_ensemble.md`, `07_annotation_method.md` |
| 9 | 2026-06-19 | Annotate **5000 images** | Flickr30k is people-centric (~28% have vehicles); 5000 imgs comfortably clears 500 train/100 val for the sparse vehicle class and gives a stronger model | `04_degrees_of_freedom.md` |
| 10 | 2026-06-20 | **Ensemble was required, not just nicer** | After the 5000-img run: OD-only gave vehicle val=98 (**SHORT** of 100); ensemble gave 138 (**OK**). The ensemble cleanup is what lets us meet the per-class minimum | `07_annotation_method.md` |
| 11 | 2026-06-20 | First-pass training on a **750/class subset** (1354 imgs), **15 epochs** | Fast iteration to pick the winning arch + see raw-vs-aug; we retrain the winner on the full dataset with tuned epochs later | `03_model_choice.md` |
| 12 | 2026-06-20 | **Faster R-CNN trained without AMP**; RetinaNet with AMP. **batch 16, 512px** | Measured: Faster R-CNN + autocast = 15 s/batch (pathological box-op fp16 path); no-AMP = 0.43 s/batch. RetinaNet+AMP fine. batch 16 ≈ 6 GB (best throughput; 24 OOMs). 512px matches the ~500px Flickr images (640 upscaled them) | — |
| 13 | 2026-06-20 | **Final model = Faster R-CNN MobileNetV3-FPN** | Tied/better accuracy than RetinaNet (mAP@.5 0.70 vs 0.67) at ~1/8 CPU latency (77 vs 466 ms) and 60% the size (72 vs 123 MB) → the edge winner. Augmentation was neutral (FRCNN) / caused underfit (RetinaNet, needs more epochs) | `03_model_choice.md` |
| 14 | 2026-06-20 | ~~Training longer overfits~~ → **root cause was LR too high** | Full-data 30-ep run (LR 0.01) peaked at epoch 3 (mAP@.5 0.66) then *declined* to 0.59. Re-run at **LR 0.0025** climbs then **plateaus ~0.655 (no decline)**. So the decline was LR degrading pretrained features, not pure overfitting. Plateau ≈ the noisy-label ceiling | `03_model_choice.md` |
| 15 | 2026-06-20 | **Caught data leakage** in the subset-vs-full comparison | The subset and full datasets were split independently → **589/690 (85%) of full-val images were in the subset's TRAIN set**, inflating the subset model to a fake 0.464. Lesson: keep one canonical split / disjoint val across experiments | `03_model_choice.md` |
| 16 | 2026-06-20 | **Final model = `fasterrcnn_mobilenet_full_lr0025`** | Leakage-free best on full val @800px: **0.437 mAP / 0.700 mAP@.5** (vs full/30ep/LR0.01 = 0.427/0.699). MobileNet edge-fitness unchanged (72 MB, ~73 ms CPU) | `03_model_choice.md` |

## How to use this file
When we make a call, add a row. Keep the "Why" short enough to drop onto a slide.
