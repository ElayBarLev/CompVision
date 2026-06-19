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

## How to use this file
When we make a call, add a row. Keep the "Why" short enough to drop onto a slide.
