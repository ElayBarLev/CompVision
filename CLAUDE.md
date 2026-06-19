# CLAUDE.md — Project Guide for Claude Code

## What this is
**Course:** 10224 Introduction to Computer Vision — Final Project 2026.
**Goal:** Semi-supervised dataset creation + training a *small, edge-deployable* object detector.

Pipeline in three stages:
1. **Auto-annotate** the Flickr image dataset with a large VLM (Florence-2 or SAM3) in
   inference mode → bounding boxes for **Person** and **Vehicle** classes (no human labels).
2. **Train a small detector** (torchvision: Faster R-CNN and RetinaNet) on that dataset.
   Compare **raw vs. augmented** training; report **mAP** + training graphs.
3. **Present** results (slides + individual video).

## Hard requirements (from the brief)
- Classes: **Person** (woman/man/boy/girl/child…) and **Vehicle** (car/bike/motorcycle).
- Dataset: **≥ 500 train + 100 val images per class.**
- Deliverables: trained weights, training code, dataset-creation code, single-image
  inference code, folder inference code, slides, individual videos.
- TTA & ensembles: the brief only requires a **written explanation** of how they'd verify
  annotation quality. We additionally **implement** them to show measurable improvement
  (bonus — keep clearly separated from the required write-up).

## Decisions already made by the user
- **Framework:** Torchvision / native PyTorch (the user's experience + a fundamental ML skill).
- **Detectors:** Train **both Faster R-CNN and RetinaNet** first, decide later with data.
- **Hardware:** Local training on an **NVIDIA RTX 3080 Laptop, 8 GB VRAM**, Windows 11.
- **Dataset download:** `kagglehub.dataset_download("hsankesara/flickr-image-dataset")`.

## Open decisions (tracked in docs/decisions_log.md)
- Annotator model: **Florence-2 vs SAM3** — see `docs/02_florence2_vs_sam3.md`.
- Degrees of freedom (object/image min/max sizes) — see `docs/04_degrees_of_freedom.md`.

## Working style for this project
- The user wants to **learn and make the decisions himself**, then explain to a partner.
  → For every non-trivial choice, write a clear, standalone doc in `docs/` (the "side
    documentation") with: the options, trade-offs, a recommendation, and the final call.
  → Prefer explaining *why* over just doing. Surface decisions; don't silently pick.
- 8 GB VRAM is the main constraint. Keep batch sizes small, use AMP/mixed precision,
  prefer the lighter model variants for inference, and free GPU memory between stages.

## Repo layout
```
docs/            Side documentation: research, decisions, write-ups for the partner
src/dataset/     download.py, annotate.py, build_dataset.py  (stage 1)
src/train/       train_fasterrcnn.py, train_retinanet.py, common train loop (stage 2)
src/infer/       infer_image.py, infer_folder.py             (deliverable)
src/tta_ensemble/ TTA + ensemble experiments                 (bonus)
src/utils/       shared helpers (coco, viz, metrics, gpu)
data/raw/        Flickr images (gitignored)
data/annotations/ auto-generated labels (gitignored)
data/processed/  train/val splits in COCO format (gitignored)
weights/         model checkpoints (gitignored)
outputs/         figures + metrics for the report (gitignored)
notebooks/       exploration / the provided Florence-2 / SAM3 baselines
```

## Environment notes
- Python 3.12.10. PyTorch installed with a CUDA wheel (see `requirements.txt` / README).
- Always verify `torch.cuda.is_available()` before training.
- Use the `Bash` tool for POSIX scripts; PowerShell is the primary shell on this machine.
