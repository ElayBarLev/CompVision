# Project Brief (our summary of the assignment)

Course 10224 — Introduction to Computer Vision, Final Project 2026.

## The story
Real CV projects usually start with *no labeled data*. This project makes us live that:
we use a big pretrained model to **auto-label** images, then train a **small** model that
could run on an edge device (Jetson / Raspberry Pi) for object detection.

## Pipeline
1. **Create the training dataset (semi-supervised).**
   - Large model (Florence-2 **or** SAM3) in *inference* mode + the **Flickr image dataset**
     → bounding boxes. No human annotation.
   - Classes:
     - **Person** — woman, man, boy, girl, child, …
     - **Vehicle** — car, bike, motorcycle.
   - Size requirement: **≥ 500 training + 100 validation images per class.**
   - We have *degrees of freedom* on min/max object size and min/max image size → must
     explain our choices (see `04_degrees_of_freedom.md`).
   - Must *write up* how **TTA** and **ensembles** could check annotation correctness
     (implementation not required — but we will do it as a bonus; see `05_tta_and_ensemble.md`).

2. **Train the model.**
   - Pick an architecture and justify it; say why we rejected the others
     (Faster R-CNN, RetinaNet, YOLO family, EfficientDet, MobileNetV4, other).
     → see `03_model_choice.md`.
   - Framework: we use **Torchvision / native PyTorch**.
   - Train on our custom dataset, report performance with **graphs + metrics (mAP)**.
   - Apply augmentations and re-train; compare **raw vs. augmented**.
   - Explain thought process + insights.

3. **Present** (page 4 of the brief — handled at the end).
   - 5 min (solo) / 7 min (pair) slide deck; structure: problem → dataset creation →
     model training (raw vs augmented, mAP, graphs) → future work → reflection.
   - Each team member records an **individual** video.

## Deliverables checklist
- [ ] Presentation (PPTX)
- [ ] Individual video per team member
- [ ] Trained model weights
- [ ] Training source code
- [ ] Dataset-creation source code
- [ ] Inference: single image **and** folder of images
- [ ] (Optional) be ready for a staff interview about the work

## Our chosen path (locked-in choices)
- Framework: **Torchvision / native PyTorch**.
- Detectors: train **both Faster R-CNN and RetinaNet**, then choose with evidence.
- Annotator: **Florence-2 vs SAM3** — pending (see `02_florence2_vs_sam3.md`).
- Hardware: local **RTX 3080 Laptop (8 GB)**.
- Bonus: implement **TTA + ensemble** to demonstrate measurable gains.
