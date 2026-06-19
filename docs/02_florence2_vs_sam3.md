# Decision Doc — Florence-2 vs. SAM3 (the auto-annotator)

> **Role of this model:** it is used only in **inference mode** to label the Flickr images.
> It is *not* the model we train or deploy. So what matters is: does it produce **good
> bounding boxes for "person" and "vehicle"** cheaply on our 8 GB GPU?

## The two candidates

### Florence-2 (Microsoft, mid-2024)
- A vision-language model (VLM). You give it an image + a task token and it returns text.
- Relevant task tokens:
  - `<OD>` — generic object detection (returns boxes + class names, ~34.7 mAP on COCO zero-shot).
  - `<CAPTION_TO_PHRASE_GROUNDING>` — give it a phrase like *"person. car. bicycle. motorcycle."*
    and it returns boxes for exactly those concepts. **This is the one we want** (open-vocabulary,
    we control the label set).
- **Output is bounding boxes directly** — exactly the annotation format we need.
- Sizes: **base ≈ 0.23 B params**, large ≈ 0.77 B. Inference fits easily in <2 GB VRAM
  (base measured ≈ 0.86 GB). Comfortable on our 8 GB card.
- Mature: out since 2024, lots of tutorials, and **the course provides a Florence-2 Colab baseline.**
- Pure HuggingFace `transformers` + PyTorch → aligns with our torchvision/PyTorch comfort zone.

### SAM3 (Meta, released 2025-11-19)
- "Segment Anything with Concepts." Detects + **segments** + tracks from **text prompts**
  ("yellow school bus") or visual prompts (points/boxes/masks).
- Strength: exhaustively finds **all instances** of an open-vocabulary concept; great at
  pixel-accurate masks and video tracking.
- **Output is segmentation masks** (+ boxes). For our task we'd take the mask and reduce it
  to its bounding box — an extra step, though box tightness from masks is usually excellent.
- **848 M params** — heavier; brand new (≈7 months old as of this project), so fewer
  tutorials and rougher tooling. Steered toward the **Ultralytics** ecosystem, not torchvision.
- Inference still fits on 8 GB, but it's a bigger, newer, less-documented dependency.

## Side-by-side

| Criterion | Florence-2 | SAM3 |
|---|---|---|
| Native output for our task | **Bounding boxes** ✅ | Masks → derive boxes (extra step) |
| Open-vocab text prompting | ✅ (`CAPTION_TO_PHRASE_GROUNDING`) | ✅ (its headline feature) |
| Size / VRAM for inference | 0.23 B, ~1 GB ✅ | 0.85 B, heavier |
| Maturity / docs / examples | High; **course Colab provided** ✅ | New (Nov 2025), thinner docs |
| Fits torchvision/PyTorch stack | ✅ (HF transformers) | Leans Ultralytics |
| Best at | Direct detection labels | Pixel-perfect masks, "find all", video |
| Annotation speed on Flickr | Fast (small model) ✅ | Slower (bigger model) |

## Recommendation → **Florence-2 (base, with phrase grounding)**

Reasoning:
1. **We need boxes, and Florence-2 emits boxes** — no mask→box conversion, fewer failure modes.
2. **Lightest path on 8 GB** — leaves headroom and annotates the dataset faster.
3. **Lowest project risk** — mature, documented, and the course gives us a baseline notebook.
4. **Stack fit** — plain PyTorch/HF, consistent with our torchvision training code.

**When SAM3 would win:** if we needed segmentation masks, video tracking, or maximal recall
of *every* instance of a concept. For tight person/vehicle boxes on still images, that
power is overkill and costs us speed, VRAM, and tooling maturity.

> A nice presentation angle: we can still *mention* SAM3 as the "mask-based alternative"
> and note that mask-derived boxes can be tighter — a good "future work" bullet.

## Final decision
- **Chosen:** _[pending user confirmation]_
- **Date:** _[fill in]_
- **Rationale (one line for the slide):** _[fill in]_

## Sources
- [Meta AI — SAM 3 announcement](https://ai.meta.com/blog/segment-anything-model-3/)
- [MarkTechPost — SAM 3 release (848M params, text prompts)](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/)
- [Roboflow — What is SAM 3](https://blog.roboflow.com/what-is-sam3/)
- [Roboflow — Florence-2 object detection](https://roboflow.com/model/florence-2)
- [Roboflow — Fine-tune Florence-2 for detection](https://blog.roboflow.com/fine-tune-florence-2-object-detection/)
- [TDS — Florence-2 overview](https://medium.com/data-science/florence-2-mastering-multiple-vision-tasks-with-a-single-vlm-model-435d251976d0)
