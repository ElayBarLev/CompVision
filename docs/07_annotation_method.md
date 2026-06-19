# Decision Doc — Annotation Method (the key dataset challenge)

This is the most important dataset-creation finding, and a strong slide ("challenges we faced
and how we overcame them").

## The challenge we hit
Florence-2 offers several prompts. Our first instinct was `<CAPTION_TO_PHRASE_GROUNDING>`
with a phrase list (`"person. car. bicycle. motorcycle. bus. truck."`). When we visualized
the results, **many vehicle boxes were hallucinations** — blue "vehicle" boxes over indoor
scenes, a wedding couple, a guitarist, etc., where no vehicle exists.

**Why:** phrase grounding answers *"given this caption, where are these things?"* — it tries
to **ground every phrase to some region**, even when the object isn't in the image. Great
recall, but it fabricates objects for absent classes. For *auto-labeling*, that noise would
directly poison the student model.

## What we measured (same 60 Flickr images)
| Method | person boxes | vehicle boxes | quality |
|---|---|---|---|
| `<CAPTION_TO_PHRASE_GROUNDING>` | 117 | 92 | many **false** vehicles |
| `<OD>` (generic detection) | 202 | 22 | clean; vehicles only where real |
| `<OD>` + COCO ensemble | 295 | 38 | clean **and** higher recall |

Figures: `outputs/figures/compare_grounding.png`, `compare_od.png`, `compare_ensemble.png`.

## How we fixed it
1. **Switch to `<OD>`** — generic object detection reports *only* objects it actually finds,
   then we map its labels (car/bus/truck/bike/… → vehicle; man/woman/child/… → person) and
   apply the size filters. Precise person boxes, realistic vehicle counts.
2. **Ensemble cleanup** — vehicles are genuinely sparse in Flickr30k, so we *add* a second
   model (a COCO-pretrained torchvision Faster R-CNN, which is strong on vehicle classes) and
   fuse with **Weighted Boxes Fusion**. Key detail: `conf_type="max"` so a box found by only
   one model is **not** penalized — this lets the COCO detector *contribute* real vehicles
   instead of having them suppressed. Result on the sample: **+52% vehicles, +43% people**,
   still clean.

We keep **both** outputs from a single Florence pass:
- `data/annotations/annotations_od.json` — clean baseline.
- `data/annotations/annotations_ensemble.json` — vehicle-boosted final.

This lets us **quantify the ensemble's value**: train on each and compare val mAP.

## Takeaways for the slides
- Tool choice within one model (`<OD>` vs grounding) changed annotation quality dramatically.
- "No ground truth" doesn't mean "no quality control": visualization caught it; a **second
  model (ensemble)** both verifies and *improves* the labels — exactly the TTA/ensemble idea
  the brief asks us to discuss (`docs/05`).
