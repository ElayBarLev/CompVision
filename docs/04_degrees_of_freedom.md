# Decision Doc — Degrees of Freedom (object & image sizes)

The brief gives us freedom over **min/max object size** and **min/max image size**, and asks
us to **explain our decisions**. These choices directly shape annotation quality and how well
the small edge model learns. Values live in `src/utils/config.py`.

## Why these knobs matter
- The annotator (Florence-2) is *not perfect*. Its worst, noisiest outputs tend to be
  **very tiny boxes** (spurious) and occasional **whole-image boxes** (useless as detections).
- Our target is an **edge model**. Teaching it to chase 8×8-pixel specks wastes capacity and
  hurts precision; it should learn the clear, mid-to-large objects an edge camera actually sees.
- Image size affects both Florence-2's accuracy (it works around ~768 px internally) and our
  8 GB VRAM budget.

## Our settings (defaults) and rationale

| Knob | Value | Why |
|---|---|---|
| `MIN_IMAGE_SIZE` (shorter side) | **320 px** | Below this, photos are thumbnails — too little detail for trustworthy auto-labels. |
| `MAX_IMAGE_SIZE` (longer side) | **2048 px** | Cap memory; we downscale larger images before annotating (Florence-2 doesn't benefit from huge inputs). |
| `MIN_BOX_AREA_FRAC` | **0.5%** of image area | Removes tiny, often-spurious boxes; keeps labels the edge model can realistically detect. |
| `MAX_BOX_AREA_FRAC` | **95%** of image area | Drops near-full-frame boxes that aren't useful "detections" (e.g. a wall-to-wall crowd). |

Flickr30k photos are mostly normal-resolution everyday scenes, so these thresholds keep the
vast majority of images while trimming the unreliable tails.

## Trade-offs to mention on the slide
- **Stricter min size → cleaner labels but fewer boxes.** Since we need ≥500 train + 100 val
  images *per class*, we balance cleanliness against having enough data. If a class runs short,
  we relax `MIN_BOX_AREA_FRAC` first.
- These are **tunable**: if validation mAP suggests we're starving the model of small objects
  (or feeding it junk), we adjust and document the change in `decisions_log.md`.

## Final decision
- **Locked values:** as in the table above (defaults in `config.py`).
- **Revisit if:** a class can't reach the 500/100 minimum, or val mAP shows a clear small-object
  problem.
