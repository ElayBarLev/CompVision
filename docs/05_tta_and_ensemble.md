# Decision Doc — TTA & Ensembles for Annotation Quality

The brief requires a **written explanation** of how **TTA** (Test-Time Augmentation) and
**ensembles** could verify the correctness of our auto-generated annotations (implementation
*not* required). We provide that write-up here **and** implement both as a bonus
(`src/tta_ensemble/`) to *show* measurable improvement on the slides.

---

## Part A — Required write-up

### The core problem
Florence-2's labels are produced by a model, so some are wrong: missed objects (false
negatives), hallucinated boxes (false positives), or loose/duplicate boxes. We have **no
ground truth** to check against. TTA and ensembles give us a *proxy for confidence*:
**predictions that survive perturbation or that multiple models agree on are more trustworthy.**

### How TTA verifies annotation correctness
Run the annotator on **several transformed versions of the same image** — e.g. horizontal
flip, a couple of scales, mild brightness/contrast changes — then map each prediction back to
the original image coordinates and compare.

- A box that appears **consistently** across transforms (high agreement, measured by IoU) is
  likely **correct** → keep it, optionally with a confidence = fraction of views it appeared in.
- A box that appears in only **one** view is likely **spurious** → flag/drop it.
- An object found only in *some* views (e.g. only after upscaling) hints at a **small-object
  miss** → flag for review or for relaxing the size filter.

We combine the multiple views with **Weighted Boxes Fusion (WBF)** to produce a single,
cleaner set of boxes and a per-box agreement score.

### How ensembles verify annotation correctness
Annotate the **same images with two (or more) different models** — e.g. Florence-2 **and**
a second detector (an off-the-shelf COCO model, or SAM3). Then:

- Boxes where models **agree** (IoU above a threshold) are high-confidence → trust them.
- Boxes only **one** model produces are uncertain → flag as likely FP, or as a recall gap of
  the other model.
- Disagreement on **class** (person vs vehicle) flags ambiguous/occluded cases for review.

Ensembling complements TTA: TTA tests robustness of *one* model to input changes; ensembles
test robustness across *different model biases*. Both turn "no ground truth" into a usable
agreement signal, and WBF merges everyone's boxes into one consensus annotation set.

### Why this improves the dataset (and the slide takeaway)
Cleaner labels → the small model learns from less noise → higher, more honest mAP. We can
quantify it: compare val mAP of a model trained on **raw** Florence-2 labels vs. on
**TTA/ensemble-cleaned** labels.

---

## Part B — Our bonus implementation (plan)
- `src/tta_ensemble/tta_annotate.py` — annotate with flips/scales, back-transform boxes,
  fuse with WBF (`ensemble-boxes`), output per-box agreement scores.
- `src/tta_ensemble/ensemble_annotate.py` — Florence-2 + a second model, fuse with WBF.
- Produce a small **before/after** comparison (box counts, a few example images, and val mAP
  of models trained on raw vs. cleaned labels) for the presentation.

> Keep clearly labeled as **bonus** so graders see we (a) met the write-up requirement and
> (b) went further.
