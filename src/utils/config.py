"""Central configuration: class definitions, label mapping, and the dataset
"degrees of freedom" (object/image size filters).

Keeping these in one place means every stage (annotate -> build -> train -> infer)
agrees on class ids, and our DoF choices are documented in a single spot.
See docs/04_degrees_of_freedom.md for the reasoning behind the numbers.
"""
from __future__ import annotations

# --- Classes the project requires (COCO-style: background is implicit id 0 in torchvision) ---
# We collapse many fine-grained things the annotator might say into just two classes.
CLASSES = ["person", "vehicle"]            # index 0,1 in our label space
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# Florence-2 (open vocabulary) may return any of these phrases; map them to our 2 classes.
# Keys are matched case-insensitively against the returned label text.
LABEL_MAP = {
    # --- person ---
    "person": "person", "people": "person", "man": "person", "woman": "person",
    "boy": "person", "girl": "person", "child": "person", "kid": "person",
    "pedestrian": "person", "human": "person",
    # --- vehicle ---
    "car": "vehicle", "vehicle": "vehicle", "truck": "vehicle", "van": "vehicle",
    "bus": "vehicle", "bike": "vehicle", "bicycle": "vehicle",
    "motorcycle": "vehicle", "motorbike": "vehicle", "scooter": "vehicle",
}

# The text prompt we feed Florence-2's phrase-grounding task. Broad enough to catch
# our concepts, narrow enough to avoid junk.
GROUNDING_PROMPT = "person. car. bicycle. motorcycle. bus. truck."

# --- Degrees of freedom (filters). These are DEFAULTS; tune and document any change. ---
# Image size: skip tiny thumbnails and cap huge images (memory + Florence-2 works at ~768px).
MIN_IMAGE_SIZE = 320          # px, shorter side; below this, detail is too poor to trust
MAX_IMAGE_SIZE = 2048         # px, longer side; above this we downscale before annotating

# Object size, as a FRACTION of image area:
#  - too-small boxes are often noisy/false annotations and bad for a small edge model
#  - too-large boxes (whole-image) are usually not useful detections
MIN_BOX_AREA_FRAC = 0.005     # 0.5% of image area
MAX_BOX_AREA_FRAC = 0.95      # 95% of image area

# Confidence-like filtering: Florence-2 grounding doesn't give scores, so we rely on
# the size filters above + (optionally) TTA/ensemble agreement (see docs/05).

# --- Dataset split minimums (the brief's hard requirement, per class) ---
MIN_TRAIN_PER_CLASS = 500
MIN_VAL_PER_CLASS = 100

# --- Annotation/fusion tuning (see docs/05). Centralised so the values are explained once. ---
PROGRESS_REPORT_INTERVAL = 50      # log every N images during long annotation runs
TTA_SCALE_FACTOR = 1.3             # "up" view zoom for TTA (boxes re-normalised, so scale-free)
WBF_TTA_IOU_THR = 0.5              # WBF IoU for fusing TTA views of the SAME model
WBF_ENSEMBLE_IOU_THR = 0.55        # WBF IoU for fusing Florence + COCO detector (different biases)
WBF_ENSEMBLE_CONF_TYPE = "max"     # don't penalise single-model boxes -> COCO can ADD vehicles
COCO_DETECTOR_SCORE_THR = 0.3      # min score to keep a torchvision-COCO detection in the ensemble
