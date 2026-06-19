"""Stage 1b — Auto-annotate Flickr images with Florence-2 (phrase grounding).

We run Florence-2 in inference mode with the `<CAPTION_TO_PHRASE_GROUNDING>` task and a
fixed phrase prompt (see config.GROUNDING_PROMPT). Florence-2 returns boxes + labels;
we map those labels to our two classes (person / vehicle) and apply the degrees-of-freedom
size filters.

Output: one COCO-style JSON of *all* accepted annotations (data/annotations/annotations.json),
which build_dataset.py then filters and splits into train/val.

Usage:
    python src/dataset/annotate.py --limit 2000          # annotate first N images
    python src/dataset/annotate.py --model microsoft/Florence-2-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (  # noqa: E402
    CLASSES, CLASS_TO_ID, LABEL_MAP, GROUNDING_PROMPT,
    MIN_IMAGE_SIZE, MAX_IMAGE_SIZE, MIN_BOX_AREA_FRAC, MAX_BOX_AREA_FRAC,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANN_DIR = PROJECT_ROOT / "data" / "annotations"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


def load_florence(model_id: str, device: str):
    """Load Florence-2 model + processor from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {model_id} (dtype={dtype}, device={device})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, dtype


@torch.inference_mode()
def detect(model, processor, dtype, device, image: Image.Image):
    """Run Florence-2 phrase grounding -> list of (bbox_xyxy, label_text)."""
    inputs = processor(text=TASK + GROUNDING_PROMPT, images=image, return_tensors="pt")
    inputs = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
              for k, v in inputs.items()}
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=TASK, image_size=(image.width, image.height)
    )
    result = parsed.get(TASK, {})
    boxes = result.get("bboxes", [])
    labels = result.get("labels", [])
    return list(zip(boxes, labels))


def map_label(raw_label: str) -> str | None:
    """Map a Florence-2 phrase to one of our classes, or None to drop it."""
    key = raw_label.strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    # try word-level match (e.g. "a red car" -> car)
    for word in key.replace(".", " ").split():
        if word in LABEL_MAP:
            return LABEL_MAP[word]
    return None


def passes_filters(bbox, img_w, img_h) -> bool:
    """Apply degrees-of-freedom box-size filters."""
    x1, y1, x2, y2 = bbox
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    if w <= 1 or h <= 1:
        return False
    frac = (w * h) / float(img_w * img_h)
    return MIN_BOX_AREA_FRAC <= frac <= MAX_BOX_AREA_FRAC


IMG_EXTS = (".jpg", ".jpeg", ".png")


def iter_images(images_root: Path, limit: int | None):
    # NON-recursive on purpose: this dataset duplicates the images across nested
    # flickr30k_images/ folders, so recursing would process each image twice.
    count = 0
    for p in sorted(images_root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p
            count += 1
            if limit and count >= limit:
                return


def _best_image_dir(base: Path) -> Path | None:
    """Return the directory holding the most images DIRECTLY (shallowest on ties).
    Handles the Flickr30k nested-duplicate layout cleanly."""
    import os
    best = None  # (count, -depth, path)
    for dp, _dn, fn in os.walk(base):
        n = sum(1 for f in fn if f.lower().endswith(IMG_EXTS))
        if n:
            depth = len(Path(dp).relative_to(base).parts)
            key = (n, -depth)
            if best is None or key > best[0]:
                best = (key, Path(dp))
    return best[1] if best else None


def resolve_images_root() -> Path:
    """Find the single directory that directly contains the Flickr images."""
    pointer = RAW_DIR / "DATASET_PATH.txt"
    base = None
    if pointer.exists():
        base = Path(pointer.read_text(encoding="utf-8").strip())
    elif any(RAW_DIR.rglob("*.jpg")):
        base = RAW_DIR
    if base is None:
        raise SystemExit("No dataset found. Run src/dataset/download.py first.")
    img_dir = _best_image_dir(base)
    if img_dir is None:
        raise SystemExit(f"No images found under {base}.")
    return img_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Florence-2-base")
    ap.add_argument("--limit", type=int, default=None, help="max images to annotate")
    ap.add_argument("--out", default=str(ANN_DIR / "annotations.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available — annotation will be slow.")

    images_root = resolve_images_root()
    print(f"Images root: {images_root}")
    model, processor, dtype = load_florence(args.model, device)

    # COCO-style scaffold
    coco = {
        "info": {"description": "Flickr auto-annotated by Florence-2", "model": args.model},
        "images": [],
        "annotations": [],
        "categories": [{"id": CLASS_TO_ID[c], "name": c} for c in CLASSES],
    }
    ann_id = 0
    kept_per_class = {c: 0 for c in CLASSES}

    for img_id, img_path in enumerate(iter_images(images_root, args.limit)):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  skip (unreadable): {img_path.name} ({e})")
            continue

        w, h = image.width, image.height
        if min(w, h) < MIN_IMAGE_SIZE:
            continue
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)))
            w, h = image.width, image.height

        dets = detect(model, processor, dtype, device, image)
        img_anns = []
        for bbox, raw_label in dets:
            cls = map_label(raw_label)
            if cls is None or not passes_filters(bbox, w, h):
                continue
            x1, y1, x2, y2 = bbox
            img_anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_TO_ID[cls],
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO xywh
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1
            kept_per_class[cls] += 1

        if img_anns:
            coco["images"].append({
                "id": img_id, "file_name": str(img_path), "width": w, "height": h,
            })
            coco["annotations"].extend(img_anns)

        if (img_id + 1) % 50 == 0:
            print(f"  processed {img_id + 1} images | kept {kept_per_class}")

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(coco), encoding="utf-8")
    print(f"\nDone. Images with annotations: {len(coco['images'])}, "
          f"boxes: {len(coco['annotations'])}, per-class: {kept_per_class}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
