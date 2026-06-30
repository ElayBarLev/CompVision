# Runbook — End-to-End Pipeline

Run from the project root with the venv active. Each step says roughly how long it takes
and what it produces.

## 0. One-time setup
```bash
# venv + GPU PyTorch (cu126) + the rest
python -m venv .venv && source .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Put Kaggle creds at `~/.kaggle/kaggle.json` (Kaggle → Settings → Create New Token).

> **Gotchas (already handled in requirements.txt, noted here so we understand them):**
> - **transformers must be `<4.50`** — Florence-2's `trust_remote_code` files break on
>   transformers 5.x (`forced_bos_token_id` error). We pin to 4.49.
> - On Windows you may see a HuggingFace **symlinks warning** — harmless (uses a bit more
>   disk). Silence it by enabling Developer Mode or setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
>
> Verify the environment any time with: `python src/utils/smoke_test.py`

## Stage 1 — Create the dataset
```bash
# 1a. Download Flickr images (~9 GB, one time)
python src/dataset/download.py

# 1b. Auto-annotate with Florence-2. Start SMALL to sanity-check, then scale up.
python src/dataset/annotate.py --limit 200          # quick test
python src/dataset/annotate.py --limit 4000         # enough to clear 500/100 per class
#   -> data/annotations/annotations.json

# 1c. Split into train/val and verify the >=500 train / >=100 val per-class requirement
python src/dataset/build_dataset.py --val-frac 0.15
#   -> data/processed/train.json, val.json  (prints OK / SHORT per class)
```
If a class shows **SHORT**, annotate more images (`--limit` higher) or relax
`MIN_BOX_AREA_FRAC` in `src/utils/config.py` (document it in `decisions_log.md`).

## Stage 2 — Train both detectors (raw, then augmented)
```bash
# Raw (no augmentation) — baseline
python src/train/train_fasterrcnn.py --epochs 20
python src/train/train_retinanet.py  --epochs 20

# Augmented — the comparison the brief wants
python src/train/train_fasterrcnn.py --epochs 20 --augment
python src/train/train_retinanet.py  --epochs 20 --augment
```
Each run saves: `weights/<tag>_best.pt`, `outputs/figures/<tag>.png` (loss + mAP curve),
`outputs/metrics/<tag>.json`. `<tag>` = e.g. `fasterrcnn_mobilenet_raw`.
On 8 GB VRAM keep `--batch-size 2`. **AMP auto-disables for Faster R-CNN** (its fp16 box ops
are ~15× slower; `--force-amp` to override); RetinaNet keeps AMP. Use `--tag` to avoid
overwriting earlier runs.

## Stage 2.5 — Re-score a checkpoint at a fixed resolution
```bash
# val mAP + per-class (person/vehicle) AP at the resolution the phone actually runs.
python src/train/eval.py --weights weights/<tag>_best.pt --min-size 512 --max-size 640
```
Training reports mAP at the *training* resolution; this re-scores any checkpoint at a chosen
deploy resolution so comparisons across runs are apples-to-apples (see `report.py`'s caveat).

## Stage 3 — Inference (deliverables)
```bash
python src/infer/infer_image.py  --weights weights/retinanet_resnet50_aug_best.pt --image some.jpg
python src/infer/infer_folder.py --weights weights/retinanet_resnet50_aug_best.pt --folder some_dir
```

## Bonus — TTA & ensemble (annotation quality)
```bash
python src/tta_ensemble/tta_annotate.py      --limit 500 --min-agreement 2
python src/tta_ensemble/ensemble_annotate.py --limit 500 --min-score 0.5
# Then build_dataset.py on the cleaned json and retrain to compare mAP (raw vs cleaned).
```

## Stage 4 — Phone demo (BONUS, not required)
Run the trained detector live on a Pixel 6a, in the phone's browser. Full doc: `docs/08`.
```bash
# 4a. Export + self-verify (CPU only; safe alongside training) -> web/model.onnx
python src/edge/export_onnx.py --weights weights/fasterrcnn_mobilenet_aug_best.pt
# 4b. Dry-run on the laptop (localhost is a secure context, so the webcam works)
python -m http.server -d web 8000        # open http://localhost:8000
# 4c. Publish web/ via GitHub Pages (HTTPS) and open the URL in Chrome on the Pixel 6a
```
If an op errors in the browser, export RetinaNet instead (simpler graph). If `model.onnx`
> 100 MB, re-export with `--min-size 320`.

## What goes on the slides (page 4)
- Dataset creation: DoF decisions (`docs/04`), challenges, TTA/ensemble (`docs/05`).
- Model training: why our arch (`docs/03`), **raw vs augmented** mAP + curves from `outputs/`.
- 1–3 example detection images (from Stage 3).
- Future work + reflection.
```
```
