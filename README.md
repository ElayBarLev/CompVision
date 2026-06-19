# CompVision Final Project 2026 — Semi-Supervised Object Detection for the Edge

Auto-label the Flickr image dataset with a large vision model (Florence-2 / SAM3), then
train a small, edge-deployable detector (torchvision Faster R-CNN / RetinaNet) to detect
**person** and **vehicle**. See [`docs/`](docs/) for the decisions and write-ups.

## Project map
| Stage | Code | Docs |
|---|---|---|
| 1. Create dataset (auto-annotate) | `src/dataset/` | `docs/01`, `02`, `04`, `05` |
| 2. Train detector(s) | `src/train/` | `docs/03` |
| 3. Inference (deliverable) | `src/infer/` | — |
| Bonus: TTA + ensemble | `src/tta_ensemble/` | `docs/05` |

## Setup

### 1. Python env
```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Git Bash:              source .venv/Scripts/activate
```

### 2. GPU setup (PyTorch with CUDA) — do this FIRST
This machine has an **NVIDIA RTX 3080 Laptop (8 GB)**. Install a CUDA wheel of PyTorch
(the default PyPI build is CPU-only):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
(cu126 carries the current PyTorch 2.12.x; the very new driver on this machine is
backward-compatible with it.)
Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Everything else
```bash
pip install -r requirements.txt
```

### 4. Kaggle credentials (for the dataset download)
`kagglehub` needs Kaggle API credentials. Either log in interactively, or place
`kaggle.json` (from your Kaggle account → Settings → Create New Token) at
`~/.kaggle/kaggle.json`.

## Usage (filled in as code lands)
```bash
python src/dataset/download.py        # download Flickr dataset via kagglehub
python src/dataset/annotate.py        # Florence-2/SAM3 -> bounding boxes
python src/dataset/build_dataset.py   # filter + COCO train/val split
python src/train/train_fasterrcnn.py  # train Faster R-CNN
python src/train/train_retinanet.py   # train RetinaNet
python src/infer/infer_image.py  --weights ... --image ...
python src/infer/infer_folder.py --weights ... --folder ...
```

## Hardware constraints (8 GB VRAM)
- Small batch sizes + gradient accumulation if needed.
- Mixed precision (AMP) for training.
- Use Florence-2 **base** for annotation; free GPU memory between stages.
