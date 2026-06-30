"""Export a trained detector to ONNX for the in-browser phone demo (BONUS — not a
project requirement; see README "Bonus: on-device phone demo").

Why this exists
---------------
torchvision detection models bake their whole pipeline into ``forward``:
``GeneralizedRCNNTransform`` (ImageNet normalize + resize) -> backbone -> heads -> NMS.
So the exported ONNX is fully self-contained: the browser just feeds a raw RGB float
tensor in [0,1] (CHW) and gets back final ``boxes / labels / scores``. No pre/post-
processing needs to be reimplemented in JavaScript.

The one knob that matters for phone speed is the transform's ``min_size``: by default the
model upscales every input so its shorter side is 800px, which is far too slow on a phone.
We lower it at export time (default 512) — a deliberate speed/accuracy trade-off, in the
spirit of docs/04_degrees_of_freedom.md.

Usage
-----
    python src/edge/export_onnx.py --weights weights/fasterrcnn_mobilenet_aug_best.pt
    python src/edge/export_onnx.py --weights ... --min-size 320 --max-size 512
    python src/edge/export_onnx.py --weights ... --sample data/raw/<some>.jpg

The script ALSO verifies the export: it runs the .onnx through onnxruntime (CPU) and
compares against the eager PyTorch output, so any unsupported-op surprise is caught here
at your desk — never on stage. Everything runs on CPU; it does not touch the GPU.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.infer.infer_common import load_model  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "edge"
WEB_MODEL = PROJECT_ROOT / "web" / "model.onnx"


def _sample_input(sample: str | None, min_size: int) -> np.ndarray:
    """Return an RGB float32 CHW array in [0,1] — exactly what the browser will feed."""
    if sample:
        from PIL import Image

        img = Image.open(sample).convert("RGB")
        # roughly match the size the browser sends (shorter side ~= min_size)
        scale = min_size / min(img.size)
        if scale < 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        return np.transpose(arr, (2, 0, 1)).copy()  # CHW
    # no sample: a synthetic frame, shorter side == min_size
    return np.random.rand(3, min_size, round(min_size * 4 / 3)).astype(np.float32)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="path to a weights/*_best.pt checkpoint")
    ap.add_argument("--min-size", type=int, default=512,
                    help="transform shorter-side size (lower = faster on phone, less accurate)")
    ap.add_argument("--max-size", type=int, default=640, help="transform longer-side cap")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--sample", default=None, help="optional image to validate the export on")
    ap.add_argument("--out", default=str(OUT_DIR / "model.onnx"))
    ap.add_argument("--no-web-copy", action="store_true",
                    help="don't also copy the result into web/model.onnx")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load on CPU (never touches the GPU; safe to run alongside training) ---
    model = load_model(args.weights, device="cpu")
    model.transform.min_size = (args.min_size,)
    model.transform.max_size = args.max_size
    model.eval()
    print(f"Loaded {args.weights}  | transform min/max -> {args.min_size}/{args.max_size}")

    x = _sample_input(args.sample, args.min_size)
    dummy = torch.from_numpy(x)  # [3,H,W] float32 in [0,1]

    # eager reference output (single image -> list of one dict)
    ref = model([dummy])[0]

    # --- export. Input is a list of one image; ONNX input becomes the [3,H,W] tensor. ---
    # NB: args must be a tuple whose single element is the image list, i.e. ([dummy],).
    # Passing a bare list [dummy] makes the (newer) legacy exporter treat it as positional
    # args and feed images=None into GeneralizedRCNN.forward -> AttributeError.
    torch.onnx.export(
        model,
        ([dummy],),
        str(out_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "input": {1: "height", 2: "width"},
            "boxes": {0: "n"},
            "labels": {0: "n"},
            "scores": {0: "n"},
        },
        do_constant_folding=True,
        # torch>=2.9 defaults torch.onnx.export to the dynamo exporter, which renames the
        # graph I/O and breaks the fixed `input`/`boxes`/`labels`/`scores` names that
        # web/app.js feeds and reads. Pin the legacy TorchScript exporter, which honors the
        # input_names/output_names/dynamic_axes above exactly and is the well-tested path
        # for torchvision detection models (RoiAlign / NMS / TopK).
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"Exported -> {out_path}  ({size_mb:.1f} MB)")

    _verify(out_path, x, ref)

    if not args.no_web_copy:
        WEB_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, WEB_MODEL)
        print(f"Copied   -> {WEB_MODEL}  (served by the web demo)")
        if size_mb > 100:
            print("  ! >100 MB: GitHub blocks files this big. Re-export with a smaller "
                  "--min-size, or use Git LFS. See docs/08_edge_phone_deploy.md.")


def _verify(onnx_path: Path, x: np.ndarray, ref: dict):
    """Run the .onnx through onnxruntime (CPU) and compare with the eager output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping verification (pip install onnxruntime).")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    boxes, labels, scores = sess.run(["boxes", "labels", "scores"], {"input": x})

    n_ref = int((ref["scores"] >= 0.5).sum())
    n_onnx = int((scores >= 0.5).sum())
    print("\n--- verification (onnxruntime CPU vs torch eager) ---")
    print(f"  detections @score>=0.5 : torch={n_ref}  onnx={n_onnx}")
    print(f"  onnx top score         : {float(scores.max()) if len(scores) else 0.0:.3f}")
    print(f"  output shapes          : boxes{boxes.shape} labels{labels.shape} scores{scores.shape}")

    if not np.isfinite(boxes).all():
        print("  ! non-finite boxes — export is broken.")
    elif len(ref["scores"]) and len(scores):
        # loose agreement: top score should be close
        d = abs(float(ref["scores"].max()) - float(scores.max()))
        ok = d < 0.05
        print(f"  top-score delta        : {d:.4f}  -> {'OK' if ok else 'CHECK (ops may differ)'}")
    print("ONNX runs in onnxruntime — the browser (WASM EP) uses the same kernels.")


if __name__ == "__main__":
    main()
