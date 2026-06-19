# 08 — Running the detector on a phone (Pixel 6a), in the browser

> **Status: BONUS.** This is **not** part of the project brief's deliverables. The brief
> asks for an *edge-deployable* detector and we already back that with numbers
> (`src/utils/edge_metrics.py`: params / size / latency). This doc is the optional extra:
> taking the trained model and **physically running it on a phone** as a live demo.

## Goal
Open a URL on a **Pixel 6a**, point the rear camera at a person / car, and see
person/vehicle boxes drawn live — with the model running **on the phone itself**, not on a
server. Easiest possible setup for a presentation.

## The decision: browser (onnxruntime-web), not a native app

| Option | On-device? | Effort | Verdict |
|---|---|---|---|
| **Browser + onnxruntime-web** | ✅ yes (runs in Chrome on the phone) | **Low** — one static page, a URL | **Chosen** |
| Native Android (ExecuTorch / ORT-Mobile + NNAPI) | ✅ yes, fastest | High — Android Studio, Kotlin, CameraX, build + install APK | Future work |
| Phone streams to a laptop server | ❌ no (laptop does the work) | Low | Rejected — defeats "edge device" |

The browser path is genuinely on-device (the `.onnx` is downloaded once and every inference
runs in the phone's CPU via WebAssembly) and needs **no app install, no Android Studio, no
cable** — just an HTTPS URL. That's the cleanest thing to show on stage.

## How it works
torchvision detection models bake the **entire** pipeline into `forward`:
`GeneralizedRCNNTransform` (ImageNet normalize + resize) → backbone → heads → **NMS**.
So the exported ONNX is self-contained — the web page only has to:

```
camera frame → RGB float[0,1], CHW tensor → session.run → boxes / labels / scores → draw
```

No pre/post-processing is reimplemented in JavaScript. Labels follow the project's mapping
(`background=0, person=1, vehicle=2`).

### Two engineering choices worth explaining
- **`min_size` lowered at export (default 512, vs the model's 800).** The baked-in transform
  upscales every frame so its shorter side is 800px — far too slow on a phone. Exporting with
  `--min-size 512` (or 320) trades a little accuracy for a big speed win. This is a new
  *degree of freedom*, same idea as `docs/04`.
- **WASM execution provider, not WebGPU.** FRCNN relies on `RoiAlign`, `NonMaxSuppression`,
  `TopK` and some control flow. ort-web's **WASM** kernels cover these reliably; its WebGPU
  backend only covers a subset and would fall back / fail. WASM gives a few FPS — enough for
  a demo. (Multi-threaded WASM needs COOP/COEP headers that GitHub Pages doesn't send, so we
  run single-threaded + SIMD.)

## Steps
1. **Export + self-verify** (CPU only — safe to run anytime):
   ```bash
   python src/edge/export_onnx.py --weights weights/fasterrcnn_mobilenet_aug_best.pt
   ```
   This writes `outputs/edge/model.onnx`, copies it to `web/model.onnx`, and **runs the model
   through onnxruntime (Python) comparing against PyTorch** — so any unsupported-op problem is
   caught here, not on stage.
2. **Dry-run on the laptop** (localhost is a "secure context", so the webcam works):
   ```bash
   python -m http.server -d web 8000
   # open http://localhost:8000
   ```
3. **Put it on the phone** — needs HTTPS for the camera API. Easiest: **GitHub Pages**
   (Settings → Pages → Deploy from branch → `/web` folder). Open the Pages URL in Chrome on
   the Pixel 6a, allow the camera, and demo. (Quick alternative without Pages: `ngrok http 8000`.)

## Expected performance
A few FPS at 512px input on the Pixel 6a's CPU via WASM. Lower `--min-size` to 320 for more
speed. This is a live "it really runs on the phone" flex, not a 30-FPS product.

## Troubleshooting & fallbacks
- **`model.onnx` > 100 MB:** GitHub blocks it. Re-export with a smaller `--min-size`, or use
  Git LFS. (fp32 MobileNet-FRCNN is ~70–80 MB.)
- **An op errors in ort-web:** export RetinaNet instead — single-stage, no `RoiAlign`, simpler
  graph: `python src/edge/export_onnx.py --weights weights/retinanet_resnet50_aug_best.pt`.
  The Python verification in step 1 surfaces this before the phone.
- **Camera won't open:** you're on `http://` over the LAN — the camera API requires HTTPS.
  Use the GitHub Pages URL (or `ngrok`).

## Future work
A native Android app (ExecuTorch or ORT-Mobile with NNAPI/GPU) would use the Tensor chip's
accelerators for smooth real-time inference — more "legit edge", much more setup. Deferred.
