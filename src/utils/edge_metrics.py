"""Edge-fitness metrics for the detectors: parameter count, model size (MB), and
inference latency (CPU and/or GPU). This is how we back the brief's "can run on an edge
device" requirement with numbers, and compare the MobileNet vs ResNet50 backbones.

As a library:
    from src.utils.edge_metrics import measure
    stats = measure(model, device="cuda", img_size=512)

As a CLI (benchmark every architecture from scratch, no training needed):
    python src/utils/edge_metrics.py --img-size 512 --runs 30
    python src/utils/edge_metrics.py --arch fasterrcnn_mobilenet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.build_model import ARCHS, build_model        # noqa: E402
from src.utils.coco_dataset import NUM_CLASSES              # noqa: E402

METRICS = PROJECT_ROOT / "outputs" / "metrics"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    """On-disk size of the weights (params + buffers), in MB."""
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 ** 2)


@torch.inference_mode()
def latency_ms(model, device: str, img_size: int, runs: int, warmup: int = 5) -> float:
    """Mean per-image inference latency in ms on the given device."""
    model.eval().to(device)
    x = [torch.rand(3, img_size, img_size, device=device)]
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


def measure(model, device: str, img_size: int = 512, runs: int = 30) -> dict:
    cpu_runs = max(5, runs // 3)   # CPU inference is slow; a third of the GPU runs is plenty
    stats = {
        "params_M": round(count_params(model) / 1e6, 2),
        "size_MB": round(model_size_mb(model), 1),
        "img_size": img_size,
        "latency_cpu_ms": round(latency_ms(model, "cpu", img_size, cpu_runs), 1),
    }
    if device == "cuda" and torch.cuda.is_available():
        stats["latency_gpu_ms"] = round(latency_ms(model, "cuda", img_size, runs), 1)
        stats["fps_gpu"] = round(1000.0 / stats["latency_gpu_ms"], 1)
    stats["fps_cpu"] = round(1000.0 / stats["latency_cpu_ms"], 1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default=None, help="default: all archs")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    archs = [args.arch] if args.arch else ARCHS
    table = {}
    for arch in archs:
        model = build_model(arch, NUM_CLASSES, pretrained=False)
        s = measure(model, device, args.img_size, args.runs)
        table[arch] = s
        print(f"{arch:<22} params={s['params_M']:>6}M  size={s['size_MB']:>6}MB  "
              f"cpu={s['latency_cpu_ms']:>7}ms ({s['fps_cpu']}fps)"
              + (f"  gpu={s.get('latency_gpu_ms')}ms ({s.get('fps_gpu')}fps)"
                 if 'latency_gpu_ms' in s else ""))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    METRICS.mkdir(parents=True, exist_ok=True)
    out = METRICS / "edge_fitness.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
