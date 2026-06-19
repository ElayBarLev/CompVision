"""Compile training results into slide-ready tables + a comparison chart.

Reads every outputs/metrics/{tag}.json produced by train.py (each has best_map,
edge_fitness, history) and emits:
  - outputs/metrics/summary.md   : raw-vs-augmented mAP table + edge-fitness table
  - outputs/figures/model_comparison.png : grouped bar chart (mAP + size/latency)
and prints the best model.

    python src/utils/report.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = PROJECT_ROOT / "outputs" / "metrics"
FIGS = PROJECT_ROOT / "outputs" / "figures"


def load_runs():
    runs = {}
    for p in sorted(METRICS.glob("*.json")):
        if p.name in {"summary.json", "edge_fitness.json"}:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and "best_map" in d:
            runs[d.get("tag", p.stem)] = d
    return runs


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No training metrics found in {METRICS}. Train a model first.")

    # raw-vs-augmented mAP table
    map_rows = []
    for tag, d in sorted(runs.items()):
        map_rows.append([d.get("arch", tag),
                         "aug" if d.get("augment") else "raw",
                         round(d["best_map"], 4),
                         round(max((h.get("map_50", 0) for h in d.get("history", [])),
                                   default=0), 4)])

    # edge-fitness table (one row per tag)
    edge_rows = []
    for tag, d in sorted(runs.items()):
        e = d.get("edge_fitness", {})
        if e:
            edge_rows.append([tag, e.get("params_M"), e.get("size_MB"),
                              e.get("latency_cpu_ms"), e.get("latency_gpu_ms", "-"),
                              e.get("fps_gpu", "-")])

    best_tag = max(runs, key=lambda t: runs[t]["best_map"])
    best = runs[best_tag]

    md = ["# Model Comparison Summary\n",
          "## Accuracy (val mAP) — raw vs augmented",
          md_table(map_rows, ["arch", "mode", "mAP@[.5:.95]", "mAP@.5"]),
          "\n## Edge fitness",
          md_table(edge_rows, ["tag", "params (M)", "size (MB)",
                               "CPU ms", "GPU ms", "GPU fps"]) if edge_rows
          else "_(no edge metrics found)_",
          f"\n## Best model: **{best_tag}**  (val mAP = {best['best_map']:.4f})",
          ""]
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved -> {METRICS / 'summary.md'}")

    _plot(runs)


def _plot(runs):
    import matplotlib.pyplot as plt

    tags = sorted(runs)
    maps = [runs[t]["best_map"] for t in tags]
    sizes = [runs[t].get("edge_fitness", {}).get("size_MB", 0) for t in tags]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.barh(tags, maps, color="#c0392b")
    ax1.set_xlabel("val mAP@[.5:.95]"); ax1.set_title("Accuracy")
    ax2.barh(tags, sizes, color="#2980b9")
    ax2.set_xlabel("model size (MB)"); ax2.set_title("Edge fitness (smaller = better)")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "model_comparison.png"
    fig.savefig(out, dpi=120)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
