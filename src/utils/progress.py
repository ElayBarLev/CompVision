"""Live training-progress dashboard.

Parses the chained training run's stdout log and renders a progress bar per run
(epochs done / total, current best mAP, latest loss). Use --watch for a live view.

    python src/utils/progress.py            # one-shot snapshot
    python src/utils/progress.py --watch     # refresh every few seconds
    python src/utils/progress.py --log <path-to-training.output>

Auto-detects the active background training log if --log is omitted.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# default training matrix (order matters)
EXPECTED = ["fasterrcnn_mobilenet_raw", "fasterrcnn_mobilenet_aug",
            "retinanet_resnet50_raw", "retinanet_resnet50_aug"]

BANNER = re.compile(r"=== Training (\S+) on")
EPOCH = re.compile(r"epoch (\d+): loss=([\d.]+) mAP=([\d.]+) mAP@\.5=([\d.]+)")
BATCH = re.compile(r"epoch (\d+) \| batch (\d+)/(\d+)")


def find_log() -> Path | None:
    """Find the most recent harness task .output that looks like a training run."""
    pat = os.path.join(os.environ.get("TEMP", "/tmp"), "claude", "*", "*", "tasks", "*.output")
    cands = []
    for p in glob.glob(pat):
        try:
            head = Path(p).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "=== Training" in head:
            cands.append((os.path.getmtime(p), p))
    if not cands:
        return None
    return Path(max(cands)[1])


def parse(log_text: str) -> dict:
    """tag -> {epochs:[(n,loss,map,map50)], cur:(epoch,batch,total)|None}"""
    runs: dict[str, dict] = {}
    cur_tag = None
    for line in log_text.replace("\r", "\n").split("\n"):
        m = BANNER.search(line)
        if m:
            cur_tag = m.group(1)
            runs.setdefault(cur_tag, {"epochs": [], "cur": None})
            continue
        if cur_tag is None:
            continue
        e = EPOCH.search(line)
        if e:
            runs[cur_tag]["epochs"].append(
                (int(e.group(1)), float(e.group(2)), float(e.group(3)), float(e.group(4))))
            runs[cur_tag]["cur"] = None
            continue
        b = BATCH.search(line)
        if b:
            runs[cur_tag]["cur"] = (int(b.group(1)), int(b.group(2)), int(b.group(3)))
    return runs


def bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


def render(runs: dict, total_epochs: int) -> str:
    lines = ["", "  ===== Training progress ====="]
    done_epochs = 0
    for i, tag in enumerate(EXPECTED, 1):
        r = runs.get(tag)
        # is this run finished? (metrics file written at run end)
        finished = (PROJECT_ROOT / "outputs" / "metrics" / f"{tag}.json").exists()
        if r is None and not finished:
            lines.append(f"  {i}. {tag:<28} [{bar(0)}]  pending")
            continue
        eps = r["epochs"] if r else []
        n = len(eps)
        done_epochs += n
        best = max((e[2] for e in eps), default=0.0)
        last = eps[-1] if eps else None
        frac = n / total_epochs
        tail = ""
        if finished:
            tail = f"DONE  best mAP={best:.3f}"
        elif r and r["cur"]:
            ce, cb, ct = r["cur"]
            frac = ((n) + cb / ct) / total_epochs
            tail = f"ep {ce}/{total_epochs} batch {cb}/{ct}  best mAP={best:.3f}"
        elif last:
            tail = (f"ep {n}/{total_epochs}  best mAP={best:.3f} "
                    f"(last loss={last[1]:.3f} mAP@.5={last[3]:.3f})")
        else:
            tail = "starting…"
        lines.append(f"  {i}. {tag:<28} [{bar(frac)}]  {tail}")

    overall = done_epochs / (len(EXPECTED) * total_epochs)
    lines.append("  " + "-" * 33)
    lines.append(f"  overall  [{bar(overall)}]  {done_epochs}/{len(EXPECTED)*total_epochs} epochs "
                 f"({overall*100:.0f}%)")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="training stdout log (auto-detected if omitted)")
    ap.add_argument("--epochs", type=int, default=15, help="epochs per run")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    log = Path(args.log) if args.log else find_log()
    if not log or not log.exists():
        raise SystemExit("No training log found. Pass --log <path>.")

    while True:
        text = log.read_text(encoding="utf-8", errors="ignore")
        out = render(parse(text), args.epochs)
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"  (watching {log.name} — Ctrl-C to stop)")
        print(out)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
