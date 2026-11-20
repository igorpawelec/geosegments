"""Development window of Krynki 1: every merge criterion over the ladder, compact table."""
import sys, time
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import hierarchy, ladder
from data import load
from metrics import crown_metrics

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
modes = sys.argv[1].split(",")
kw = {}
for a in sys.argv[2:]:
    key, val = a.split("="); kw[key] = val if key == "start" else float(val)
lev = ladder(float(kw.pop("kmax", 16.0)), k_min=float(kw.pop("kmin", 0.71)))
for mode in modes:
    t = time.perf_counter()
    h = hierarchy(sc["data"], lev, valid=sc["valid"], mode=mode, **kw)
    dt = time.perf_counter() - t
    rows = [(k, crown_metrics(h.labels[k], sc["crowns"], sc["valid"])[0]) for k in h.levels]
    best = max(rows, key=lambda r: r[1]["iou_mean"])
    print(f"== {mode} {kw}: {dt:.0f} s, {sum(h.passes.values())} passes; best IoU {best[1]['iou_mean']:.3f} at k={best[0]:.2f} with {best[1]['segments']} segments", flush=True)
    print("   k      seg   IoU  >=.5    OS    US  pc   1px")
    for k, m in rows:
        if m["segments"] < 40 or m["segments"] > 60000:
            continue
        print(f"{k:6.2f} {m['segments']:6d} {m['iou_mean']:.3f} {m['iou50']:.3f} {m['os']:.3f} {m['us']:.3f} {m['pieces_median']:3.0f} {m['one_px_segments']:5d}", flush=True)
