"""Quick look on the 600 x 600 px window of Krynki 1 with the most reference crowns."""
import sys, time
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import hierarchy, ladder
from data import load
from metrics import crown_metrics

mode = sys.argv[1] if len(sys.argv) > 1 else "bhat"
kw = {}
for a in sys.argv[2:]:
    key, val = a.split("="); kw[key] = float(val)
full = load("krynki1")
cr = full["crowns"]
best, win = -1, None
for r0 in range(0, 1801, 100):
    for c0 in range(0, 1801, 100):
        w = cr[r0:r0 + 600, c0:c0 + 600]
        nc = len(np.unique(w)) - 1
        if nc > best and full["valid"][r0:r0 + 600, c0:c0 + 600].mean() > 0.95:
            best, win = nc, (r0, r0 + 600, c0, c0 + 600)
print("window", win, "crowns", best)
sc = load("krynki1", window=win)
lev = ladder(12.0, k_min=1.0)
t = time.perf_counter()
h = hierarchy(sc["data"], lev, valid=sc["valid"], mode=mode, **kw)
print(f"mode {mode} {kw}: {time.perf_counter() - t:.1f} s, passes per level {[h.passes[k] for k in h.levels]}")
print(f"{'k':>6s} {'segments':>9s} {'IoU mean':>8s} {'IoU med':>8s} {'IoU>=.5':>8s} {'OS':>6s} {'US':>6s} {'pieces':>6s} {'1-px':>7s} {'med px':>7s}")
for k in h.levels:
    m, _ = crown_metrics(h.labels[k], sc["crowns"], sc["valid"])
    print(f"{k:6.2f} {m['segments']:9d} {m['iou_mean']:8.3f} {m['iou_median']:8.3f} {m['iou50']:8.3f} {m['os']:6.3f} {m['us']:6.3f} {m['pieces_median']:6.0f} {m['one_px_segments']:7d} {m['seg_px_median']:7.0f}")
