"""First look at the prototype on Krynki 1: one hierarchy run, crown metrics at every level."""
import sys, time
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import hierarchy, ladder
from data import load
from metrics import crown_metrics

name = sys.argv[1] if len(sys.argv) > 1 else "krynki1"
mode = sys.argv[2] if len(sys.argv) > 2 else "bhat"
sc = load(name)
print(name, sc["data"].shape, "valid share", round(float(sc["valid"].mean()), 3), "crowns", sc["n_crowns"], flush=True)
lev = ladder(12.0, k_min=1.0)
h = hierarchy(sc["data"], lev, valid=sc["valid"], mode=mode, verbose=True)
print(f"total {h.seconds:.1f} s")
print(f"{'k':>6s} {'segments':>9s} {'IoU mean':>8s} {'IoU med':>8s} {'IoU>=.5':>8s} {'F1 area':>8s} {'OS':>6s} {'US':>6s} {'pieces':>6s} {'1-px':>8s} {'med px':>7s}")
for k in h.levels:
    m, _ = crown_metrics(h.labels[k], sc["crowns"], sc["valid"])
    print(f"{k:6.2f} {m['segments']:9d} {m['iou_mean']:8.3f} {m['iou_median']:8.3f} {m['iou50']:8.3f} {m['f1_area']:8.3f} {m['os']:6.3f} {m['us']:6.3f} {m['pieces_median']:6.0f} {m['one_px_segments']:8d} {m['seg_px_median']:7.0f}")
np.savez_compressed(OUT + f"{name}_{mode}_hier.npz", levels=np.array(h.levels), **{f"k{k:.4f}": h.labels[k] for k in h.levels})
