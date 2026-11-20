"""Does the choice of units matter? Mean criterion and adaptels under three kinds of whitening."""
import sys
import time
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import basins, ladder, merge_hierarchy, whiten
from data import load
from metrics import crown_metrics
import baselines as bl

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["full", "diag", "scalar"]
for kind in kinds:
    z, info = whiten(data, valid, kind=kind)
    if kind == "full":
        ev = np.linalg.eigvalsh(info["S"])
        print("difference covariance, sd along its principal axes (DN):", np.round(np.sqrt(ev[::-1]), 2))
    t = time.perf_counter()
    h = merge_hierarchy(z, valid, levels=ladder(14.0, k_min=1.0), mode="mean", init=basins(z, valid))
    rows = [(k, crown_metrics(h.labels[k], crowns, valid)[0]) for k in h.levels]
    best = max(rows, key=lambda r: r[1]["iou_mean"])
    print(f"== mean, {kind}: best IoU {best[1]['iou_mean']:.3f} at k={best[0]:.2f}, {best[1]['segments']} segments, OS {best[1]['os']:.2f} US {best[1]['us']:.2f}; "
          f"IoU by level: " + " ".join(f"{k:.1f}:{m['iou_mean']:.2f}" for k, m in rows if 300 < m['segments'] < 30000), flush=True)
    grid = [8, 11, 16, 22, 32, 45, 64, 90]
    out = []
    for thr in grid:
        lab = bl.adaptels(z, valid, thr)
        m, _ = crown_metrics(lab, crowns, valid)
        out.append((thr, m))
    best = max(out, key=lambda r: r[1]["iou_mean"])
    print(f"== adaptels, {kind}: best IoU {best[1]['iou_mean']:.3f} at T={best[0]}, {best[1]['segments']} segments; " +
          " ".join(f"{t}:{m['iou_mean']:.2f}" for t, m in out), flush=True)
