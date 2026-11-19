"""Patches instead of single crowns: touching reference crowns are one object.

A colour-based segmenter cannot separate two crowns of the same colour that touch; a method
with a size parameter does so only by being tuned to the crown size. Scoring against
connected dead-crown patches asks the size-free question: is each anomalous object, whatever
its size, delineated as one segment?
"""
import sys
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import basins, ladder, merge_hierarchy, whiten
from data import load
from metrics import crown_metrics
import baselines as bl

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
sizes = np.bincount(patches.ravel())[1:]
ncr = np.array([len(np.unique(crowns[patches == i])) for i in range(1, npatch + 1)])
print(f"{sc['n_crowns']} crowns form {npatch} patches; crowns per patch: 1 -> {(ncr == 1).sum()}, 2-3 -> {((ncr > 1) & (ncr < 4)).sum()}, 4+ -> {(ncr >= 4).sum()}; "
      f"patch size px: min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}")


def table(name, named):
    rows = []
    for p, lab in named:
        mc, _ = crown_metrics(lab, crowns, valid)
        mp, _ = crown_metrics(lab, patches, valid)
        rows.append((p, mc, mp))
    bc = max(rows, key=lambda r: r[1]["iou_mean"]); bp = max(rows, key=lambda r: r[2]["iou_mean"])
    print(f"== {name}: best crown IoU {bc[1]['iou_mean']:.3f} at {bc[0]} ({bc[1]['segments']} seg) | best patch IoU {bp[2]['iou_mean']:.3f} at {bp[0]} ({bp[2]['segments']} seg, >=0.5: {bp[2]['iou50']:.2f})")
    print("   " + "  ".join(f"{p}:{mc['iou_mean']:.2f}/{mp['iou_mean']:.2f}" for p, mc, mp in rows), flush=True)
    return rows


z, _ = whiten(data, valid, kind="full")
for mode in ("mean", "bhat"):
    h = merge_hierarchy(z, valid, levels=ladder(16.0, k_min=1.41), mode=mode, init=basins(z, valid))
    table(f"prototype {mode} (crown/patch IoU by k)", [(f"{k:.1f}", h.labels[k]) for k in h.levels if h.counts[k] > 30])
zd, _ = whiten(data, valid, kind="diag")
table("adaptels, per-band units (by T)", [(t, bl.adaptels(zd, valid, t)) for t in (16, 22, 32, 45, 64, 90, 128, 180)])
table("MRS shape 0.5/0.7 (by scale)", [(s, bl.multiresolution(data, valid, s, 0.5, 0.7)) for s in (18, 22, 27, 33, 40, 50, 65, 80)])
table("MRS shape 0.1/0.5 (by scale)", [(s, bl.multiresolution(data, valid, s)) for s in (18, 22, 27, 33, 40, 50, 65, 80)])
