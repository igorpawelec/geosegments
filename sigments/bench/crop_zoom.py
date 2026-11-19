"""Zoom on single dead-crown patches: where does each segmenter put the boundary?"""
import sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import basins, ladder, merge_hierarchy, whiten
from data import load
from metrics import crown_metrics
import baselines as bl
from panels import stretch, boundaries, ORANGE, INK

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
mode = sys.argv[1] if len(sys.argv) > 1 else "mean"
kshow = float(sys.argv[2]) if len(sys.argv) > 2 else 4.76
out = sys.argv[3] if len(sys.argv) > 3 else OUT + "crop_zoom.png"
z, _ = whiten(data, valid, kind="full")
h = merge_hierarchy(z, valid, levels=ladder(kshow * 1.001, k_min=1.41), mode=mode, init=basins(z, valid))
zd, _ = whiten(data, valid, kind="diag")
labs = [(f"prototype {mode} k={kshow:g}", h.at(kshow)), ("adaptels T=45 (per-band units)", bl.adaptels(zd, valid, 45)),
        ("MRS 33 (0.5/0.7)", bl.multiresolution(data, valid, 33, 0.5, 0.7))]
per = [crown_metrics(lab, patches, valid)[1] for _, lab in labs]
iou = [dict(zip(d["crown"].tolist(), d["iou"].tolist())) for d in per]
sizes = np.bincount(patches.ravel()); sizes[0] = 0
pick = [int(i) for i in np.argsort(-sizes)[:3]] + [int(i) for i in np.argsort(-sizes)[10:13]] + [int(i) for i in np.argsort(-sizes)[24:27]]
rgb = stretch(data[[3, 0, 1]], valid)
fig, axs = plt.subplots(len(pick), 4, figsize=(11, 2.75 * len(pick)), dpi=110)
for row, pid in enumerate(pick):
    rr, cc = np.where(patches == pid)
    r0, r1 = max(rr.min() - 14, 0), min(rr.max() + 15, rgb.shape[0]); c0, c1 = max(cc.min() - 14, 0), min(cc.max() + 15, rgb.shape[1])
    for col in range(4):
        ax = axs[row, col]
        img = np.repeat(np.repeat(rgb[r0:r1, c0:c1], 6, axis=0), 6, axis=1).copy()
        if col > 0:
            b = boundaries(labs[col - 1][1][r0:r1, c0:c1])
            img[np.repeat(np.repeat(b, 6, axis=0), 6, axis=1) & (np.indices(img.shape[:2]).sum(0) % 2 == 0)] = (1, 1, 1)
        ax.imshow(img, interpolation="nearest")
        ax.contour(np.repeat(np.repeat(patches[r0:r1, c0:c1] == pid, 6, axis=0), 6, axis=1), levels=[0.5], colors=[ORANGE], linewidths=1.3)
        ax.set_axis_off()
        cap = f"patch of {sizes[pid]} px" if col == 0 else f"{labs[col - 1][0]}: IoU {iou[col - 1].get(pid, 0):.2f}"
        ax.text(0.5, -0.03, cap, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)
plt.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.03, wspace=0.03, hspace=0.14)
fig.savefig(out, facecolor="white"); print("saved", out)
