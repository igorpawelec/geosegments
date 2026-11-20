"""Zoom on dead-crown patches of a full scene, from the label rasters saved by full_eval.py."""
import sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from data import load
from metrics import crown_metrics
from panels import stretch, boundaries, ORANGE, INK

scene = sys.argv[1]
runs = sys.argv[2].split(",")                       # e.g. proto_8,adaptels_260,mrs_shape_33
out = sys.argv[3] if len(sys.argv) > 3 else OUT + f"zoom_{scene}.png"
ranks = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 3, 8, 20, 40, 70, 100, 140]
sc = load(scene)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
labs = [(r, np.load(OUT + f"labels_{scene}/{r}.npz")["labels"]) for r in runs]
iou = []
for _, lab in labs:
    _, d = crown_metrics(lab, patches, valid, min_px=1)
    iou.append(dict(zip(d["crown"].tolist(), d["iou"].tolist())))
sizes = np.bincount(patches.ravel())
sizes[0] = 0
order = np.argsort(-sizes)
pick = [int(order[r]) for r in ranks if r < npatch]
fig, axs = plt.subplots(len(pick), len(labs) + 1, figsize=(2.9 * (len(labs) + 1), 2.9 * len(pick)), dpi=100)
for row, pid in enumerate(pick):
    rr, cc = np.where(patches == pid)
    r0, r1 = max(rr.min() - 16, 0), min(rr.max() + 17, valid.shape[0])
    c0, c1 = max(cc.min() - 16, 0), min(cc.max() + 17, valid.shape[1])
    rgb = stretch(data[[3, 0, 1]][:, r0:r1, c0:c1], valid[r0:r1, c0:c1], 1, 99)
    f = max(1, int(round(220 / max(r1 - r0, c1 - c0))))
    for col in range(len(labs) + 1):
        ax = axs[row, col]
        img = np.repeat(np.repeat(rgb, f, axis=0), f, axis=1).copy()
        if col > 0:
            b = np.repeat(np.repeat(boundaries(labs[col - 1][1][r0:r1, c0:c1]), f, axis=0), f, axis=1)
            img[b & (np.indices(b.shape).sum(0) % 2 == 0)] = (1, 1, 1)
        ax.imshow(img, interpolation="nearest")
        ax.contour(np.repeat(np.repeat(patches[r0:r1, c0:c1] == pid, f, axis=0), f, axis=1), levels=[0.5], colors=[ORANGE], linewidths=1.2)
        ax.set_axis_off()
        cap = f"patch of {sizes[pid]} px" if col == 0 else f"{labs[col - 1][0]}: IoU {iou[col - 1].get(pid, 0):.2f}"
        ax.text(0.5, -0.03, cap, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)
plt.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.03, wspace=0.03, hspace=0.13)
fig.savefig(out, facecolor="white")
print("saved", out)
