"""Membership growth from every dead-crown patch: is there a plateau before the flood?

Region(k) = connected pixels within k of the reference colour. Scores per patch the best k,
and rules that choose k from the area curve alone:
  flood/2   half of the tolerance at which the region floods (the half-contrast contour);
  stable    the k where the area changes least between k / rho and k * rho, below the flood.
"""
import sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import whiten
from sigments.flood import flood_levels
from data import load

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
kind = sys.argv[1] if len(sys.argv) > 1 else "full"
adapt = (sys.argv[2] == "adapt") if len(sys.argv) > 2 else False
window = int(sys.argv[3]) if len(sys.argv) > 3 else 3
z, _ = whiten(data, valid, kind=kind)
total = int(valid.sum())
RHO = 1.2
rows, curves = [], {}
for pid in range(1, npatch + 1):
    m = patches == pid
    size = int(m.sum())
    seed = np.unravel_index(np.argmax(ndimage.distance_transform_edt(m)), m.shape)
    order, level = flood_levels(z, valid, seed, max_px=total, window=window, adapt=adapt)
    inter = np.cumsum(m.ravel()[order])
    n = np.arange(1, order.size + 1)
    iou = inter / (n + size - inter)
    ks, last = np.unique(level, return_index=True)            # region(ks[i]) ends before last[i + 1]
    area = np.r_[last[1:], order.size].astype(float)
    iou_k = iou[area.astype(int) - 1]
    best = int(np.argmax(iou_k))
    flood_i = min(int(np.searchsorted(area, 0.02 * total)), ks.size - 1)    # first k reaching 2 % of the scene
    k_flood = ks[flood_i]
    i_half = max(int(np.searchsorted(ks, 0.5 * k_flood, side="right")) - 1, 0)
    below = np.arange(ks.size) < flood_i
    a_hi = area[np.minimum(np.searchsorted(ks, ks * RHO, side="right") - 1, ks.size - 1)]
    a_lo = area[np.maximum(np.searchsorted(ks, ks / RHO, side="right") - 1, 0)]
    stab = np.where(below & (area >= 6) & (ks * RHO < k_flood), (a_hi - a_lo) / area, np.inf)
    i_stab = int(np.argmin(stab)) if np.isfinite(stab).any() else i_half
    rows.append((pid, size, iou_k[best], ks[best], k_flood, iou_k[i_half], ks[i_half], iou_k[i_stab], ks[i_stab]))
    curves[pid] = (ks, area, iou_k, ks[best], k_flood)
r = np.array(rows)
ref_name = "running mean" if adapt else "seed"
print(f"units {kind}, reference {ref_name} ({window} px window)")
print("patch size | oracle IoU  k | flood k | flood/2 IoU  k | stable IoU  k")
for x in r[np.argsort(r[:, 1])]:
    print(f"{int(x[0]):5d} {int(x[1]):5d} | {x[2]:.2f} {x[3]:5.1f} | {x[4]:5.1f} | {x[5]:.2f} {x[6]:5.1f} | {x[7]:.2f} {x[8]:5.1f}")
print(f"mean IoU: oracle k per patch {r[:, 2].mean():.3f} | flood/2 rule {r[:, 5].mean():.3f} | stable rule {r[:, 7].mean():.3f}")
for kf in (2, 3, 4, 5, 6, 8, 10):
    v = [c[2][max(np.searchsorted(c[0], kf, side="right") - 1, 0)] for c in curves.values()]
    print(f"  fixed k={kf}: mean IoU {np.mean(v):.3f}")
ratio = r[:, 3] / r[:, 4]
print("correlation log(oracle k) with log(size): %.2f; oracle k / flood k: median %.2f (quartiles %.2f-%.2f)" % (
    np.corrcoef(np.log(r[:, 3]), np.log(r[:, 1]))[0, 1], np.median(ratio), np.percentile(ratio, 25), np.percentile(ratio, 75)))
pick = [int(x[0]) for x in r[np.argsort(r[:, 1])][[2, 8, 14, 20, 26, 30, 33, 35, 37]]]
fig, axs = plt.subplots(3, 3, figsize=(12, 9), dpi=110)
for ax, pid in zip(axs.ravel(), pick):
    ks, area, iou_k, kb, kfl = curves[pid]
    ax.step(ks, area, where="post", color="#2a6fbb", lw=1.6)
    ax.set_yscale("log")
    ax.set_xlim(0, min(ks.max(), 1.4 * kfl))
    ax.axhline((patches == pid).sum(), color="#8a8a8a", lw=0.8, ls=":")
    ax.axvline(kb, color="#eb6834", lw=1.2)
    ax.axvline(kfl / 2, color="#1f1f1f", lw=1.0, ls="--")
    ax2 = ax.twinx()
    ax2.step(ks, iou_k, where="post", color="#eb6834", lw=1.0, alpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax.text(0.03, 0.95, f"patch of {(patches == pid).sum()} px", transform=ax.transAxes, va="top", fontsize=8)
    ax.tick_params(labelsize=7)
fig.text(0.5, 0.01, "tolerance k (noise units): blue = area of the region (px), orange = IoU with the patch; "
         "orange line = best k, dashed = half of the flood tolerance, dotted = patch size", ha="center", fontsize=8)
plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.07, wspace=0.16, hspace=0.14)
fig.savefig(OUT + f"crop_flood_{kind}.png", facecolor="white")
