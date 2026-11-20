"""Does the growth curve of an adaptel show when the object is full?

For every dead-crown patch an adaptel is grown from its most interior pixel and recorded.
The area as a function of the threshold T is compared with the IoU against the patch, and
two rules that pick T from the curve alone are scored against the best T per patch.
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
from sigments.grow import growth
from data import load

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
kind = sys.argv[1] if len(sys.argv) > 1 else "diag"
z, _ = whiten(data, valid, kind=kind)
RHO = 1.25
rows = []
curves = {}
for pid in range(1, npatch + 1):
    m = patches == pid
    size = int(m.sum())
    dt = ndimage.distance_transform_edt(m)
    seed = np.unravel_index(np.argmax(dt), m.shape)
    order, join = growth(z, valid, seed, max_px=max(6000, 25 * size))
    inside = m.ravel()[order]
    n = np.arange(1, order.size + 1)
    inter = np.cumsum(inside)
    iou = inter / (n + size - inter)
    # region(T) = order[join < T]; evaluate at every distinct join value
    ts, first = np.unique(join, return_index=True)          # region just below ts[i] has first[i] pixels
    area = first[1:].astype(float); tt = ts[1:]; iou_t = iou[first[1:] - 1]
    best = int(np.argmax(iou_t))
    # rule "stable": T minimising the relative growth of the area between T / rho and T * rho
    a_hi = np.searchsorted(join, tt * RHO, side="left"); a_lo = np.searchsorted(join, tt / RHO, side="left")
    ok = (area >= 8) & (a_hi < order.size) & (a_lo >= 4)
    stab = np.where(ok, (a_hi - a_lo) / np.maximum(area, 1), np.inf)
    i_stab = int(np.argmin(stab))
    # rule "gap": the largest multiplicative gap between consecutive joining thresholds
    gap = np.where((area >= 8)[:-1], tt[1:] / tt[:-1], 0.0) if tt.size > 1 else np.array([0.0])
    i_gap = int(np.argmax(gap))
    rows.append((pid, size, iou_t[best], tt[best], area[best], iou_t[i_stab], tt[i_stab], area[i_stab], stab[i_stab], iou_t[i_gap], tt[i_gap], gap[i_gap]))
    curves[pid] = (tt, area, iou_t, tt[best], tt[i_stab])
r = np.array(rows)
print("patch size | oracle IoU   T   area | stable rule IoU   T   area  growth | gap rule IoU   T  gap")
for x in r[np.argsort(r[:, 1])]:
    print(f"{int(x[0]):5d} {int(x[1]):5d} | {x[2]:.2f} {x[3]:6.1f} {int(x[4]):5d} | {x[5]:.2f} {x[6]:6.1f} {int(x[7]):5d} {x[8]:5.2f} | {x[9]:.2f} {x[10]:6.1f} {x[11]:5.2f}")
print(f"mean IoU: oracle T per patch {r[:, 2].mean():.3f} | stable rule {r[:, 5].mean():.3f} | gap rule {r[:, 9].mean():.3f}")
for T in (16, 22, 32, 45, 64, 90):
    v = [c[2][max(np.searchsorted(c[0], T, side='right') - 1, 0)] for c in curves.values()]
    print(f"  fixed T={T}: mean IoU {np.mean(v):.3f}")
print("correlation log(oracle T) with log(size): %.2f; log(stable T) with log(oracle T): %.2f" % (
    np.corrcoef(np.log(r[:, 3]), np.log(r[:, 1]))[0, 1], np.corrcoef(np.log(r[:, 6]), np.log(r[:, 3]))[0, 1]))

pick = [int(x[0]) for x in r[np.argsort(r[:, 1])][[2, 8, 14, 20, 26, 30, 33, 35, 37]]]
fig, axs = plt.subplots(3, 3, figsize=(12, 9), dpi=110)
for ax, pid in zip(axs.ravel(), pick):
    tt, area, iou_t, tb, tstab = curves[pid]
    ax.plot(tt, area, color="#2a6fbb", lw=1.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2, tt.max())
    ax.axhline((patches == pid).sum(), color="#8a8a8a", lw=0.8, ls=":")
    ax.axvline(tb, color="#eb6834", lw=1.2); ax.axvline(tstab, color="#1f1f1f", lw=1.0, ls="--")
    ax2 = ax.twinx(); ax2.plot(tt, iou_t, color="#eb6834", lw=1.0, alpha=0.8); ax2.set_ylim(0, 1); ax2.set_yticks([])
    ax.text(0.03, 0.95, f"patch of {(patches == pid).sum()} px", transform=ax.transAxes, va="top", fontsize=8)
    ax.tick_params(labelsize=7)
fig.text(0.5, 0.01, "threshold T (accumulated distance, noise units): blue = area of the adaptel (px), orange = IoU with the patch; "
         "orange line = best T, dashed = stable rule, dotted = patch size", ha="center", fontsize=8)
plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.07, wspace=0.16, hspace=0.14)
fig.savefig(OUT + "crop_growth.png", facecolor="white")
