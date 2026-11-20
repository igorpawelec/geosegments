"""First-leak rule: stop a region just before it breaks through into something else.

In membership growth every pixel behind a barrier joins at the level of the barrier, so a
break-through shows as many pixels with exactly the same level. A leak is declared at the
first level whose pixel count is at least ``psi`` times the area grown so far plus ``m0``
pixels; the region is everything below ``phi`` times that level.
"""
import sys
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import whiten
from sigments.flood import flood_levels
from data import load

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
total = int(valid.sum())
rng = np.random.default_rng(3)
z, _ = whiten(data, valid, kind="full")


def curve(m, seed, window):
    size = int(m.sum())
    order, level = flood_levels(z, valid, seed, max_px=total, window=window, adapt=False)
    inter = np.cumsum(m.ravel()[order])
    n = np.arange(1, order.size + 1)
    iou = inter / (n + size - inter)
    ks, last = np.unique(level, return_index=True)
    area = np.r_[last[1:], order.size].astype(float)
    return size, ks, area, iou[area.astype(int) - 1]


def first_leak(ks, area, psi, m0):
    before = np.r_[0.0, area[:-1]]
    mass = area - before
    hit = np.where(mass >= psi * before + m0)[0]
    hit = hit[hit > 0]
    return int(hit[0]) if hit.size else ks.size - 1


def at(ks, iou_k, k):
    return iou_k[max(int(np.searchsorted(ks, k, side="left")) - 1, 0)]      # strictly below k


for window in (1, 3):
    for where in ("centre", "random"):
        curves = []
        for pid in range(1, npatch + 1):
            m = patches == pid
            if where == "centre":
                seeds = [np.unravel_index(np.argmax(ndimage.distance_transform_edt(m)), m.shape)]
            else:
                rr, cc = np.where(m)
                seeds = [(rr[i], cc[i]) for i in rng.integers(0, rr.size, size=3)]
            for s in seeds:
                curves.append(curve(m, s, window))
        oracle = np.mean([c[3].max() for c in curves])
        print(f"seed reference window {window}, seed at {where}: oracle {oracle:.3f}")
        print("   psi   m0 | phi 0.70  0.80  0.90  1.00 | median leak k, median area at the leak / patch size (phi = 1)")
        for psi in (0.15, 0.25, 0.35, 0.5, 0.75, 1.0):
            for m0 in (4, 8, 16):
                idx = [first_leak(c[1], c[2], psi, m0) for c in curves]
                vals = [np.mean([at(c[1], c[3], phi * c[1][i]) for c, i in zip(curves, idx)]) for phi in (0.7, 0.8, 0.9, 1.0)]
                rel = np.median([c[2][max(i - 1, 0)] / c[0] for c, i in zip(curves, idx)])
                print(f"  {psi:4.2f} {m0:4d} |     {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} {vals[3]:.3f} | {np.median([c[1][i] for c, i in zip(curves, idx)]):5.1f}  {rel:5.2f}", flush=True)
