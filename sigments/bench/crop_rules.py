"""Self-chosen tolerance: which rule, read from the area curve alone, comes close to the best k?

Variants: how the flood is detected, which fraction of the flood tolerance is taken, where
the seed sits inside the patch, the reference colour and the units.
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
PHI = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
rng = np.random.default_rng(3)


def curve(z, m, seed, window, adapt):
    size = int(m.sum())
    order, level = flood_levels(z, valid, seed, max_px=total, window=window, adapt=adapt)
    inter = np.cumsum(m.ravel()[order])
    n = np.arange(1, order.size + 1)
    iou = inter / (n + size - inter)
    ks, last = np.unique(level, return_index=True)
    area = np.r_[last[1:], order.size].astype(float)
    return ks, area, iou[area.astype(int) - 1]


def flood_k(ks, area, how):
    if how == "share":                                   # first k reaching 2 % of the scene
        return ks[min(int(np.searchsorted(area, 0.02 * total)), ks.size - 1)]
    ok = area[:-1] >= 6
    if how == "jump":                                    # the largest multiplicative jump of the area
        j = np.where(ok, area[1:] / area[:-1], 0.0)
        return ks[1:][int(np.argmax(j))]
    if how == "steep":                                   # where the area grows fastest against log k, over a 10 % step
        hi = area[np.minimum(np.searchsorted(ks, ks * 1.1, side="right") - 1, ks.size - 1)]
        g = np.where(area >= 6, hi / area, 0.0)
        return ks[int(np.argmax(g))] * 1.1
    raise ValueError(how)


def at(ks, iou_k, k):
    return iou_k[max(int(np.searchsorted(ks, k, side="right")) - 1, 0)]


seeds = {}
for pid in range(1, npatch + 1):
    m = patches == pid
    dt = ndimage.distance_transform_edt(m)
    rr, cc = np.where(m)
    pick = rng.integers(0, rr.size, size=3)
    seeds[pid] = {"centre": [np.unravel_index(np.argmax(dt), m.shape)], "random": [(rr[i], cc[i]) for i in pick]}

print("mean IoU over patches; columns = fraction of the flood tolerance")
print(f"{'units':6s} {'ref':5s} {'win':>3s} {'seed':7s} {'flood':6s} | " + " ".join(f"{p:5.2f}" for p in PHI) + " | oracle  best fixed k  corr(k*, size)  median k*/flood")
for kind in ("full", "diag"):
    z, _ = whiten(data, valid, kind=kind)
    for adapt in (False, True):
        for window in (1, 3):
            for where in ("centre", "random"):
                curves = []
                for pid in range(1, npatch + 1):
                    m = patches == pid
                    for seed in seeds[pid][where]:
                        curves.append((int(m.sum()),) + curve(z, m, seed, window, adapt))
                oracle = np.array([c[3].max() for c in curves])
                kbest = np.array([c[1][int(np.argmax(c[3]))] for c in curves])
                size = np.array([c[0] for c in curves])
                fixed = max((np.mean([at(c[1], c[3], kf) for c in curves]), kf) for kf in np.arange(2, 16, 0.5))
                for how in ("share", "jump", "steep"):
                    kf = np.array([flood_k(c[1], c[2], how) for c in curves])
                    vals = [np.mean([at(c[1], c[3], p * f) for c, f in zip(curves, kf)]) for p in PHI]
                    print(f"{kind:6s} {'mean' if adapt else 'seed':5s} {window:3d} {where:7s} {how:6s} | " + " ".join(f"{v:5.3f}" for v in vals) +
                          f" | {oracle.mean():.3f}   {fixed[0]:.3f} at {fixed[1]:4.1f}   {np.corrcoef(np.log(kbest), np.log(size))[0, 1]:5.2f}          {np.median(kbest / kf):.2f}", flush=True)
