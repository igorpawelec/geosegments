"""Per-patch view: which objects does each variant delineate, and at which level?

For every dead-crown patch: its size, its contrast to the surroundings, the best IoU over
all levels of two hierarchies (raw and variance-stabilised bands) and the level where that
best occurs; the same for adaptels over T. Shows whether one global level can serve all.
"""
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import basins, ladder, merge_hierarchy, whiten
from sigments.noise import stabilise
from data import load
from metrics import crown_metrics
import baselines as bl

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
size = np.bincount(patches.ravel(), minlength=npatch + 1)


def sweep(named):
    best = np.zeros(npatch + 1); arg = np.zeros(npatch + 1)
    for p, lab in named:
        _, d = crown_metrics(lab, patches, valid, min_px=1)
        better = d["iou"] > best[d["crown"]]
        best[d["crown"][better]] = d["iou"][better]; arg[d["crown"][better]] = p
    return best[1:], arg[1:]


st, _ = stabilise(data, valid)
res = {}
for name, src in (("raw", data), ("stab", st)):
    z, _ = whiten(src, valid, kind="full")
    h = merge_hierarchy(z, valid, levels=ladder(16.0, k_min=1.0, ratio=2 ** 0.125), mode="mean", init=basins(z, valid))
    res[name] = sweep([(k, h.labels[k]) for k in h.levels])
    # contrast of every patch to a 4-pixel ring around it, in the units of this variant
    con = []
    for i in range(1, npatch + 1):
        m = patches == i
        ring = ndimage.binary_dilation(m, iterations=5) & ~ndimage.binary_dilation(m, iterations=1) & valid & (patches == 0)
        con.append(np.linalg.norm(z[:, m].mean(1) - z[:, ring].mean(1)) if ring.any() else np.nan)
    res[name + "_contrast"] = np.array(con)
zd, _ = whiten(data, valid, kind="diag")
res["adaptels"] = sweep([(t, bl.adaptels(zd, valid, t)) for t in (11, 16, 22, 27, 32, 38, 45, 54, 64, 76, 90, 128)])

order = np.argsort(res["raw_contrast"])
print("patch  size  contrast raw/stab | oracle IoU (level): raw-units hierarchy, stabilised hierarchy, adaptels (T)")
for i in order:
    print(f"{i + 1:5d} {size[i + 1]:5d}  {res['raw_contrast'][i]:6.1f} {res['stab_contrast'][i]:6.1f} | "
          f"{res['raw'][0][i]:.2f} (k {res['raw'][1][i]:4.1f})   {res['stab'][0][i]:.2f} (k {res['stab'][1][i]:4.1f})   {res['adaptels'][0][i]:.2f} (T {res['adaptels'][1][i]:3.0f})")
lo = order[: npatch // 2]; hi = order[npatch // 2:]
for label, idx in (("lower-contrast half", lo), ("higher-contrast half", hi), ("all", order)):
    print(f"{label:22s} oracle IoU: raw {res['raw'][0][idx].mean():.3f}  stabilised {res['stab'][0][idx].mean():.3f}  adaptels {res['adaptels'][0][idx].mean():.3f}")
print("correlation of the best level with contrast: raw %.2f, stabilised %.2f; of the best T with patch size: %.2f" % (
    np.corrcoef(res["raw_contrast"], res["raw"][1])[0, 1], np.corrcoef(res["stab_contrast"], res["stab"][1])[0, 1],
    np.corrcoef(np.log(size[1:]), np.log(res["adaptels"][1]))[0, 1]))
