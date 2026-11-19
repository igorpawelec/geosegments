"""How good is the hierarchy itself? Per-crown best IoU over all levels (an oracle cut).

If every crown could pick its own level, how well would it be delineated? The same question
for adaptels and multiresolution segmentation over their parameter ranges tells whether a
fixed threshold or the family of regions is what limits each method.
"""
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import basins, ladder, merge_hierarchy, whiten
from data import load
from metrics import crown_metrics
import baselines as bl

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]


def oracle(named_labels):
    per = {}
    for name, lab in named_labels:
        m, d = crown_metrics(lab, crowns, valid)
        per[name] = (m, dict(zip(d["crown"].tolist(), d["iou"].tolist())))
    ids = sorted(set().union(*[set(v[1]) for v in per.values()]))
    best = np.array([max(v[1].get(c, 0.0) for v in per.values()) for c in ids])
    which = [max(per, key=lambda n: per[n][1].get(c, 0.0)) for c in ids]
    fixed = max(per.items(), key=lambda kv: kv[1][0]["iou_mean"])
    return best, which, fixed


for kind in ("full", "diag"):
    z, _ = whiten(data, valid, kind=kind)
    for mode in ("mean", "bhat"):
        h = merge_hierarchy(z, valid, levels=ladder(16.0, k_min=1.0, ratio=2 ** 0.125), mode=mode, init=basins(z, valid))
        best, which, fixed = oracle([(f"{k:.2f}", h.labels[k]) for k in h.levels])
        lv = np.array([float(w) for w in which])
        print(f"prototype {mode:5s} {kind:5s}: fixed level {fixed[0]} IoU {fixed[1][0]['iou_mean']:.3f} | oracle per crown {best.mean():.3f} (>=0.5: {(best >= 0.5).mean():.2f}), "
              f"chosen k quartiles {np.percentile(lv, [10, 25, 50, 75, 90]).round(1)}", flush=True)

zd, _ = whiten(data, valid, kind="diag")
best, which, fixed = oracle([(f"{t}", bl.adaptels(zd, valid, t)) for t in (8, 11, 16, 22, 27, 32, 38, 45, 54, 64, 90)])
print(f"adaptels (per-band units): fixed T {fixed[0]} IoU {fixed[1][0]['iou_mean']:.3f} | oracle per crown {best.mean():.3f} (>=0.5: {(best >= 0.5).mean():.2f}), "
      f"chosen T quartiles {np.percentile([float(w) for w in which], [10, 25, 50, 75, 90]).round(0)}", flush=True)
best, which, fixed = oracle([(f"{s}", bl.multiresolution(data, valid, s, 0.5, 0.7)) for s in (8, 11, 14, 18, 22, 27, 33, 40, 50, 65)])
print(f"MRS shape 0.5/0.7: fixed scale {fixed[0]} IoU {fixed[1][0]['iou_mean']:.3f} | oracle per crown {best.mean():.3f} (>=0.5: {(best >= 0.5).mean():.2f}), "
      f"chosen scale quartiles {np.percentile([float(w) for w in which], [10, 25, 50, 75, 90]).round(0)}", flush=True)
