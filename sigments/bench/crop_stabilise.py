"""Does the variance-stabilising transform help? Crown and patch IoU on the development window."""
import sys
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
patches, _ = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
modes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mean"]
st, info = stabilise(data, valid)
for name, level_sd in zip(sc["bands"], info["curves"]):
    print(f"{name:6s} sd by brightness: " + " ".join(f"{l:.0f}:{s:.1f}" for l, s in zip(*level_sd)))
for label, src in (("raw bands", data), ("stabilised bands", st)):
    z, _ = whiten(src, valid, kind="full")
    for mode in modes:
        h = merge_hierarchy(z, valid, levels=ladder(16.0, k_min=1.41), mode=mode, init=basins(z, valid))
        rows = [(k, crown_metrics(h.labels[k], crowns, valid)[0], crown_metrics(h.labels[k], patches, valid)[0]) for k in h.levels if h.counts[k] > 30]
        bc = max(rows, key=lambda r: r[1]["iou_mean"]); bp = max(rows, key=lambda r: r[2]["iou_mean"])
        print(f"== {mode}, {label}: best crown IoU {bc[1]['iou_mean']:.3f} at k={bc[0]:.1f} ({bc[1]['segments']} seg) | best patch IoU {bp[2]['iou_mean']:.3f} at k={bp[0]:.1f} ({bp[2]['segments']} seg, >=0.5 {bp[2]['iou50']:.2f})")
        print("   " + "  ".join(f"{k:.1f}:{mc['iou_mean']:.2f}/{mp['iou_mean']:.2f}" for k, mc, mp in rows), flush=True)
    for kind in ("diag",):
        zz, _ = whiten(src, valid, kind=kind)
        rows = []
        for t in (11, 16, 22, 32, 45, 64, 90):
            lab = bl.adaptels(zz, valid, t)
            rows.append((t, crown_metrics(lab, crowns, valid)[0], crown_metrics(lab, patches, valid)[0]))
        bc = max(rows, key=lambda r: r[1]["iou_mean"]); bp = max(rows, key=lambda r: r[2]["iou_mean"])
        print(f"== adaptels ({kind} units), {label}: best crown IoU {bc[1]['iou_mean']:.3f} at T={bc[0]} | best patch IoU {bp[2]['iou_mean']:.3f} at T={bp[0]}")
        print("   " + "  ".join(f"{t}:{mc['iou_mean']:.2f}/{mp['iou_mean']:.2f}" for t, mc, mp in rows), flush=True)
