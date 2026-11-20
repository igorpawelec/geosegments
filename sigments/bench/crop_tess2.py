"""Sensitivity of the membership tessellation (stabilised bands, competition) to its internal settings."""
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import tessellate, whiten
from sigments.noise import stabilise
from data import load
from metrics import crown_metrics

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, _ = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
KS = (5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9)
for kind in ("full", "diag"):
    z, _ = whiten(stabilise(data, valid)[0], valid, kind=kind)
    for share in (0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        for window in (1, 3):
            rows = []
            for k in KS:
                lab, n = tessellate(z, valid, k=k, calm_share=share, window=window)
                rows.append((k, crown_metrics(lab, crowns, valid)[0], crown_metrics(lab, patches, valid)[0]))
            bp = max(rows, key=lambda r: r[2]["iou_mean"]); bc = max(rows, key=lambda r: r[1]["iou_mean"])
            print(f"{kind:4s} calm {share:4.2f} window {window}: patch best {bp[2]['iou_mean']:.3f} at k={bp[0]:g} ({bp[2]['segments']} seg, 1-px {bp[2]['one_px_segments']}), crown best {bc[1]['iou_mean']:.3f} at k={bc[0]:g} | patch by k: " +
                  " ".join(f"{m['iou_mean']:.2f}" for _, _, m in rows), flush=True)
