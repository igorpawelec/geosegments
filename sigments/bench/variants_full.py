"""Which ingredient hurts on Krynki 2? The tessellation under different units, on both full scenes."""
import sys
import time
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import tessellate, whiten
from sigments.noise import stabilise
from data import load
from metrics import crown_metrics

scenes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["krynki1", "krynki2"]
KS = (4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20)
for scene in scenes:
    sc = load(scene)
    data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
    patches, _ = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
    st = stabilise(data, valid)[0]
    for kind in ("full", "diag", "scalar"):
        for use_st in (True, False):
            for smooth in (0.0, 0.7):
                z, info = whiten(st if use_st else data, valid, kind=kind)
                if smooth > 0:
                    w = ndimage.gaussian_filter(valid.astype(float), smooth)
                    z = np.stack([ndimage.gaussian_filter(np.where(valid, b, 0.0), smooth) / np.maximum(w, 1e-9) for b in z])
                    z, _ = whiten(z, valid, kind="scalar")              # back to noise units after smoothing
                rows = []
                for k in KS:
                    lab, n = tessellate(z, valid, k=k)
                    rows.append((k, crown_metrics(lab, crowns, valid)[0], crown_metrics(lab, patches, valid)[0]))
                bp = max(rows, key=lambda r: r[2]["iou_mean"])
                bc = max(rows, key=lambda r: r[1]["iou_mean"])
                print(f"{scene} {kind:6s} {'stabilised' if use_st else 'raw':10s} smooth {smooth:3.1f}: patch best {bp[2]['iou_mean']:.3f} at k={bp[0]:g} ({bp[2]['segments']} seg, OS {bp[2]['os']:.2f} US {bp[2]['us']:.2f}) | "
                      f"crown best {bc[1]['iou_mean']:.3f} at k={bc[0]:g} | patch by k: " + " ".join(f"{m['iou_mean']:.2f}" for _, _, m in rows), flush=True)
