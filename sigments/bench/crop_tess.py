"""The membership tessellation on the development window: crown and patch IoU over its settings."""
import sys
import time
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
kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["full"]
shares = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5]
windows = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1]
ks = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [4, 5, 6, 7, 8, 9, 10, 12]
tessellate(whiten(data, valid)[0][:, :64, :64], valid[:64, :64], 6.0)          # compile
print("reference on this window: adaptels crown 0.475 / patch 0.451, MRS (0.5/0.7) 0.443 / 0.463")
variants = [(kd, st, cp) for kd in kinds for st in (False, True) for cp in (False, True)]
stab = stabilise(data, valid)[0]
for kind, use_stab, compete in variants:
    z, _ = whiten(stab if use_stab else data, valid, kind=kind)
    for share in shares:
        for window in windows:
            line = []
            for k in ks:
                t = time.perf_counter()
                lab, n = tessellate(z, valid, k=k, calm_share=share, window=window, compete=compete)
                dt = time.perf_counter() - t
                mc, _ = crown_metrics(lab, crowns, valid)
                mp, _ = crown_metrics(lab, patches, valid)
                line.append((k, mc, mp, dt))
            bc = max(line, key=lambda r: r[1]["iou_mean"])
            bp = max(line, key=lambda r: r[2]["iou_mean"])
            print(f"== units {kind}{' stabilised' if use_stab else ''}, {'compete' if compete else 'first claim'}, calm share {share}, seed window {window}: best crown IoU {bc[1]['iou_mean']:.3f} at k={bc[0]:g} ({bc[1]['segments']} seg) | "
                  f"best patch IoU {bp[2]['iou_mean']:.3f} at k={bp[0]:g} ({bp[2]['segments']} seg, >=0.5: {bp[2]['iou50']:.2f}); {line[-1][3]:.2f} s per run")
            print("   k: crown/patch IoU, segments, OS/US(patch): " + "  ".join(
                f"{k:g}: {mc['iou_mean']:.2f}/{mp['iou_mean']:.2f} {mc['segments']} {mp['os']:.2f}/{mp['us']:.2f}" for k, mc, mp, _ in line), flush=True)
