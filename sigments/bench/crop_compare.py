"""Krynki 1 development window: prototype against adaptels and MRS, numbers and a picture."""
import sys, time
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import hierarchy, ladder
from data import load
from metrics import crown_metrics
import baselines as bl
from panels import panels

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
def line(name, lab, dt=0.0):
    m, _ = crown_metrics(lab, crowns, valid)
    print(f"{name:22s} seg {m['segments']:7d}  IoU {m['iou_mean']:.3f}  med {m['iou_median']:.3f}  >=.5 {m['iou50']:.3f}  OS {m['os']:.3f}  US {m['us']:.3f}  pieces {m['pieces_median']:.0f}  {dt:5.1f} s", flush=True)
    return m
res = {}
for t in (60, 110, 160, 220, 300, 400):
    t0 = time.perf_counter(); lab = bl.adaptels(data, valid, t); res[f"adaptels {t}"] = (lab, line(f"adaptels {t}", lab, time.perf_counter() - t0))
for s in (14, 18, 22, 27, 33, 40):
    t0 = time.perf_counter(); lab = bl.multiresolution(data, valid, s); res[f"MRS {s}"] = (lab, line(f"MRS {s} (0.1/0.5)", lab, time.perf_counter() - t0))
for s in (14, 18, 22, 27, 33):
    t0 = time.perf_counter(); lab = bl.multiresolution(data, valid, s, 0.5, 0.7); res[f"MRSs {s}"] = (lab, line(f"MRS {s} (0.5/0.7)", lab, time.perf_counter() - t0))
np.savez_compressed(OUT + "crop_baselines.npz", **{k.replace(" ", "_"): v[0] for k, v in res.items()})
best_a = max((k for k in res if k.startswith("adaptels")), key=lambda k: res[k][1]["iou_mean"])
best_m = max((k for k in res if k.startswith("MRS")), key=lambda k: res[k][1]["iou_mean"])
print("best:", best_a, best_m)
sub = (150, 450, 150, 450)
panels(data, valid, crowns, [("orthophoto (NIR, red, green), reference crowns in orange", None),
                             (f"{best_a}: IoU {res[best_a][1]['iou_mean']:.2f}", res[best_a][0]),
                             (f"{best_m}: IoU {res[best_m][1]['iou_mean']:.2f}", res[best_m][0])], sub,
       OUT + "crop_baselines.png")
