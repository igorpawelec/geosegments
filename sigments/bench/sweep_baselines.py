"""Crown metrics of adaptels, multiresolution segmentation and SICLE over their parameter ranges."""
import csv, sys, time
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from data import load
from metrics import crown_metrics
import baselines as bl
from sigments import whiten

scene = sys.argv[1]
which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["adaptels", "adaptels_noise", "mrs", "mrs_shape", "sicle"]
sc = load(scene)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
z, _ = whiten(data, valid)
GRID = {
    "adaptels": [12, 16, 20, 25, 30, 40, 50, 60, 80, 110],
    "adaptels_noise": [3, 4, 6, 8, 11, 16, 22, 32, 45, 64],
    "mrs": [6, 8, 11, 14, 18, 22, 27, 33, 40, 50, 65],
    "mrs_shape": [8, 11, 14, 18, 22, 27, 33, 40, 50],
    "sicle": [4000, 8000, 16000, 32000, 64000],
}
RUN = {
    "adaptels": lambda p: bl.adaptels(data, valid, p),
    "adaptels_noise": lambda p: bl.adaptels(z, valid, p),
    "mrs": lambda p: bl.multiresolution(data, valid, p, shape=0.1, compactness=0.5),
    "mrs_shape": lambda p: bl.multiresolution(data, valid, p, shape=0.5, compactness=0.7),
    "sicle": lambda p: bl.sicle(data, valid, p),
}
rows = []
for alg in which:
    for p in GRID[alg]:
        t = time.perf_counter()
        try:
            lab = RUN[alg](p)
        except Exception as e:                      # keep the sweep alive, report the failure
            print(f"{scene} {alg} {p}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        dt = time.perf_counter() - t
        m, _ = crown_metrics(lab, crowns, valid)
        rows.append({"scene": scene, "algorithm": alg, "param": p, "seconds": round(dt, 1), **m})
        print(f"{scene} {alg:15s} {p:7g}  seg {m['segments']:8d}  IoU {m['iou_mean']:.3f}  IoU>=.5 {m['iou50']:.3f}  OS {m['os']:.3f}  US {m['us']:.3f}  {dt:6.1f} s", flush=True)
        with open(OUT + f"baselines_{scene}_{'-'.join(which)}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")
