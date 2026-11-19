"""Full-scene evaluation: the prototype, adaptels and multiresolution segmentation over their parameters.

Scores every run against the reference crowns and against the patches they form when
touching crowns are joined. Writes results/full_<scene>.csv and the label rasters.
"""
import csv
import os
import sys
import time
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import tessellate, whiten
from sigments.noise import stabilise
from data import load
from metrics import crown_metrics
import baselines as bl

scene = sys.argv[1]
which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["proto", "adaptels", "mrs_shape", "mrs"]
sc = load(scene)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
print(f"{scene}: {sc['n_crowns']} crowns in {npatch} patches, valid share {valid.mean():.3f}", flush=True)
os.makedirs(OUT + f"labels_{scene}", exist_ok=True)

t = time.perf_counter()
z, _ = whiten(stabilise(data, valid)[0], valid, kind="full")
t_units = time.perf_counter() - t
tessellate(z[:, :64, :64], valid[:64, :64], 6.0)
GRID = {
    "proto": [4, 5, 6, 6.5, 7, 7.5, 8, 9, 10, 12],
    "adaptels": [40, 60, 80, 110, 150, 200, 260, 340, 450],
    "mrs_shape": [14, 18, 22, 27, 33, 40, 50, 65, 80, 100],
    "mrs": [14, 18, 22, 27, 33, 40, 50, 65, 80, 100],
}
RUN = {
    "proto": lambda p: tessellate(z, valid, k=p)[0],
    "adaptels": lambda p: bl.adaptels(data, valid, p),
    "mrs_shape": lambda p: bl.multiresolution(data, valid, p, shape=0.5, compactness=0.7),
    "mrs": lambda p: bl.multiresolution(data, valid, p, shape=0.1, compactness=0.5),
}
rows = []
for alg in which:
    for p in GRID[alg]:
        t = time.perf_counter()
        lab = RUN[alg](p)
        dt = time.perf_counter() - t + (t_units if alg == "proto" else 0.0)
        mc, _ = crown_metrics(lab, crowns, valid)
        mp, _ = crown_metrics(lab, patches, valid)
        row = {"scene": scene, "algorithm": alg, "param": p, "seconds": round(dt, 2), "segments": mc["segments"],
               "one_px_segments": mc["one_px_segments"], "seg_px_median": mc["seg_px_median"]}
        row.update({"crown_" + k: round(v, 4) for k, v in mc.items() if k in ("iou_mean", "iou_median", "iou50", "f1_area", "os", "us", "afi_median", "pieces_median")})
        row.update({"patch_" + k: round(v, 4) for k, v in mp.items() if k in ("iou_mean", "iou_median", "iou50", "f1_area", "os", "us", "afi_median", "pieces_median")})
        rows.append(row)
        np.savez_compressed(OUT + f"labels_{scene}/{alg}_{p:g}.npz", labels=lab)
        print(f"{scene} {alg:10s} {p:6g}  seg {mc['segments']:7d}  crown IoU {mc['iou_mean']:.3f} (>=.5 {mc['iou50']:.2f})  patch IoU {mp['iou_mean']:.3f} (>=.5 {mp['iou50']:.2f})  "
              f"OS/US patch {mp['os']:.2f}/{mp['us']:.2f}  {dt:6.1f} s", flush=True)
        with open(OUT + f"full_{scene}_{'-'.join(which)}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
print("DONE", flush=True)
