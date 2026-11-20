"""The prototype as shipped in ``sigments.segment`` on a full scene, over k. Writes results/final_<scene>.csv."""
import csv
import os
import sys
import time
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import noise_units, tessellate
from data import load
from metrics import crown_metrics

scene = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "proto"
opts = {}
for a in sys.argv[3:]:
    key, val = a.split("=")
    opts[key] = float(val) if key == "smooth" else (val == "1")
sc = load(scene)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, npatch = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
os.makedirs(OUT + f"labels_{scene}", exist_ok=True)
tessellate(np.zeros((1, 32, 32)), np.ones((32, 32), dtype=bool), 6.0)
t = time.perf_counter()
z, _ = noise_units(data, valid, **opts)
t_units = time.perf_counter() - t
rows = []
for k in (5, 6, 7, 8, 9, 10, 11, 12, 14, 16):
    t = time.perf_counter()
    lab, n = tessellate(z, valid, k=k)
    dt = time.perf_counter() - t
    mc, _ = crown_metrics(lab, crowns, valid)
    mp, _ = crown_metrics(lab, patches, valid)
    row = {"scene": scene, "algorithm": tag, "param": k, "seconds": round(dt + t_units, 2), "segments": mc["segments"],
           "one_px_segments": mc["one_px_segments"], "seg_px_median": mc["seg_px_median"]}
    row.update({"crown_" + key: round(v, 4) for key, v in mc.items() if key in ("iou_mean", "iou_median", "iou50", "f1_area", "os", "us", "afi_median", "pieces_median")})
    row.update({"patch_" + key: round(v, 4) for key, v in mp.items() if key in ("iou_mean", "iou_median", "iou50", "f1_area", "os", "us", "afi_median", "pieces_median")})
    rows.append(row)
    np.savez_compressed(OUT + f"labels_{scene}/{tag}_{k:g}.npz", labels=lab)
    print(f"{scene} {tag} k={k:4g}  seg {mc['segments']:7d}  crown IoU {mc['iou_mean']:.3f} (>=.5 {mc['iou50']:.2f})  patch IoU {mp['iou_mean']:.3f} (>=.5 {mp['iou50']:.2f})  "
          f"OS/US patch {mp['os']:.2f}/{mp['us']:.2f}  units {t_units:.1f} s + segmentation {dt:.1f} s", flush=True)
with open(OUT + f"final_{scene}_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("DONE")
