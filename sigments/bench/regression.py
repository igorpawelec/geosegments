"""After a code change: does the pipeline still give the label rasters saved by final_eval.py?"""
import sys
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import noise_units, tessellate
from data import load

ok = True
for scene, tag, kw, k in (("krynki1", "proto_st", dict(smooth=0.7, stabilised=True), 9), ("krynki1", "proto_wh", dict(smooth=0.7, stabilised=False), 9),
                          ("krynki2", "proto_st", dict(smooth=0.7, stabilised=True), 11)):
    sc = load(scene)
    z, valid = noise_units(sc["data"], sc["valid"], **kw)
    lab, n = tessellate(z, valid, k=k)
    old = np.load(OUT + f"labels_{scene}/{tag}_{k:g}.npz")["labels"]
    same = bool(np.array_equal(lab, old))
    ok &= same
    print(f"{scene} {tag} k={k}: {n} segments, identical to the saved raster: {same}" + ("" if same else f" ({(lab != old).mean():.4%} of pixels differ)"), flush=True)
sys.exit(0 if ok else 1)
