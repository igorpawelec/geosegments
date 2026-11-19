"""Picture of chosen prototype levels beside the best baseline on the development window."""
import sys
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import hierarchy, ladder
from data import load
from metrics import crown_metrics
from panels import panels

WIN = (900, 1500, 1800, 2400)
mode = sys.argv[1]; ks = [float(v) for v in sys.argv[2].split(",")]; out = sys.argv[3]
kw = {}
for a in sys.argv[4:]:
    key, val = a.split("="); kw[key] = val if key == "start" else float(val)
sc = load("krynki1", window=WIN)
base = np.load(OUT + "crop_baselines.npz")
h = hierarchy(sc["data"], ladder(max(ks) * 1.01, k_min=0.71), valid=sc["valid"], mode=mode, **kw)
items = []
for k in ks:
    lab = h.at(k); m, _ = crown_metrics(lab, sc["crowns"], sc["valid"])
    items.append((f"{mode} k={k:g}: {m['segments']} segments, IoU {m['iou_mean']:.2f}", lab))
lab = base["adaptels_220"]; m, _ = crown_metrics(lab, sc["crowns"], sc["valid"])
items.append((f"adaptels 220: {m['segments']} segments, IoU {m['iou_mean']:.2f}", lab))
panels(sc["data"], sc["valid"], sc["crowns"], items, (150, 450, 150, 450), out)
print("saved", out)
