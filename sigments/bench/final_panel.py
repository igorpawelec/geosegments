"""One window of a scene: orthophoto, the prototype, adaptels and MRS at their best settings."""
import sys
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from data import load
from metrics import crown_metrics
from panels import panels

scene = sys.argv[1]
runs = [r.split("=") for r in sys.argv[2].split(",")]          # e.g. "prototype k=10=proto_st_10,adaptels 200=adaptels_200"
window = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (1050, 1350, 1950, 2250)
sc = load(scene)
data, valid, crowns = sc["data"], sc["valid"], sc["crowns"]
patches, _ = ndimage.label(crowns > 0, structure=np.ones((3, 3)))
r0, r1, c0, c1 = window
items = [("orthophoto (NIR, red, green); reference crowns in orange", None)]
for parts in runs:
    name, file = "=".join(parts[:-1]), parts[-1]
    lab = np.load(OUT + f"labels_{scene}/{file}.npz")["labels"]
    m, _ = crown_metrics(lab, patches, valid)
    inside = len(np.unique(lab[r0:r1, c0:c1]))
    items.append((f"{name}: {inside} segments in view, patch IoU {m['iou_mean']:.2f} on the scene", lab))
panels(data, valid, crowns, items, window, OUT + f"panel_{scene}.png")
print("saved", OUT + f"panel_{scene}.png")
