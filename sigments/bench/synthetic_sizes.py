"""Is the parameter a size? Blobs of one contrast and eight sizes on a textured background.

Every method is tuned on the middle size and then judged on all sizes with that one
setting. A parameter that means "tolerance" keeps working; a parameter that means "size"
fits one size and misses the others. Also reported: the setting that would have been best
for each size on its own.
"""
import csv
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import noise_units, tessellate
import baselines as bl

rng = np.random.default_rng(11)
RADII = (1.6, 2.5, 4, 6, 9, 14, 20, 28)
REPS = 12
CONTRAST = 30.0          # grey levels between a blob and its surroundings, along a random colour direction
cell = 96
H = cell * REPS
W = cell * len(RADII)

# background: smooth texture (correlated) plus white noise, three bands with a common brightness component
base = ndimage.gaussian_filter(rng.normal(size=(H, W)), 6) * 60
tex = np.stack([base + ndimage.gaussian_filter(rng.normal(size=(H, W)), 2) * 12 for _ in range(3)])
truth = np.zeros((H, W), dtype=np.int32)
size_of = {}
yy, xx = np.mgrid[0:H, 0:W]
img = tex + 110.0
bid = 0
for j, rad in enumerate(RADII):
    for i in range(REPS):
        bid += 1
        cy = i * cell + cell / 2 + rng.uniform(-6, 6)
        cx = j * cell + cell / 2 + rng.uniform(-6, 6)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
        truth[m] = bid
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        img[:, m] += (direction * CONTRAST)[:, None]
        size_of[bid] = (j, int(m.sum()))
img = np.stack([ndimage.gaussian_filter(b, 0.8) for b in img])          # optics
img += rng.normal(0, 2.0, size=img.shape)                               # sensor noise
img = np.clip(np.round(img), 0, 255)
valid = np.ones((H, W), dtype=bool)
z, _ = noise_units(img, valid, smooth=0.7, stabilised=True)          # the pipeline of sigments.segment
z0, _ = noise_units(img, valid, smooth=0.0, stabilised=True)         # the same without the smoothing step
print(f"scene {H} x {W}, {bid} blobs, radii {RADII}, median blob size per column: {[int(np.median([s for (c, s) in size_of.values() if c == j])) for j in range(len(RADII))]}")


def per_blob_iou(lab):
    nseg = int(lab.max()) + 1
    seg_size = np.bincount(lab.ravel(), minlength=nseg)
    inside = truth > 0
    key = truth[inside].astype(np.int64) * nseg + lab[inside]
    uk, cnt = np.unique(key, return_counts=True)
    b = uk // nseg
    s = uk % nseg
    out = np.zeros(bid + 1)
    x = np.bincount(truth[inside], minlength=bid + 1)
    for bb, ss, c in zip(b, s, cnt):
        out[bb] = max(out[bb], c / (x[bb] + seg_size[ss] - c))
    return out[1:]


col = np.array([size_of[b][0] for b in range(1, bid + 1)])
METHODS = {
    "prototype (k)": ([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20], lambda p: tessellate(z, valid, k=p)[0]),
    "prototype without smoothing (k)": ([3, 4, 5, 6, 7, 8, 9, 10, 12, 14], lambda p: tessellate(z0, valid, k=p)[0]),
    "adaptels (T)": ([15, 22, 32, 45, 64, 90, 128, 180, 256, 360, 512, 720], lambda p: bl.adaptels(img, valid, p)),
    "MRS 0.1/0.5 (scale)": ([8, 11, 16, 22, 32, 45, 64, 90, 128, 180], lambda p: bl.multiresolution(img, valid, p)),
    "MRS 0.5/0.7 (scale)": ([8, 11, 16, 22, 32, 45, 64, 90, 128, 180], lambda p: bl.multiresolution(img, valid, p, 0.5, 0.7)),
}
MID = 3                                                                  # tune on radius 6
rows = []
curves = {}
for name, (grid, run) in METHODS.items():
    table = np.array([[per_blob_iou(run(p))[col == j].mean() for j in range(len(RADII))] for p in grid])
    tuned = int(np.argmax(table[:, MID]))
    best_each = [grid[int(np.argmax(table[:, j]))] for j in range(len(RADII))]
    curves[name] = (table[tuned], table.max(axis=0), grid[tuned], best_each)
    print(f"== {name}: tuned on radius {RADII[MID]} -> {grid[tuned]}")
    print("   IoU by radius at that setting: " + "  ".join(f"{r}: {v:.2f}" for r, v in zip(RADII, table[tuned])))
    print("   best setting for each radius:  " + "  ".join(f"{r}: {b}" for r, b in zip(RADII, best_each)))
    print("   IoU with the best setting each: " + "  ".join(f"{v:.2f}" for v in table.max(axis=0)), flush=True)
    for j, r in enumerate(RADII):
        rows.append({"method": name, "radius": r, "iou_tuned_on_middle": round(float(table[tuned, j]), 4), "iou_best_for_this_size": round(float(table[:, j].max()), 4),
                     "tuned_setting": grid[tuned], "best_setting_for_this_size": best_each[j]})
with open(OUT + "synthetic_sizes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
np.savez_compressed(OUT + "synthetic_sizes_scene.npz", img=img.astype(np.uint8), truth=truth)

colors = {"prototype (k)": "#eb6834", "prototype without smoothing (k)": "#f3a683", "adaptels (T)": "#0d366b", "MRS 0.1/0.5 (scale)": "#5b9bd9", "MRS 0.5/0.7 (scale)": "#9cc3ee"}
areas = [int(np.median([s for (c, s) in size_of.values() if c == j])) for j in range(len(RADII))]
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
for name, (tuned, best, setting, _) in curves.items():
    ax.plot(areas, tuned, marker="o", ms=5, lw=2, color=colors[name], ls="--" if "without" in name else "-", label=f"{name.split(' (')[0]}, one setting ({setting:g})")
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel("object size (pixels)")
ax.set_ylabel("mean IoU of the object with its best segment")
ax.axvline(areas[MID], color="#8a8a8a", lw=0.8, ls=":")
ax.text(areas[MID] * 1.05, 0.03, "size the setting was tuned on", fontsize=7.5, color="#555555")
ax.grid(True, color="#e6e6e6", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(OUT + "synthetic_sizes.png", facecolor="white")
print("saved figure and csv")
