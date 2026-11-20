"""Example pictures: the prototype beside adaptels, and both over their parameter.

1. gallery_compare.png   one Krynki 1 window, both methods at the same number of segments on
                         the scene: boundaries on the orthophoto, every segment in its own
                         colour, and the distribution of segment sizes.
2. gallery_scales.png    a 37 m window: the prototype at five k, adaptels at five T. Every
                         segment is painted with its mean colour, so the panel shows what the
                         segmentation keeps of the image.
3. gallery_snp21.png     another site and another sensor (spruce, RGB), the 30 m window of the
                         package READMEs: the prototype at three k beside adaptels at their default.
"""
import colorsys
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.transform import rowcol
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT, ADAPTELS
from sigments import segment
from data import load
import baselines as bl
from panels import stretch, ORANGE, INK



def up(a, f):
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)


def edges(lab, f):
    l = up(lab, f)
    b = np.zeros(l.shape, dtype=bool)
    b[:, 1:] |= l[:, 1:] != l[:, :-1]
    b[1:, :] |= l[1:, :] != l[:-1, :]
    return b


def overlay(rgb, lab, f, line=(1.0, 1.0, 1.0)):
    img = up(rgb, f).copy()
    img[edges(lab, f)] = line
    return img


def mean_colour(rgb, lab, f):
    """Every segment painted with the mean colour of its pixels in view, thin dark boundaries."""
    _, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(lab.shape)
    n = np.bincount(inv.ravel())
    out = np.stack([(np.bincount(inv.ravel(), weights=rgb[..., c].ravel()) / n)[inv] for c in range(3)], axis=-1)
    img = up(out, f)
    img[edges(lab, f)] = (0.08, 0.08, 0.08)
    return img


def own_colour(lab, f):
    """Every segment in its own colour; hues step by the golden angle so that neighbours differ."""
    ids, inv = np.unique(lab, return_inverse=True)
    pal = np.array([colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.55 + 0.35 * ((i * 37) % 7) / 6, 0.62 + 0.33 * ((i * 53) % 5) / 4) for i in range(ids.size)])
    return up(pal[inv].reshape(lab.shape + (3,)), f)


def show(ax, img, caption, outline=None, f=1, size=8.5):
    ax.imshow(img, interpolation="nearest")
    if outline is not None:
        ax.contour(up(outline, f), levels=[0.5], colors=[ORANGE], linewidths=1.1)
    ax.set_axis_off()
    ax.text(0.5, -0.025, caption, transform=ax.transAxes, ha="center", va="top", fontsize=size, color=INK, linespacing=1.35)


def count(lab):
    return len(np.unique(lab[lab >= 0]))


# ---------------------------------------------------------------- Krynki 1, comparison
sc = load("krynki1")
get = lambda name: np.load(OUT + f"labels_krynki1/{name}.npz")["labels"]
r0, r1, c0, c1 = 1050, 1350, 1950, 2250
rgb = stretch(sc["data"][[3, 0, 1]][:, r0:r1, c0:c1], sc["valid"][r0:r1, c0:c1])
crowns = sc["crowns"][r0:r1, c0:c1] > 0
proto, adap = get("proto_st_9"), get("adaptels_260")
F = 3
fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.4), dpi=120)
show(axs[0, 0], up(rgb, F), "orthophoto (NIR, red, green), 75 x 75 m; reference dead crowns in orange", crowns, F)
show(axs[0, 1], overlay(rgb, proto[r0:r1, c0:c1], F), f"prototype, k = 9: {count(proto):,} segments on the scene, {count(proto[r0:r1, c0:c1])} here", crowns, F)
show(axs[0, 2], overlay(rgb, adap[r0:r1, c0:c1], F), f"adaptels, T = 260: {count(adap):,} segments on the scene, {count(adap[r0:r1, c0:c1])} here", crowns, F)
sizes_p = np.bincount(proto[proto >= 0])
sizes_a = np.bincount(adap[adap >= 0])
ax = axs[1, 0]
bins = np.logspace(0, 5.5, 45)
ax.hist(sizes_p[sizes_p > 0], bins=bins, histtype="step", color=ORANGE, lw=2, label="prototype")
ax.hist(sizes_a[sizes_a > 0], bins=bins, histtype="step", color="#0d366b", lw=2, label="adaptels")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("segment size (pixels)")
ax.set_ylabel("number of segments on the scene")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, color="#e6e6e6", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_box_aspect(1)
show(axs[1, 1], own_colour(proto[r0:r1, c0:c1], F), "prototype: every segment in its own colour")
show(axs[1, 2], own_colour(adap[r0:r1, c0:c1], F), "adaptels: every segment in its own colour")
plt.subplots_adjust(left=0.05, right=0.995, top=0.995, bottom=0.04, wspace=0.04, hspace=0.09)
fig.savefig(OUT + "gallery_compare.png", facecolor="white")
plt.close(fig)
print("sizes on the scene: prototype median %d px, largest %d px, 90 %% of segments within %d-%d px | adaptels median %d px, largest %d px, 90 %% within %d-%d px" % (
    np.median(sizes_p[sizes_p > 0]), sizes_p.max(), *np.percentile(sizes_p[sizes_p > 0], [5, 95]),
    np.median(sizes_a[sizes_a > 0]), sizes_a.max(), *np.percentile(sizes_a[sizes_a > 0], [5, 95])))

# ---------------------------------------------------------------- Krynki 1, scales
r0, r1, c0, c1 = 1140, 1290, 2100, 2250
F = 5
rgb = stretch(sc["data"][[3, 0, 1]][:, r0:r1, c0:c1], sc["valid"][r0:r1, c0:c1])
crowns = sc["crowns"][r0:r1, c0:c1] > 0
fig, axs = plt.subplots(2, 6, figsize=(22.5, 8.3), dpi=110)
show(axs[0, 0], up(rgb, F), "orthophoto (NIR, red, green)\n37 x 37 m", None, F, 10)
show(axs[1, 0], up(rgb, F), "the same, reference dead crowns\noutlined in orange", crowns, F, 10)
for ax, k in zip(axs[0, 1:], (5, 7, 9, 12, 16)):
    lab = get(f"proto_st_{k}")
    show(ax, mean_colour(rgb, lab[r0:r1, c0:c1], F), f"prototype, k = {k}\n{count(lab[r0:r1, c0:c1])} segments in view, {count(lab):,} on the scene", None, F, 10)
for ax, t in zip(axs[1, 1:], (80, 150, 260, 340, 450)):
    lab = get(f"adaptels_{t}")
    show(ax, mean_colour(rgb, lab[r0:r1, c0:c1], F), f"adaptels, T = {t}\n{count(lab[r0:r1, c0:c1])} segments in view, {count(lab):,} on the scene", None, F, 10)
plt.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.065, wspace=0.02, hspace=0.13)
fig.savefig(OUT + "gallery_scales.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- SNP_21 (spruce, RGB)
T = ADAPTELS + "test_data/"
with rasterio.open(T + "SNP_21_2020_1.tif") as src:
    d = src.read().astype(np.float64)
    tf = src.transform
    nodata = src.nodata
valid = np.isfinite(d).all(axis=0) & ~(d == 0).all(axis=0)
if nodata is not None and not np.isnan(nodata):
    valid &= ~(d == nodata).all(axis=0)
d = np.where(valid, d, 0.0)
pts = gpd.read_file(T + "dead_trees_test.shp")
pr, pc = rowcol(tf, pts.geometry.x.values, pts.geometry.y.values)
pr, pc = np.asarray(pr), np.asarray(pc)
r0, r1, c0, c1 = 120, 240, 90, 210                      # the 30 m window of the package READMEs
F = 6
img = stretch(d[:3], valid)[r0:r1, c0:c1]
inwin = (pr >= r0) & (pr < r1) & (pc >= c0) & (pc < c1)
runs = [(f"prototype, k = {k}", segment(d[:3], k=k, valid=valid)[0]) for k in (6, 9, 13)]
runs.append(("adaptels, T = 60 (default)", bl.adaptels(d[:3], valid, 60)))
fig, axs = plt.subplots(1, 5, figsize=(21, 4.75), dpi=110)
show(axs[0], up(img, F), f"SNP_21, spruce, RGB, 30 x 30 m\n{int(inwin.sum())} of the {len(pts)} digitised dead trees", None, F, 10)
for ax, (name, lab) in zip(axs[1:], runs):
    lab = np.where(valid, lab, -1)
    hit = lab[pr, pc]
    alone = sum(int((hit == h).sum() == 1) for h in hit)
    med = int(np.median([(lab == h).sum() for h in hit]))
    show(ax, mean_colour(img, lab[r0:r1, c0:c1], F), f"{name}: {count(lab):,} segments on the plot\n{alone} of {len(pts)} trees alone in a segment, median {med} px", None, F, 10)
for ax in axs:
    ax.scatter((pc[inwin] - c0) * F + F / 2, (pr[inwin] - r0) * F + F / 2, s=26, c=ORANGE, edgecolors="white", linewidths=0.8)
plt.subplots_adjust(left=0.003, right=0.997, top=0.995, bottom=0.115, wspace=0.02)
fig.savefig(OUT + "gallery_snp21.png", facecolor="white")
plt.close(fig)
print("saved gallery_compare.png, gallery_scales.png, gallery_snp21.png")
