"""Side-by-side panels: one window of a scene, segment boundaries of several label rasters, reference crowns."""
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORANGE, INK, BLUE = "#eb6834", "#1f1f1f", "#2a6fbb"


def stretch(img, valid, lo=2, hi=98):
    out = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        v = img[i][valid]
        a, b = np.percentile(v, [lo, hi]) if v.size else (0, 1)
        out[i] = np.clip((img[i] - a) / max(b - a, 1e-6), 0, 1)
    return np.moveaxis(out, 0, -1)


def boundaries(lab):
    b = np.zeros(lab.shape, dtype=bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]
    return b


def panels(data, valid, crowns, labelled, window, path, bands=(3, 0, 1), title_extra=None):
    """``labelled``: list of (caption, label raster or None). ``window``: (r0, r1, c0, c1)."""
    r0, r1, c0, c1 = window
    rgb = stretch(data[list(bands)][:, r0:r1, c0:c1], valid[r0:r1, c0:c1])
    cr = crowns[r0:r1, c0:c1]
    n = len(labelled)
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 5.0), dpi=150)
    for ax, (cap, lab) in zip(np.atleast_1d(axs), labelled):
        img = rgb.copy()
        if lab is not None:
            img[boundaries(lab[r0:r1, c0:c1])] = (1.0, 1.0, 1.0)
        ax.imshow(img, interpolation="nearest")
        ax.contour(cr > 0, levels=[0.5], colors=[ORANGE], linewidths=1.1)
        ax.set_axis_off()
        ax.text(0.5, -0.03, cap, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.09, wspace=0.03)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
