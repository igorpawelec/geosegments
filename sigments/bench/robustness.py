"""How much does each segmenter change when the input changes in ways that should not matter?

Each method runs with the setting that was best on this window, then again on a perturbed
copy with the same setting. Agreement of the two results: boundary F1 (a boundary pixel
counts as matched when the other result has one within a pixel) and the adjusted Rand index.
"""
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import segment
from data import load
import baselines as bl
from panels import boundaries

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid = sc["data"], sc["valid"]
rng = np.random.default_rng(5)


METHODS = {
    "prototype, default (k=9)": lambda d, v: segment(d, k=9.0, valid=v, smooth=0.7, stabilised=True)[0],
    "prototype, whitened + smoothed (k=9)": lambda d, v: segment(d, k=9.0, valid=v, smooth=0.7, stabilised=False)[0],
    "prototype, whitened only (k=8)": lambda d, v: segment(d, k=8.0, valid=v, smooth=0.0, stabilised=False)[0],
    "adaptels (T=220)": lambda d, v: bl.adaptels(d, v, 220),
    "MRS 0.5/0.7 (scale 33)": lambda d, v: bl.multiresolution(d, v, 33, 0.5, 0.7),
}


def ari(a, b):
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok].astype(np.int64), b[ok].astype(np.int64)
    _, cnt = np.unique(a * (b.max() + 1) + b, return_counts=True)
    comb = lambda v: (v.astype(np.float64) * (v - 1) / 2).sum()
    sij, sa, sb, tot = comb(cnt), comb(np.bincount(a)), comb(np.bincount(b)), a.size * (a.size - 1) / 2
    e = sa * sb / tot
    return (sij - e) / (0.5 * (sa + sb) - e)


def bf1(a, b):
    ba, bb = boundaries(a), boundaries(b)
    s = np.ones((3, 3), dtype=bool)
    p = (ba & ndimage.binary_dilation(bb, s)).sum() / max(ba.sum(), 1)
    r = (bb & ndimage.binary_dilation(ba, s)).sum() / max(bb.sum(), 1)
    return 2 * p * r / max(p + r, 1e-12)


noisy = np.clip(np.round(data + rng.normal(0, 1.0, size=data.shape)), 1, 255)
gain = data * 1.3
scaled = data * np.array([1.0, 257.0, 0.01, 3.0])[:, None, None] + np.array([0.0, 1000.0, -5.0, 40.0])[:, None, None]
mix = np.einsum("ij,jrc->irc", rng.normal(size=(4, 4)) + 2 * np.eye(4), data)
rot_d, rot_v = np.ascontiguousarray(np.rot90(data, 1, axes=(1, 2))), np.ascontiguousarray(np.rot90(valid, 1))
CASES = [
    ("noise of one grey level", lambda f: f(noisy, valid)),
    ("raster rotated by 90 degrees", lambda f: np.ascontiguousarray(np.rot90(f(rot_d, rot_v), -1))),
    ("all bands brighter by 30 %", lambda f: f(gain, valid)),
    ("bands rescaled one by one", lambda f: f(scaled, valid)),
    ("bands mixed by a random matrix", lambda f: f(mix, valid)),
]
print(f"{'':40s}" + "".join(f"{c[0]:>32s}" for c in CASES))
print(f"{'method (segments)':40s}" + "".join(f"{'boundary F1 / ARI':>32s}" for _ in CASES))
for name, f in METHODS.items():
    ref = f(data, valid)
    cells = []
    for _, case in CASES:
        try:
            other = case(f)
            cells.append(f"{bf1(ref, other):.2f} / {ari(ref, other):.2f}")
        except Exception as e:
            cells.append(type(e).__name__)
    print(f"{name + ' (' + str(int(ref.max()) + 1) + ')':40s}" + "".join(f"{c:>32s}" for c in cells), flush=True)
