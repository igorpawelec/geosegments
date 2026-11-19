"""What the tessellation guarantees, checked on the Krynki 1 development window.

Determinism, invariance to rescaling single bands and to mixing them, independence of the
raster orientation. Agreement is the adjusted Rand index of the two partitions (1.0 = the
same partition).
"""
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import tessellate, whiten
from sigments.noise import stabilise
from data import load

WIN = (900, 1500, 1800, 2400)
sc = load("krynki1", window=WIN)
data, valid = sc["data"], sc["valid"]
rng = np.random.default_rng(5)


def run(d, v, use_stab=True, k=8.0):
    src = stabilise(d, v)[0] if use_stab else d
    z, _ = whiten(src, v, kind="full")
    return tessellate(z, v, k=k)[0]


def agree(a, b):
    """Adjusted Rand index of two partitions (1 = identical, 0 = no better than chance)."""
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok].astype(np.int64), b[ok].astype(np.int64)
    key = a * (b.max() + 1) + b
    _, cnt = np.unique(key, return_counts=True)
    comb = lambda v: (v.astype(np.float64) * (v - 1) / 2).sum()
    sij, sa, sb, tot = comb(cnt), comb(np.bincount(a)), comb(np.bincount(b)), a.size * (a.size - 1) / 2
    expected = sa * sb / tot
    return (sij - expected) / (0.5 * (sa + sb) - expected)


def same(a, b):
    ok = (a >= 0) & (b >= 0)
    key = a[ok].astype(np.int64) * (b.max() + 1) + b[ok]
    return len(np.unique(key)) == len(np.unique(a[ok])) == len(np.unique(b[ok]))


for use_stab in (True, False):
    name = "stabilised + whitened" if use_stab else "whitened only"
    ref = run(data, valid, use_stab)
    print(f"== {name}: {ref.max() + 1} segments")
    print(f"   same run twice:                       {agree(ref, run(data, valid, use_stab)):.4f}")
    scaled = data * np.array([1.0, 257.0, 0.01, 3.0])[:, None, None] + np.array([0.0, 1000.0, -5.0, 40.0])[:, None, None]
    print(f"   bands rescaled and shifted one by one: {agree(ref, run(scaled, valid, use_stab)):.4f}")
    a = rng.normal(size=(4, 4)) + 2 * np.eye(4)
    mixed = np.einsum("ij,jrc->irc", a, data)
    print(f"   bands mixed by a random matrix:        {agree(ref, run(mixed, valid, use_stab)):.4f}")
    rot = run(np.ascontiguousarray(np.rot90(data, 1, axes=(1, 2))), np.ascontiguousarray(np.rot90(valid, 1)), use_stab)
    print(f"   raster rotated by 90 degrees:          {agree(ref, np.ascontiguousarray(np.rot90(rot, -1))):.4f}")
    flip = run(np.ascontiguousarray(data[:, :, ::-1]), np.ascontiguousarray(valid[:, ::-1]), use_stab)
    print(f"   raster mirrored:                       {agree(ref, np.ascontiguousarray(flip[:, ::-1])):.4f}")
    noisy = np.clip(np.round(data + rng.normal(0, 1.0, size=data.shape)), 1, 255)
    print(f"   one grey level of noise added:         {agree(ref, run(noisy, valid, use_stab)):.4f}", flush=True)
