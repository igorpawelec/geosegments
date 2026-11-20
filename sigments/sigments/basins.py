"""Starting regions: the catchment basins of the gradient, in noise units.

Merging single pixels pair by pair lets the mixed pixels of a blurred edge find each other
first and form thin rings around objects. A watershed of the gradient magnitude hands every
pixel of such a ramp to the side it descends to, so the starting regions already end on the
crest of the edge.
"""
from __future__ import annotations

import numpy as np


def gradient_energy(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Squared gradient magnitude from central differences, summed over the bands.

    Invalid neighbours and the raster border are replaced by the pixel itself.
    """
    nb, rows, cols = z.shape
    g = np.zeros((rows, cols))
    for axis in (1, 2):
        fwd = np.roll(z, -1, axis=axis)
        bwd = np.roll(z, 1, axis=axis)
        use_f = np.roll(valid, -1, axis=axis - 1).copy()
        use_b = np.roll(valid, 1, axis=axis - 1).copy()
        if axis == 1:
            use_f[-1, :] = False
            use_b[0, :] = False
        else:
            use_f[:, -1] = False
            use_b[:, 0] = False
        fwd = np.where(use_f, fwd, z)
        bwd = np.where(use_b, bwd, z)
        span = use_f.astype(np.float64) + use_b.astype(np.float64)
        d = (fwd - bwd) / np.maximum(span, 1.0)
        g += (d * d).sum(axis=0)
    g[~valid] = 0.0
    return g


def basins(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Label raster of the watershed basins of the gradient (int32, -1 = invalid)."""
    from skimage.segmentation import watershed
    g = np.sqrt(gradient_energy(z, valid))
    lab = watershed(g, connectivity=1, mask=valid)
    out = lab.astype(np.int32) - 1
    out[~valid] = -1
    return out
