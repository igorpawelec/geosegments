"""Prototype of a segmentation with one dimensionless parameter."""
from __future__ import annotations

import numpy as np

from .basins import basins
from .merge import Hierarchy, ladder, merge_hierarchy
from .noise import stabilise, whiten
from .tessellate import tessellate

__all__ = ["segment", "noise_units", "tessellate", "whiten", "stabilise", "hierarchy", "ladder",
           "merge_hierarchy", "Hierarchy", "basins"]


def _prepare(data, valid):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[None]
    if valid is None:
        valid = np.isfinite(data).all(axis=0)
    return data, np.asarray(valid, dtype=bool)


def noise_units(data, valid=None, smooth=0.7, stabilised=True):
    """The band stack in units of its own pixel-to-pixel variability.

    ``smooth``      sigma, in pixels, of a Gaussian applied after whitening (0 = none). The
                    stack is brought back to noise units afterwards, so ``k`` keeps its meaning.
    ``stabilised``  pass every band through its variance-stabilising transform first. Helps
                    when variability depends strongly on brightness; costs the invariance to
                    mixing the bands.
    """
    data, valid = _prepare(data, valid)
    z, _ = whiten(stabilise(data, valid)[0] if stabilised else data, valid, kind="full")
    if smooth and smooth > 0:
        from scipy import ndimage
        w = np.maximum(ndimage.gaussian_filter(valid.astype(np.float64), smooth), 1e-9)
        z = np.stack([ndimage.gaussian_filter(np.where(valid, b, 0.0), smooth) / w for b in z])
        z[:, ~valid] = 0.0
        z, _ = whiten(z, valid, kind="scalar")
    return z, valid


def segment(data, k=9.0, valid=None, smooth=0.7, stabilised=True, **kw):
    """Segment ``data`` (bands, rows, cols). Returns ``(labels, n)``; -1 marks invalid pixels.

    ``k`` is the only parameter that matters: how far, in units of the image's own
    pixel-to-pixel variability, a pixel may be from the colour of the seed of its region.
    """
    z, valid = noise_units(data, valid, smooth=smooth, stabilised=stabilised)
    return tessellate(z, valid, k=k, **kw)


def hierarchy(data, levels, valid=None, keep=None, **kw) -> Hierarchy:
    """First attempt, kept for reference: region merging up a ladder of levels (see merge.py)."""
    data, valid = _prepare(data, valid)
    z, _ = whiten(data, valid)
    start = kw.pop("start", "basins")
    init = basins(z, valid) if start == "basins" else None
    return merge_hierarchy(z, valid, levels=levels, keep=keep, init=init, **kw)
