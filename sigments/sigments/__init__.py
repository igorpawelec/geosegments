"""Prototype of a segmentation with one dimensionless parameter."""
from __future__ import annotations

import numpy as np

from .basins import basins
from .merge import Hierarchy, ladder, merge_hierarchy
from .noise import stabilise, whiten
from .tessellate import tessellate

__version__ = "0.1.2"

__all__ = ["segment", "noise_units", "tessellate", "whiten", "stabilise", "hierarchy", "ladder",
           "merge_hierarchy", "Hierarchy", "basins"]

# Bins of the variability curve that noise.stabilise fits to every band. The curve needs
# at least one pair of neighbouring pixels in each, so this is also the smallest number
# of pairs a stabilised run can work from.
_BINS = 24


def _prepare(data, valid):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[None]
    # A mask says which pixels to use; it cannot make a pixel that is not a
    # number usable. Taken as given, a NaN under a True of the mask went into
    # the noise estimate, and a scene that is one segment came out as hundreds.
    finite = np.isfinite(data).all(axis=0)
    valid = finite if valid is None else np.asarray(valid, dtype=bool) & finite
    return data, valid


def _neighbours(data, valid):
    """``(pairs, identical)``: the pairs of 4-adjacent valid pixels, which the noise unit is
    estimated from, and how many of them hold the same values in every band."""
    hm = valid[:, 1:] & valid[:, :-1]
    vm = valid[1:, :] & valid[:-1, :]
    same_h = (data[:, :, 1:] == data[:, :, :-1]).all(axis=0) & hm
    same_v = (data[:, 1:, :] == data[:, :-1, :]).all(axis=0) & vm
    return int(hm.sum()) + int(vm.sum()), int(same_h.sum()) + int(same_v.sum())


def _no_unit(why):
    return ValueError("sigments measures in units of the image's own pixel-to-pixel variability "
                      f"and cannot estimate that unit here: {why}.")


def noise_units(data, valid=None, smooth=0.7, stabilised=True):
    """The band stack in units of its own pixel-to-pixel variability.

    ``smooth``      sigma, in pixels, of a Gaussian applied after whitening (0 = none). The
                    stack is brought back to noise units afterwards, so ``k`` keeps its meaning.
    ``stabilised``  pass every band through its variance-stabilising transform first. Helps
                    when variability depends strongly on brightness; costs the invariance to
                    mixing the bands.

    Raises ``ValueError``, with the reason, for an image that has no such unit to measure:
    no valid pixel, no two valid pixels next to each other, neighbouring pixels that do
    not differ anywhere, fewer pairs of neighbours than the variability curve has bins
    (``stabilised`` only), or an estimate that does not come out as a finite, positive
    number.
    """
    data, valid = _prepare(data, valid)
    n = int(valid.sum())
    pairs, identical = _neighbours(data, valid)
    # Refused before anything is estimated: these have no unit, whatever the settings. Left
    # to the estimate, some of them stopped inside it and some came out with a result made
    # of nothing (a constant image without smoothing: one segment, from a unit of 1e-312).
    if n == 0:
        raise _no_unit("it has no valid pixel")
    if pairs == 0:
        raise _no_unit(f"no two of its {n} valid pixels are neighbours")
    if identical == pairs:
        raise _no_unit("neighbouring pixels do not differ anywhere, as in a constant image")
    if stabilised and pairs < _BINS:
        raise _no_unit(f"{n} valid pixels in {pairs} pairs of neighbours are too few for the "
                       f"variability curve, which needs {_BINS} pairs")
    # What is left can still fail inside the estimate: a covariance that is not positive
    # definite, a median difference of zero after smoothing. To the caller that is the same
    # thing, an image without a usable unit, and it is reported as that, with what is known.
    try:
        with np.errstate(all="ignore"):
            z, _ = whiten(stabilise(data, valid, bins=_BINS)[0] if stabilised else data, valid, kind="full")
            if smooth and smooth > 0:
                from scipy import ndimage
                w = np.maximum(ndimage.gaussian_filter(valid.astype(np.float64), smooth), 1e-9)
                z = np.stack([ndimage.gaussian_filter(np.where(valid, b, 0.0), smooth) / w for b in z])
                z[:, ~valid] = 0.0
                z, _ = whiten(z, valid, kind="scalar")
    except (IndexError, ZeroDivisionError, FloatingPointError, RuntimeWarning, np.linalg.LinAlgError) as e:
        if 2 * identical > pairs:
            raise _no_unit(f"more than half of its neighbouring pixels are identical ({identical} of "
                           f"{pairs} pairs), so their typical difference is zero") from e
        raise _no_unit("the estimate of it failed (" + type(e).__name__ + ": " + str(e) + ")") from e
    if not np.isfinite(z[:, valid]).all():
        raise _no_unit("the estimate of it did not come out finite; the values may be too large to square")
    return z, valid


def segment(data, k=9.0, valid=None, smooth=0.7, stabilised=True, **kw):
    """Segment ``data`` (bands, rows, cols). Returns ``(labels, n)``; -1 marks invalid pixels.

    ``k`` is the only parameter that matters: how far, in units of the image's own
    pixel-to-pixel variability, a pixel may be from the colour of the seed of its region.

    Raises ``ValueError`` when ``k`` is not a positive number, and when the image has no
    such unit to measure (see :func:`noise_units`).
    """
    try:
        as_number = float(k)
    except (TypeError, ValueError):
        as_number = float("nan")
    if not (np.isfinite(as_number) and as_number > 0):
        raise ValueError(f"k must be a positive number, got {k!r}: it is how far, in units of the "
                         "image's own pixel-to-pixel variability, a pixel may be from the colour of "
                         "the seed of its region.")
    z, valid = noise_units(data, valid, smooth=smooth, stabilised=stabilised)
    return tessellate(z, valid, k=k, **kw)


def hierarchy(data, levels, valid=None, keep=None, **kw) -> Hierarchy:
    """First attempt, kept for reference: region merging up a ladder of levels (see merge.py)."""
    data, valid = _prepare(data, valid)
    z, _ = whiten(data, valid)
    start = kw.pop("start", "basins")
    init = basins(z, valid) if start == "basins" else None
    return merge_hierarchy(z, valid, levels=levels, keep=keep, init=init, **kw)
