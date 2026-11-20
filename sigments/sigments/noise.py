"""Put a band stack into units of its own pixel-to-pixel variability.

The covariance of the differences between adjacent pixels is estimated robustly and the
stack is transformed so that this covariance becomes the identity. After the transform a
distance of 1 along any direction of the feature space is "one typical difference between
two neighbouring pixels", whatever the bands were: 8- or 16-bit, reflectance, a colour
space, a canopy height model stacked on an orthophoto.

The estimator uses only Mahalanobis distances, so it is affine equivariant: for any
invertible matrix A and offset b, the covariance of A x + b is A S A'. Distances measured
after whitening are therefore identical for the original bands and for any invertible
linear recombination of them.
"""
from __future__ import annotations

import numpy as np

# Median of the chi-square distribution with 1..12 degrees of freedom.
_CHI2_MEDIAN = (0.454936, 1.386294, 2.365974, 3.356694, 4.351460, 5.348121,
                6.345811, 7.344121, 8.342833, 9.341818, 10.340998, 11.340322)


def _chi2_median(dof: int) -> float:
    if dof <= len(_CHI2_MEDIAN):
        return _CHI2_MEDIAN[dof - 1]
    return dof * (1.0 - 2.0 / (9.0 * dof)) ** 3          # Wilson-Hilferty


def adjacent_differences(data: np.ndarray, valid: np.ndarray, max_pairs: int = 40_000_000) -> np.ndarray:
    """Differences between 4-adjacent valid pixels, as an array (bands, pairs).

    Pairs are thinned with a fixed stride when there are more than ``max_pairs``; no
    random numbers are involved, so the result is reproducible.
    """
    h = (data[:, :, 1:] - data[:, :, :-1])[:, valid[:, 1:] & valid[:, :-1]]
    v = (data[:, 1:, :] - data[:, :-1, :])[:, valid[1:, :] & valid[:-1, :]]
    d = np.concatenate([h, v], axis=1)
    if d.shape[1] > max_pairs:
        d = d[:, :: int(np.ceil(d.shape[1] / max_pairs))]
    return np.ascontiguousarray(d, dtype=np.float64)


def difference_covariance(d: np.ndarray, iters: int = 30, tol: float = 1e-4) -> np.ndarray:
    """Robust covariance of difference vectors ``d`` (bands, pairs), centred at zero.

    Shape from the half of the pairs with the smallest Mahalanobis distance, scale from
    the median distance of all pairs (which is set to the chi-square median). Pairs that
    straddle an object edge are the heavy tail and do not enter the shape.
    """
    nb, m = d.shape
    med = _chi2_median(nb)

    def distances(cov):
        y = np.linalg.solve(np.linalg.cholesky(cov), d)
        return np.einsum("ij,ij->j", y, y)

    s = d @ d.T / m
    s += np.eye(nb) * (1e-12 * max(np.trace(s) / nb, 1e-300))
    maha = distances(s)
    for _ in range(iters):
        q = np.median(maha)
        if not np.isfinite(q) or q <= 0:                 # more than half the pairs identical
            q = np.percentile(maha[maha > 0], 50) if (maha > 0).any() else 1.0
        core = d[:, maha <= q]
        new = core @ core.T / core.shape[1]
        new += np.eye(nb) * (1e-12 * max(np.trace(new) / nb, 1e-300))
        maha = distances(new)
        q = np.median(maha)
        if q > 0:                                        # scale: the median pair looks like unit noise
            new *= q / med
            maha *= med / q
        change = np.abs(new - s).max() / max(np.abs(s).max(), 1e-300)
        s = new
        if change < tol:
            break
    return s


def whiten(data: np.ndarray, valid: np.ndarray | None = None, kind: str = "full"):
    """Return ``(z, info)``: the stack in noise units and what was estimated.

    ``z`` has the shape of ``data`` (bands, rows, cols), float64. Adjacent valid pixels of
    ``z`` differ, in the median, like two draws of unit Gaussian noise. ``info`` holds the
    covariance ``S`` of the differences, the whitening matrix ``W`` (``z = W (x - centre)``)
    and the centre that was subtracted.

    ``kind``  "full"   the whole covariance: invariant to any invertible mix of the bands;
              "diag"   every band by its own scale: invariant to rescaling single bands;
              "scalar" one scale for all bands: invariant to a common rescaling only.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[None]
    if valid is None:
        valid = np.isfinite(data).all(axis=0)
    d = adjacent_differences(data, valid)
    if kind == "full":
        s = difference_covariance(d)
    elif kind == "diag":
        s = np.diag([difference_covariance(d[i:i + 1])[0, 0] for i in range(d.shape[0])])
    elif kind == "scalar":
        e = (d * d).sum(axis=0)
        s = np.eye(d.shape[0]) * (np.median(e) / _chi2_median(d.shape[0]))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    w = np.linalg.inv(np.linalg.cholesky(s))
    centre = np.array([b[valid].mean() for b in data])
    z = np.einsum("ij,jrc->irc", w, data - centre[:, None, None])
    z[:, ~valid] = 0.0
    return z, {"S": s, "W": w, "centre": centre}


def _trimmed_sd(d: np.ndarray, keep: float = 0.8) -> float:
    """Scale of differences ``d`` (centred at zero) from the ``keep`` share with the smallest magnitude.

    Finer than a median absolute deviation on integer data, and still blind to the tail of
    pairs that straddle an edge. 0.6617 is the same statistic for a unit Gaussian at 0.8.
    """
    a = np.abs(d)
    q = np.quantile(a, keep)
    core = a[a <= q]
    return float(np.sqrt(np.mean(core * core)) / 0.6617) if core.size else 0.0


def variability_curve(band: np.ndarray, valid: np.ndarray, bins: int = 24):
    """Typical difference between adjacent pixels of one band as a function of brightness.

    Returns ``(level, sd)``: bin medians of the pair mean and the trimmed sd of the pair
    difference in each bin.
    """
    hm = valid[:, 1:] & valid[:, :-1]
    vm = valid[1:, :] & valid[:-1, :]
    m = np.concatenate([((band[:, 1:] + band[:, :-1]) * 0.5)[hm], ((band[1:, :] + band[:-1, :]) * 0.5)[vm]])
    d = np.concatenate([(band[:, 1:] - band[:, :-1])[hm], (band[1:, :] - band[:-1, :])[vm]])
    order = np.argsort(m, kind="stable")
    m, d = m[order], d[order]
    edges = np.linspace(0, m.size, bins + 1).astype(np.int64)
    level = np.array([np.median(m[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    sd = np.array([_trimmed_sd(d[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return level, sd


def stabilise(data: np.ndarray, valid: np.ndarray | None = None, bins: int = 24, floor: float = 0.1):
    """Variance-stabilising transform of every band, estimated from the image itself.

    Pixel-to-pixel variability usually depends on brightness: shadows are smooth, sunlit
    texture is rough. A difference of five grey levels is then an event in a shadow and
    nothing on a sunlit crown. Each band is passed through f(x) = integral of dx / sd(x),
    with sd(x) the variability curve of that band, so that one unit means "one typical
    difference between neighbours at this brightness" everywhere. The curve is floored at
    ``floor`` times its median, which keeps flat or saturated ranges from being stretched
    without limit. Rescaling or shifting a band leaves its transform unchanged.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[None]
    if valid is None:
        valid = np.isfinite(data).all(axis=0)
    out = np.zeros_like(data)
    curves = []
    for i in range(data.shape[0]):
        level, sd = variability_curve(data[i], valid, bins)
        sd = np.maximum(sd, floor * np.median(sd[sd > 0]) if (sd > 0).any() else 1.0)
        keep = np.r_[True, np.diff(level) > 0]
        level, sd = level[keep], sd[keep]
        lo, hi = float(data[i][valid].min()), float(data[i][valid].max())
        grid = np.linspace(lo, hi, 2049)
        inv = 1.0 / np.interp(grid, level, sd)
        f = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(grid))])
        out[i] = np.interp(data[i], grid, f)
        curves.append((level, sd))
    out[:, ~valid] = 0.0
    return out, {"curves": curves}
