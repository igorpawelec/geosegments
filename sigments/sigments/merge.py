"""Region merging by statistical separability, with one dimensionless threshold.

Every region carries a Gaussian summary of its pixels in noise units (see ``noise.py``):
count, mean and covariance. Two adjacent regions are merged while they are closer than
``k`` standard deviations, measured by the Bhattacharyya distance between their Gaussians

    D = 1/8 dm' Sm^-1 dm + 1/2 ln( |S| / sqrt(|Sa| |Sb|) ),      merge  <=>  D <= k^2 / 8

For two regions of equal covariance this is exactly "the means differ by less than k
standard deviations". The criterion is an effect size: it does not grow with the number of
pixels, so a small region that differs from its surroundings is never absorbed because it
is small, and a large one is never split because it is large.

Three corrections keep it meaningful for very small regions:

* ``floor``  every covariance gets ``floor * I`` added: the noise that is always there.
  In noise units a single-band pixel has variance 1/2 (adjacent differences have 1).
* ``nu``     a region of n pixels uses (n * its own covariance + nu * its partner's) / (n + nu):
  a handful of pixels cannot yet claim a texture of their own.
* ``Sm``     the mean term is measured against S + Sa/na + Sb/nb, the spread plus the
  uncertainty of the two means, so one deviating pixel is weaker evidence than fifty.

Merging is done in passes. In a pass every region finds its closest neighbour; pairs that
chose each other and are below the current level are merged, all at once. Mutually best
pairs are disjoint, so the result of a pass does not depend on any processing order. The
level climbs a fixed geometric ladder, which makes the output a hierarchy: the partition
at a higher k is a coarsening of the partition at a lower k.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

MODES = {"mean": 0, "diag": 1, "bhat": 2, "edge": 3, "edge_max": 4, "edge_and_mean": 5, "edge_or_mean": 6,
         "edge_ratio": 7, "edge_global": 8}
INF = 1.0e300


@njit(cache=True, nogil=True, inline="always")
def _chol_logdet(a, nb, low):
    """Cholesky of ``a`` into ``low`` (lower triangle); returns ln|a|."""
    ld = 0.0
    for i in range(nb):
        for j in range(i + 1):
            s = a[i, j]
            for q in range(j):
                s -= low[i, q] * low[j, q]
            if i == j:
                if s < 1e-300:
                    s = 1e-300
                low[i, i] = math.sqrt(s)
                ld += math.log(s)
            else:
                low[i, j] = s / low[j, j]
    return ld


@njit(cache=True, nogil=True)
def _pair_cost(a, b, n, sm, sq, nb, floor, nu, mode, ca, cb, sa, sb, s, low, dm, y, im, isum, m_e, s_e, zc):
    na = n[a]
    nbb = n[b]
    t = 0
    for i in range(nb):
        mai = sm[a, i] / na
        mbi = sm[b, i] / nbb
        dm[i] = mai - mbi
        for j in range(i, nb):
            va = sq[a, t] / na - mai * (sm[a, j] / na)
            vb = sq[b, t] / nbb - mbi * (sm[b, j] / nbb)
            if i == j:
                if va < 0.0:
                    va = 0.0
                if vb < 0.0:
                    vb = 0.0
            ca[i, j] = va
            ca[j, i] = va
            cb[i, j] = vb
            cb[j, i] = vb
            t += 1
    if mode == 0 or mode >= 3:                      # fixed unit covariance: distance of the means
        d2 = 0.0
        se = 1.0 + 2.0 * floor * (1.0 / na + 1.0 / nbb)
        for i in range(nb):
            d2 += dm[i] * dm[i]
        q_mean = d2 / se
        if mode == 0:
            return q_mean / 8.0
        # boundary: mean squared step across the common border, minus what texture alone gives
        c2 = s_e / m_e
        prior = 4.0
        ra = (isum[a] + prior * nb) / (im[a] + prior)
        rb = (isum[b] + prior * nb) / (im[b] + prior)
        rp = (isum[a] + isum[b] + prior * nb) / (im[a] + im[b] + prior)
        if mode == 4:
            rp = max(ra, rb)
        if mode == 8:
            rp = float(nb)
        if mode == 7:                               # ratio to the smoother side, never below the global level
            q = (c2 / max(float(nb), min(ra, rb)) - 1.0) * nb
            return max(q, 0.0) / 8.0
        q_edge = c2 - rp - zc * rp * math.sqrt(2.0 / (nb * m_e))
        if q_edge < 0.0:
            q_edge = 0.0
        if mode == 5:
            return max(q_edge, q_mean) / 8.0
        if mode == 6:
            return min(q_edge, q_mean) / 8.0
        return q_edge / 8.0
    wa = na / (na + nu)
    wb = nbb / (nbb + nu)
    for i in range(nb):
        for j in range(nb):
            if mode == 1 and i != j:
                sa[i, j] = 0.0
                sb[i, j] = 0.0
                s[i, j] = 0.0
                continue
            va = wa * ca[i, j] + (1.0 - wa) * cb[i, j]
            vb = wb * cb[i, j] + (1.0 - wb) * ca[i, j]
            if i == j:
                va += floor
                vb += floor
            sa[i, j] = va
            sb[i, j] = vb
            s[i, j] = 0.5 * (va + vb)
    ld_a = _chol_logdet(sa, nb, low)
    ld_b = _chol_logdet(sb, nb, low)
    ld_s = _chol_logdet(s, nb, low)
    shape_term = 0.5 * (ld_s - 0.5 * (ld_a + ld_b))
    # mean term against the spread plus the uncertainty of the two means
    for i in range(nb):
        for j in range(nb):
            s[i, j] += sa[i, j] / na + sb[i, j] / nbb
    _chol_logdet(s, nb, low)
    d2 = 0.0
    for i in range(nb):
        v = dm[i]
        for q in range(i):
            v -= low[i, q] * y[q]
        y[i] = v / low[i, i]
        d2 += y[i] * y[i]
    return d2 / 8.0 + shape_term


@njit(cache=True, nogil=True, parallel=True)
def _costs(idx, ea, eb, n, sm, sq, nb, floor, nu, mode, out, im, isum, em, es, zc):
    m = idx.shape[0]
    chunk = 4096
    for c in prange((m + chunk - 1) // chunk):
        ca = np.empty((nb, nb))
        cb = np.empty((nb, nb))
        sa = np.empty((nb, nb))
        sb = np.empty((nb, nb))
        s = np.empty((nb, nb))
        low = np.zeros((nb, nb))
        dm = np.empty(nb)
        y = np.empty(nb)
        for t in range(c * chunk, min(m, (c + 1) * chunk)):
            e = idx[t]
            out[e] = _pair_cost(ea[e], eb[e], n, sm, sq, nb, floor, nu, mode, ca, cb, sa, sb, s, low, dm, y, im, isum, em[e], es[e], zc)


@njit(cache=True, nogil=True)
def _mutual_pairs(ea, eb, cost, best, bestc, theta, pa, pb, pe):
    """Closest neighbour of every region, then the pairs that chose each other below theta.

    Ties go to the neighbour with the smaller id. ``best`` and ``bestc`` must be reset to
    -1 / INF for the regions that have edges; they are reset here on the way out.
    """
    for e in range(ea.shape[0]):
        a = ea[e]
        b = eb[e]
        c = cost[e]
        if c < bestc[a] or (c == bestc[a] and b < best[a]):
            bestc[a] = c
            best[a] = b
        if c < bestc[b] or (c == bestc[b] and a < best[b]):
            bestc[b] = c
            best[b] = a
    k = 0
    for e in range(ea.shape[0]):
        a = ea[e]
        b = eb[e]
        if best[a] == b and best[b] == a and cost[e] <= theta:
            pa[k] = a
            pb[k] = b
            pe[k] = e
            k += 1
    for e in range(ea.shape[0]):
        best[ea[e]] = -1
        best[eb[e]] = -1
        bestc[ea[e]] = INF
        bestc[eb[e]] = INF
    return k


@njit(cache=True, nogil=True)
def _roots(parent):
    out = np.empty(parent.shape[0], dtype=np.int32)
    for i in range(parent.shape[0]):
        r = i
        while parent[r] != r:
            r = parent[r]
        j = i
        while parent[j] != r:                       # path compression
            nxt = parent[j]
            parent[j] = r
            j = nxt
        out[i] = r
    return out


def _tri(nb):
    return [(i, j) for i in range(nb) for j in range(i, nb)]


def ladder(k_max: float, k_min: float = 0.5, ratio: float = 2 ** 0.25) -> np.ndarray:
    """The fixed geometric ladder of levels up to ``k_max`` (which is appended if it is not on it)."""
    levels = []
    k = k_min
    while k < k_max * (1 - 1e-9):
        levels.append(k)
        k *= ratio
    levels.append(float(k_max))
    return np.array(levels)


class Hierarchy:
    """Result of ``merge_hierarchy``: one label raster per requested level."""

    def __init__(self, shape, valid):
        self.shape = shape
        self.valid = valid
        self.levels: list[float] = []
        self.labels: dict[float, np.ndarray] = {}
        self.counts: dict[float, int] = {}
        self.passes: dict[float, int] = {}
        self.seconds = 0.0

    def at(self, k: float) -> np.ndarray:
        key = min(self.levels, key=lambda v: abs(v - k))
        return self.labels[key]


def merge_hierarchy(z, valid=None, levels=None, keep=None, mode="mean", floor=0.5, nu=None, zc=1.0,
                    init=None, max_passes=100000, verbose=False) -> Hierarchy:
    """Merge regions of ``z`` (bands, rows, cols; noise units) up a ladder of levels.

    ``levels``  increasing thresholds k; merging runs to convergence at each in turn.
    ``keep``    the levels whose label raster is stored (default: all of ``levels``).
    ``init``    label raster of the regions to start from (None = single pixels).
    Labels are int32, 0..n-1 in raster order of the first pixel of each region, -1 = invalid.
    """
    t0 = time.perf_counter()
    z = np.asarray(z, dtype=np.float64)
    nb, rows, cols = z.shape
    if valid is None:
        valid = np.isfinite(z).all(axis=0)
    if levels is None:
        levels = ladder(3.0)
    levels = [float(v) for v in levels]
    keep = set(levels) if keep is None else {min(levels, key=lambda v: abs(v - q)) for q in keep}
    nu = float(nb + 2) if nu is None else float(nu)
    code = MODES[mode]

    vf = valid.ravel()
    ids = np.full(rows * cols, -1, dtype=np.int32)
    if init is None:
        kk = int(vf.sum())
        ids[vf] = np.arange(kk, dtype=np.int32)
    else:
        _, first, inv = np.unique(np.asarray(init).ravel()[vf], return_index=True, return_inverse=True)
        rank = np.empty(first.shape[0], dtype=np.int32)          # number regions by their first pixel
        rank[np.argsort(first, kind="stable")] = np.arange(first.shape[0], dtype=np.int32)
        ids[vf] = rank[inv]
        kk = int(first.shape[0])
    ids2 = ids.reshape(rows, cols)
    pix = ids[vf]

    zf = z.reshape(nb, -1)[:, vf]                                # (nb, valid pixels)
    n = np.bincount(pix, minlength=kk).astype(np.float64)
    sm = np.empty((kk, nb))
    for i in range(nb):
        sm[:, i] = np.bincount(pix, weights=zf[i], minlength=kk)
    tri = _tri(nb)
    sq = np.empty((kk, len(tri)))
    for t, (i, j) in enumerate(tri):
        sq[:, t] = np.bincount(pix, weights=zf[i] * zf[j], minlength=kk)
    del zf

    hm = valid[:, 1:] & valid[:, :-1]
    vm = valid[1:, :] & valid[:-1, :]
    pa_ = np.concatenate([ids2[:, :-1][hm], ids2[:-1, :][vm]])
    pb_ = np.concatenate([ids2[:, 1:][hm], ids2[1:, :][vm]])
    w2 = np.concatenate([((z[:, :, 1:] - z[:, :, :-1]) ** 2).sum(axis=0)[hm],
                         ((z[:, 1:, :] - z[:, :-1, :]) ** 2).sum(axis=0)[vm]])
    same = pa_ == pb_
    im = np.bincount(pa_[same], minlength=kk).astype(np.float64)
    isum = np.bincount(pa_[same], weights=w2[same], minlength=kk)
    lo = np.minimum(pa_[~same], pb_[~same]).astype(np.int64)
    hi = np.maximum(pa_[~same], pb_[~same]).astype(np.int64)
    key, inv = np.unique(lo * kk + hi, return_inverse=True)
    ea = (key // kk).astype(np.int32)
    eb = (key % kk).astype(np.int32)
    em = np.bincount(inv, minlength=key.shape[0]).astype(np.float64)
    es = np.bincount(inv, weights=w2[~same], minlength=key.shape[0])
    del pa_, pb_, w2, same, lo, hi, key, inv

    cost = np.empty(ea.shape[0])
    _costs(np.arange(ea.shape[0], dtype=np.int64), ea, eb, n, sm, sq, nb, floor, nu, code, cost, im, isum, em, es, zc)

    parent = np.arange(kk, dtype=np.int32)
    best = np.full(kk, -1, dtype=np.int32)
    bestc = np.full(kk, INF)
    pa = np.empty(kk // 2 + 1, dtype=np.int32)
    pb = np.empty(kk // 2 + 1, dtype=np.int32)
    pe = np.empty(kk // 2 + 1, dtype=np.int64)
    changed = np.zeros(kk, dtype=np.bool_)
    out = Hierarchy((rows, cols), valid)
    alive = kk

    for level in levels:
        theta = level * level / 8.0
        passes = 0
        while passes < max_passes and ea.shape[0]:
            npairs = _mutual_pairs(ea, eb, cost, best, bestc, theta, pa, pb, pe)
            if npairs == 0:
                break
            passes += 1
            a = pa[:npairs]
            b = pb[:npairs]
            n[a] += n[b]
            sm[a] += sm[b]
            sq[a] += sq[b]
            e = pe[:npairs]
            im[a] += im[b] + em[e]
            isum[a] += isum[b] + es[e]
            parent[b] = a
            alive -= npairs
            changed[a] = True
            changed[b] = True
            touched = changed[ea] | changed[eb]
            changed[a] = False
            changed[b] = False
            ta = parent[ea[touched]]
            tb = parent[eb[touched]]
            inner = ta != tb
            lo = np.minimum(ta[inner], tb[inner]).astype(np.int64)
            hi = np.maximum(ta[inner], tb[inner]).astype(np.int64)
            key, inv = np.unique(lo * kk + hi, return_inverse=True)
            na_ = (key // kk).astype(np.int32)
            nb_ = (key % kk).astype(np.int32)
            m_new = np.bincount(inv, weights=em[touched][inner], minlength=key.shape[0])
            s_new = np.bincount(inv, weights=es[touched][inner], minlength=key.shape[0])
            # pairs that were interior to a merged region but not the joining edge (parallel edges cannot exist), none to add
            untouched = ~touched
            first_new = int(untouched.sum())
            ea = np.concatenate([ea[untouched], na_])
            eb = np.concatenate([eb[untouched], nb_])
            em = np.concatenate([em[untouched], m_new])
            es = np.concatenate([es[untouched], s_new])
            cost = np.concatenate([cost[untouched], np.empty(na_.shape[0])])
            _costs(np.arange(first_new, ea.shape[0], dtype=np.int64), ea, eb, n, sm, sq, nb, floor, nu, code, cost, im, isum, em, es, zc)
        if verbose:
            print(f"  k={level:6.3f}  passes {passes:4d}  regions {alive:9d}  edges {ea.shape[0]:9d}  "
                  f"{time.perf_counter() - t0:7.1f} s", flush=True)
        out.levels.append(level)
        out.counts[level] = alive
        out.passes[level] = passes
        if level in keep:
            root = _roots(parent)
            _, lab = np.unique(root, return_inverse=True)            # ids are raster-ordered already
            full = np.full(rows * cols, -1, dtype=np.int32)
            full[vf] = lab.astype(np.int32)[pix]
            out.labels[level] = full.reshape(rows, cols)
    out.seconds = time.perf_counter() - t0
    return out
