"""Segmentation by membership: calm pixels seed regions, a tolerance in noise units fills them.

1. Every pixel gets a "calmness": the gradient energy around it, in noise units. Calm pixels
   are the cores of things, pixels on a blurred edge are not.
2. Pixels are visited from the calmest on. A calm pixel that is still free becomes a seed,
   and its region is every free pixel that can be reached from it without stepping on one
   that differs from the seed colour by more than ``k``. Because the test is against the
   seed and not against the previous pixel, a region cannot creep along a gradient, and
   because nothing is accumulated along the way, ``k`` says nothing about the size of the
   region.
3. Pixels left over, mostly the mixed pixels of edges, go to the neighbouring region whose
   seed colour they are closest to (the smallest largest-deviation on the way).

The visiting order comes from the image, not from the raster, so the result does not depend
on where the scan starts.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .basins import gradient_energy


@njit(cache=True, nogil=True)
def _seed_regions(z, valid, order, calm, k2, window, labels, refs, where):
    """Step 2. Returns the number of regions; ``refs`` receives their seed colours."""
    nb, rows, cols = z.shape
    stack = np.empty(rows * cols, dtype=np.int64)
    ref = np.empty(nb)
    buf = np.empty((nb, 9))
    nlab = 0
    for t in range(order.shape[0]):
        s = order[t]
        if not calm[s]:
            continue
        if labels[s] != -1:
            continue
        r = s // cols
        c = s % cols
        if window == 1:
            for i in range(nb):
                ref[i] = z[i, r, c]
        else:                                           # mean of the free pixels of the 3 x 3 block
            m = 0
            for rr in range(max(r - 1, 0), min(r + 2, rows)):
                for cc in range(max(c - 1, 0), min(c + 2, cols)):
                    if valid[rr, cc] and labels[rr * cols + cc] == -1:
                        for i in range(nb):
                            buf[i, m] = z[i, rr, cc]
                        m += 1
            for i in range(nb):
                ref[i] = np.mean(buf[i, :m])
        for i in range(nb):
            refs[nlab, i] = ref[i]
        where[nlab] = s
        labels[s] = nlab
        stack[0] = s
        top = 1
        while top > 0:
            top -= 1
            p = stack[top]
            pr = p // cols
            pc = p % cols
            for q in range(4):
                rr = pr + (1 if q == 0 else (-1 if q == 1 else 0))
                cc = pc + (1 if q == 2 else (-1 if q == 3 else 0))
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols or not valid[rr, cc]:
                    continue
                n = rr * cols + cc
                if labels[n] != -1:
                    continue
                e = 0.0
                for i in range(nb):
                    v = z[i, rr, cc] - ref[i]
                    e += v * v
                if e <= k2:
                    labels[n] = nlab
                    stack[top] = n
                    top += 1
        nlab += 1
    return nlab


@njit(cache=True, nogil=True)
def _assign_rest(z, valid, labels, refs, cap):
    """Step 3. Free pixels join the adjacent region with the smallest minimax deviation from its seed colour.

    Returns the number of pushes that did not fit the heap of ``cap`` entries (0 = complete).
    A pixel is pushed at most once per neighbour, so 4 * size + 16 entries always suffice.
    """
    nb, rows, cols = z.shape
    size = rows * cols
    cost = np.full(size, np.inf)
    hd = np.empty(cap)
    hp = np.empty(cap, dtype=np.int64)
    hl = np.empty(cap, dtype=np.int32)
    hn = 0
    dropped = 0
    for p in range(size):
        if labels[p] < 0:
            continue
        pr = p // cols
        pc = p % cols
        lab = labels[p]
        for q in range(4):
            rr = pr + (1 if q == 0 else (-1 if q == 1 else 0))
            cc = pc + (1 if q == 2 else (-1 if q == 3 else 0))
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols or not valid[rr, cc]:
                continue
            n = rr * cols + cc
            if labels[n] != -1:
                continue
            e = 0.0
            for i in range(nb):
                v = z[i, rr, cc] - refs[lab, i]
                e += v * v
            if e < cost[n]:
                if hn >= cap:
                    dropped += 1
                    continue
                cost[n] = e
                i = hn
                hn += 1
                while i > 0:
                    par = (i - 1) // 2
                    if hd[par] <= e:
                        break
                    hd[i] = hd[par]
                    hp[i] = hp[par]
                    hl[i] = hl[par]
                    i = par
                hd[i] = e
                hp[i] = n
                hl[i] = lab
    while hn > 0:
        d0 = hd[0]
        p = hp[0]
        lab = hl[0]
        hn -= 1
        if hn > 0:
            ld = hd[hn]
            lp = hp[hn]
            ll = hl[hn]
            i = 0
            while True:
                c = 2 * i + 1
                if c >= hn:
                    break
                if c + 1 < hn and hd[c + 1] < hd[c]:
                    c += 1
                if hd[c] >= ld:
                    break
                hd[i] = hd[c]
                hp[i] = hp[c]
                hl[i] = hl[c]
                i = c
            hd[i] = ld
            hp[i] = lp
            hl[i] = ll
        if labels[p] != -1:
            continue
        labels[p] = lab
        pr = p // cols
        pc = p % cols
        for q in range(4):
            rr = pr + (1 if q == 0 else (-1 if q == 1 else 0))
            cc = pc + (1 if q == 2 else (-1 if q == 3 else 0))
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols or not valid[rr, cc]:
                continue
            n = rr * cols + cc
            if labels[n] != -1:
                continue
            e = 0.0
            for i in range(nb):
                v = z[i, rr, cc] - refs[lab, i]
                e += v * v
            if e < d0:
                e = d0
            if e < cost[n]:
                if hn >= cap:
                    dropped += 1
                    continue
                cost[n] = e
                i = hn
                hn += 1
                while i > 0:
                    par = (i - 1) // 2
                    if hd[par] <= e:
                        break
                    hd[i] = hd[par]
                    hp[i] = hp[par]
                    hl[i] = hl[par]
                    i = par
                hd[i] = e
                hp[i] = n
                hl[i] = lab
    return dropped


def tessellate(z, valid, k=8.0, calm_share=0.5, window=3, compete=True, return_stages=False):
    """Segment ``z`` (bands, rows, cols; noise units). Returns ``(labels, n)``, -1 = invalid.

    ``k``           tolerance: how far, in noise units, a pixel may be from the seed colour.
    ``calm_share``  share of the pixels, the calmest, that are allowed to seed a region.
    ``window``      1: the seed colour is the seed pixel; 3: the mean of its 3 x 3 block (a mean, not a
                    band-wise median, so that mixing the bands does not change it).
    ``compete``     True: once the seeds are known, every pixel goes to the seed it is closest to;
                    False: a pixel stays with the first seed that reached it within ``k``.
    """
    z = np.ascontiguousarray(z, dtype=np.float64)
    nb, rows, cols = z.shape
    valid = np.ascontiguousarray(valid, dtype=np.bool_)
    g = gradient_energy(z, valid).ravel()
    vf = valid.ravel()
    order = np.argsort(np.where(vf, g, np.inf), kind="stable")[: int(vf.sum())]
    calm = np.zeros(rows * cols, dtype=np.bool_)
    calm[order[: max(1, int(round(calm_share * order.size)))]] = True
    labels = np.full(rows * cols, -1, dtype=np.int32)
    labels[~vf] = -2
    refs = np.empty((int(calm.sum()), nb))
    where = np.empty(int(calm.sum()), dtype=np.int64)
    n = _seed_regions(z, valid, order, calm, float(k) ** 2, int(window), labels, refs, where)
    seeded = labels.copy() if return_stages else None
    if compete:
        labels[labels >= 0] = -1
        labels[where[:n]] = np.arange(n, dtype=np.int32)
    start = labels.copy()
    size = rows * cols
    if _assign_rest(z, valid, labels, refs[:n], size + 16) > 0:      # rare: retry with the heap that always fits
        labels[:] = start
        if _assign_rest(z, valid, labels, refs[:n], 4 * size + 16) > 0:
            raise RuntimeError("the competition heap overflowed, which should be impossible")
    del start
    free = (labels == -1).reshape(rows, cols)                 # valid islands without a calm pixel
    if free.any():
        from scipy import ndimage
        extra, m = ndimage.label(free)
        labels[free.ravel()] = (extra[free] - 1 + n).astype(np.int32)
        n += int(m)
    labels[labels == -2] = -1
    out = labels.reshape(rows, cols)
    if return_stages:
        seeded[seeded == -2] = -1
        return out, n, seeded.reshape(rows, cols)
    return out, n
