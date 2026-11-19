"""Membership growth: the region of a seed as a function of the tolerance.

The level of a pixel is the largest deviation from the reference colour on the best path
from the seed to it (a minimax path cost). The region at tolerance k is the connected set
of pixels with level <= k: everything that can be reached without ever stepping on a pixel
that differs from the reference by more than k. Unlike an accumulated cost this does not
grow with the length of the path, so the tolerance says nothing about the size of the
region: a small object and a large one of the same contrast fill at the same k.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _flood_levels(z, valid, seed_r, seed_c, ref, max_px, adapt):
    nb, rows, cols = z.shape
    cap = max_px * 4 + 16
    hd = np.empty(cap)                       # binary min-heap on the deviation from the reference
    hp = np.empty(cap, dtype=np.int64)
    seen = np.zeros(rows * cols, dtype=np.bool_)
    order = np.empty(max_px, dtype=np.int64)
    level = np.empty(max_px)
    mean = ref.copy()
    cnt = 1.0
    s = seed_r * cols + seed_c
    seen[s] = True
    hd[0] = 0.0
    hp[0] = s
    hn = 1
    k = 0
    cur = 0.0
    dr = (0, 0, 1, -1)
    dc = (1, -1, 0, 0)
    while hn > 0 and k < max_px:
        d0 = hd[0]
        p = hp[0]
        hn -= 1
        if hn > 0:
            ld = hd[hn]
            lp = hp[hn]
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
                i = c
            hd[i] = ld
            hp[i] = lp
        if d0 > cur:
            cur = d0
        order[k] = p
        level[k] = cur
        k += 1
        r = p // cols
        c0 = p % cols
        if adapt and k > 1:
            for i in range(nb):
                mean[i] += (z[i, r, c0] - mean[i]) / cnt
        cnt += 1.0
        for q in range(4):
            rr = r + dr[q]
            cc = c0 + dc[q]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols or not valid[rr, cc]:
                continue
            n = rr * cols + cc
            if seen[n]:
                continue
            seen[n] = True
            e = 0.0
            for i in range(nb):
                v = mean[i] - z[i, rr, cc]
                e += v * v
            d = np.sqrt(e)
            i = hn
            hn += 1
            while i > 0:
                par = (i - 1) // 2
                if hd[par] <= d:
                    break
                hd[i] = hd[par]
                hp[i] = hp[par]
                i = par
            hd[i] = d
            hp[i] = n
            if hn >= cap - 1:
                break
    return order[:k], level[:k]


def flood_levels(z, valid, seed, max_px=20000, window=1, adapt=False):
    """Pixels in order of joining and the tolerance each one needs.

    The region at tolerance k is ``order[level <= k]``. The reference colour is the median
    of a ``window`` x ``window`` block at the seed, or the running mean of the region when
    ``adapt`` is set. ``level`` is non-decreasing.
    """
    z = np.ascontiguousarray(z, dtype=np.float64)
    r, c = int(seed[0]), int(seed[1])
    h = window // 2
    blk = z[:, max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1].reshape(z.shape[0], -1)
    ok = valid[max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1].ravel()
    ref = np.median(blk[:, ok], axis=1)
    return _flood_levels(z, np.ascontiguousarray(valid), r, c, np.ascontiguousarray(ref), int(max_px), bool(adapt))
