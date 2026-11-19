"""Adaptel-style growth from one seed, recorded instead of cut by a threshold.

An adaptel adds a pixel when the cell that discovers it has an accumulated distance below
the threshold T, and the accumulated distance of the new pixel is that of its discoverer
plus its distance to the running mean of the region. Growth from a given seed is therefore
the same sequence for every T, and the adaptel at threshold T is its prefix: the pixels
whose discoverer had an accumulated distance below T. ``growth`` returns that sequence, so
the whole family of regions of one seed can be examined at once.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _growth(z, valid, seed_r, seed_c, max_px, t_max):
    nb, rows, cols = z.shape
    cap = max_px * 4 + 16
    hd = np.empty(cap)                       # binary min-heap on accumulated distance
    hp = np.empty(cap, dtype=np.int64)
    hn = 0
    dist = np.full(rows * cols, -1.0)
    order = np.empty(max_px, dtype=np.int64)      # pixels in order of joining
    join = np.empty(max_px)                       # accumulated distance of the discoverer
    acc = np.empty(max_px)                        # accumulated distance of the pixel itself
    mean = np.zeros(nb)
    s = seed_r * cols + seed_c
    for i in range(nb):
        mean[i] = z[i, seed_r, seed_c]
    dist[s] = 0.0
    order[0] = s
    join[0] = 0.0
    acc[0] = 0.0
    k = 1
    hd[0] = 0.0
    hp[0] = s
    hn = 1
    dr = (0, 0, 1, -1)
    dc = (1, -1, 0, 0)
    while hn > 0 and k < max_px:
        d0 = hd[0]
        p = hp[0]
        hn -= 1
        if hn > 0:                                  # sift the last element down from the root
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
        if d0 >= t_max:
            break
        r = p // cols
        c0 = p % cols
        for q in range(4):
            rr = r + dr[q]
            cc = c0 + dc[q]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols or not valid[rr, cc]:
                continue
            n = rr * cols + cc
            if dist[n] >= 0.0:
                continue
            e = 0.0
            for i in range(nb):
                v = mean[i] / k - z[i, rr, cc]
                e += v * v
            d = d0 + np.sqrt(e)
            dist[n] = d
            for i in range(nb):
                mean[i] += z[i, rr, cc]
            order[k] = n
            join[k] = d0
            acc[k] = d
            k += 1
            i = hn                                  # push
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
            if k >= max_px or hn >= cap - 1:
                break
    return order[:k], join[:k], acc[:k]


def growth(z, valid, seed, max_px=20000, t_max=np.inf):
    """Growth sequence of the adaptel seeded at ``seed`` (row, col) on ``z`` (bands, rows, cols).

    Returns ``(order, join)``: flat pixel indices in order of joining, and for each the
    accumulated distance of the cell that discovered it. The adaptel at threshold T is
    ``order[join < T]`` and ``join`` is non-decreasing.
    """
    z = np.ascontiguousarray(z, dtype=np.float64)
    order, join, _ = _growth(z, np.ascontiguousarray(valid), int(seed[0]), int(seed[1]), int(max_px), float(t_max))
    return order, join
