"""Feasibility of a size-free merge criterion for heterogeneous objects: edge sharpness.

Hypothesis: the boundary between two parts of one crown is a soft transition, the outline
of a crown is a sharp edge blurred only by the optics. For every pair of adjacent segments
of the prototype: the contrast of their mean colours, and how much of that contrast happens
within one pixel step across their common boundary. Pairs inside one reference patch are
compared with pairs across a patch outline (AUC: 0.5 = no information, 1 = separable).
"""
import sys
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
from sigments import noise_units
from data import load



def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    ranks = np.empty(allv.size)
    order = np.argsort(allv, kind="stable")
    ranks[order] = np.arange(1, allv.size + 1)
    return (ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


for scene, run in (("krynki1", "proto_st_9"), ("krynki2", "proto_st_11")):
    sc = load(scene)
    z, valid = noise_units(sc["data"], sc["valid"], smooth=0.7, stabilised=True)
    lab = np.load(OUT + f"labels_{scene}/{run}.npz")["labels"]
    patches, npatch = ndimage.label(sc["crowns"] > 0, structure=np.ones((3, 3)))
    nseg = int(lab.max()) + 1
    v = valid.ravel()
    lf = lab.ravel()
    n = np.bincount(lf[v], minlength=nseg).astype(float)
    mu = np.stack([np.bincount(lf[v], weights=z[i].ravel()[v], minlength=nseg) / np.maximum(n, 1) for i in range(z.shape[0])], axis=1)
    # which patch does a segment belong to: the patch holding most of its pixels, and that share
    inp = patches.ravel()[v]
    key = lf[v].astype(np.int64) * (npatch + 1) + inp
    uk, cnt = np.unique(key, return_counts=True)
    seg, pat = uk // (npatch + 1), uk % (npatch + 1)
    share_out = np.zeros(nseg)
    share_out[seg[pat == 0]] = cnt[pat == 0] / n[seg[pat == 0]]
    best_pat = np.zeros(nseg, dtype=np.int64)
    best_share = np.zeros(nseg)
    for s, p, c in zip(seg[pat > 0], pat[pat > 0], cnt[pat > 0]):
        if c / n[s] > best_share[s]:
            best_share[s], best_pat[s] = c / n[s], p
    inside = best_share >= 0.6
    outside = share_out >= 0.9
    # adjacent pixel pairs across segment boundaries
    hm = valid[:, 1:] & valid[:, :-1] & (lab[:, 1:] != lab[:, :-1])
    vm = valid[1:, :] & valid[:-1, :] & (lab[1:, :] != lab[:-1, :])
    a = np.concatenate([lab[:, :-1][hm], lab[:-1, :][vm]]).astype(np.int64)
    b = np.concatenate([lab[:, 1:][hm], lab[1:, :][vm]]).astype(np.int64)
    step = np.concatenate([np.sqrt(((z[:, :, 1:] - z[:, :, :-1]) ** 2).sum(0))[hm], np.sqrt(((z[:, 1:, :] - z[:, :-1, :]) ** 2).sum(0))[vm]])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    uk, inv, m = np.unique(lo * nseg + hi, return_inverse=True, return_counts=True)
    mean_step = np.bincount(inv, weights=step) / m
    pa, pb = uk // nseg, uk % nseg
    contrast = np.linalg.norm(mu[pa] - mu[pb], axis=1)
    sharp = mean_step / np.maximum(contrast, 1e-9)
    ok = (m >= 4) & (contrast > 1.0)
    internal = ok & inside[pa] & inside[pb] & (best_pat[pa] == best_pat[pb])
    outline = ok & ((inside[pa] & outside[pb]) | (inside[pb] & outside[pa]))
    print(f"{scene}: {int(internal.sum())} boundaries inside a patch, {int(outline.sum())} on a patch outline")
    for name, val in (("contrast of the two segments (noise units)", contrast), ("mean step across the boundary", mean_step), ("share of the contrast within one step", sharp)):
        print(f"   {name:44s} inside: median {np.median(val[internal]):5.2f} | outline: median {np.median(val[outline]):5.2f} | AUC outline over inside {auc(val[outline], val[internal]):.2f}")
    sys.stdout.flush()
