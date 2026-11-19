"""Crown-level agreement between a segmentation and reference crowns.

Definitions of the segmentation-accuracy paper (Pawelec et al. 2026, Table 4): x is a
reference crown, y the segment with the greatest intersection with it.
"""
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
import numpy as np


def crown_metrics(labels, crowns, valid, min_px=4):
    """Per-crown arrays and their summary for one label raster.

    Crowns smaller than ``min_px`` pixels after rasterisation are left out.
    """
    lab = labels.astype(np.int64)
    nseg = int(lab.max()) + 1
    seg_size = np.bincount(lab[valid], minlength=nseg)
    inside = (crowns > 0) & valid
    cid = crowns[inside].astype(np.int64)
    key = cid * nseg + lab[inside]
    uk, cnt = np.unique(key, return_counts=True)
    c_of = uk // nseg
    s_of = uk % nseg
    ncr = int(crowns.max()) + 1
    x_size = np.bincount(cid, minlength=ncr)
    order = np.lexsort((-cnt, c_of))                       # per crown, largest intersection first
    first = np.r_[True, c_of[order][1:] != c_of[order][:-1]]
    c = c_of[order][first]
    inter = cnt[order][first].astype(np.float64)
    y = seg_size[s_of[order][first]].astype(np.float64)
    x = x_size[c].astype(np.float64)
    pieces = np.bincount(c_of, minlength=ncr)[c]            # segments touching the crown
    keep = x >= min_px
    c, inter, y, x, pieces = c[keep], inter[keep], y[keep], x[keep], pieces[keep]
    iou = inter / (x + y - inter)
    rec = inter / x
    prec = inter / y
    f1 = 2 * prec * rec / (prec + rec)
    out = {"crowns": int(keep.sum()), "segments": int((seg_size > 0).sum()),
           "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
           "iou50": float((iou >= 0.5).mean()), "f1_area": float(f1.mean()),
           "os": float((1 - rec).mean()), "us": float((1 - prec).mean()),
           "afi_median": float(np.median((x - y) / x)), "pieces_median": float(np.median(pieces)),
           "seg_px_median": float(np.median(seg_size[seg_size > 0])),
           "one_px_segments": int((seg_size == 1).sum())}
    return out, {"crown": c, "iou": iou, "x": x, "y": y}
