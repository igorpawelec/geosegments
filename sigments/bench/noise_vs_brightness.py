"""Is the pixel-to-pixel variability constant? Robust sd of adjacent differences by brightness, per band."""
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from data import load

sc = load("krynki1")
data, valid = sc["data"], sc["valid"]
mad = lambda v: 1.4826 * np.median(np.abs(v))
print("robust sd of the difference between adjacent pixels, by the mean of the pair (deciles of brightness)")
for i, name in enumerate(sc["bands"]):
    b = data[i]
    m = np.concatenate([((b[:, 1:] + b[:, :-1]) / 2)[valid[:, 1:] & valid[:, :-1]], ((b[1:, :] + b[:-1, :]) / 2)[valid[1:, :] & valid[:-1, :]]])
    d = np.concatenate([(b[:, 1:] - b[:, :-1])[valid[:, 1:] & valid[:, :-1]], (b[1:, :] - b[:-1, :])[valid[1:, :] & valid[:-1, :]]])
    edges = np.percentile(m, np.arange(0, 101, 10))
    which = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, 9)
    sd = [mad(d[which == q]) for q in range(10)]
    mid = [np.median(m[which == q]) for q in range(10)]
    print(f"{name:6s} brightness " + " ".join(f"{v:6.0f}" for v in mid))
    print(f"{'':6s} sd         " + " ".join(f"{v:6.2f}" for v in sd) + f"   ratio brightest/darkest {sd[-1] / max(sd[0], 1e-9):.1f}")
