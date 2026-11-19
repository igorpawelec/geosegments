"""Why does k not carry over from Krynki 1 to Krynki 2? The unit itself, measured on both scenes."""
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from data import load
from sigments.noise import _trimmed_sd

for scene in ("krynki1", "krynki2"):
    sc = load(scene)
    d, v = sc["data"], sc["valid"]
    print(scene)
    for i, name in enumerate(sc["bands"]):
        b = d[i]
        vals = b[v]
        spread = 1.4826 * np.median(np.abs(vals - np.median(vals)))
        lag = {}
        for k in (1, 2, 4, 8):
            dd = np.concatenate([(b[:, k:] - b[:, :-k])[v[:, k:] & v[:, :-k]], (b[k:, :] - b[:-k, :])[v[k:, :] & v[:-k, :]]])
            lag[k] = _trimmed_sd(dd[:: max(1, dd.size // 2_000_000)])
        print(f"  {name:6s} spread of values {spread:5.1f} | sd of differences at 1, 2, 4, 8 px: " + " ".join(f"{lag[k]:5.1f}" for k in (1, 2, 4, 8)) +
              f" | spread / lag-1 = {spread / lag[1]:.1f}, lag-4 / lag-1 = {lag[4] / lag[1]:.1f}")
