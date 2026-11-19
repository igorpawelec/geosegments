"""Clean timing of the pipeline on a full Krynki scene (run it when nothing else is running)."""
import time
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import noise_units, tessellate
from sigments.noise import stabilise, whiten, adjacent_differences, difference_covariance
from data import load
import baselines as bl

sc = load("krynki1")
data, valid = sc["data"], sc["valid"]
tessellate(np.zeros((1, 32, 32)), np.ones((32, 32), dtype=bool), 6.0)
t = time.perf_counter(); st = stabilise(data, valid)[0]; t_st = time.perf_counter() - t
t = time.perf_counter(); d = adjacent_differences(st, valid); t_d = time.perf_counter() - t
t = time.perf_counter(); s = difference_covariance(d); t_c = time.perf_counter() - t
print(f"stabilise {t_st:.1f} s | differences {t_d:.1f} s ({d.shape[1]} pairs) | robust covariance {t_c:.1f} s")
for name, kw in (("stabilised + smoothed", dict(smooth=0.7, stabilised=True)), ("whitened + smoothed", dict(smooth=0.7, stabilised=False)), ("whitened only", dict(smooth=0.0, stabilised=False))):
    t = time.perf_counter(); z, _ = noise_units(data, valid, **kw); t_u = time.perf_counter() - t
    t = time.perf_counter(); lab, n = tessellate(z, valid, k=9.0); t_s = time.perf_counter() - t
    print(f"{name:24s}: noise units {t_u:5.1f} s + segmentation {t_s:4.1f} s = {t_u + t_s:5.1f} s for {valid.sum() / 1e6:.2f} M pixels, {n} segments")
t = time.perf_counter(); bl.adaptels(data, valid, 200); print(f"adaptels T=200: {time.perf_counter() - t:.1f} s")
t = time.perf_counter(); bl.multiresolution(data, valid, 27, 0.5, 0.7); print(f"MRS 27 (0.5/0.7): {time.perf_counter() - t:.1f} s")
