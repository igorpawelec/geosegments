# -*- coding: utf-8 -*-
"""
How long adaptels take on this machine, on the GeoTIFF shipped with the repository.

    python examples/timing.py

Two sizes: the 400 x 400 px test orthophoto as it is, and the same image tiled
5 x 5 into 2000 x 2000 px. The first call compiles the Numba kernels and is
left out; the time printed is the best of the repeats, which is the number
least disturbed by whatever else the machine is doing. The README quotes the
output of this script.
"""
import os
import time

import numpy as np
import rasterio

from pygeoadaptels import adaptels_from_array

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with rasterio.open(os.path.join(HERE, "test_data", "SNP_21_2020_1.tif")) as src:
    small = src.read().astype(np.float64)
large = np.tile(small, (1, 5, 5))

adaptels_from_array(small[:, :50, :50], threshold=60.0)          # compile
for name, data, repeats in (("400 x 400 px", small, 10), ("2000 x 2000 px", large, 3)):
    best, n = float("inf"), 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, n = adaptels_from_array(data, threshold=60.0)
        best = min(best, time.perf_counter() - t0)
    print(f"{name}, {data.shape[0]} bands, threshold 60: {n} adaptels in {best:.2f} s "
          f"(best of {repeats})")
