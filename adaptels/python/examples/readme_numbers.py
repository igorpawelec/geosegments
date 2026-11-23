# -*- coding: utf-8 -*-
"""
The numbers of the README's Reference section, on the GeoTIFF shipped with the repository.

    python examples/readme_numbers.py

Prints, in the order the README states them, the adaptel counts of the
threshold table (the top-left 200 x 200 px of test_data/SNP_21_2020_1.tif,
read as a plain array) and the counts of the Contiguity paragraph (the whole
400 x 400 px raster, as a plain array and through create_adaptels on the
file). Writes nothing. The figure captions come from www/figures.py and the
timings from examples/timing.py.
"""
import os

import numpy as np
import rasterio
from scipy import ndimage

from pygeoadaptels import adaptels_from_array, create_adaptels, enforce_connectivity

# package root, so the script works wherever it is launched from
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(HERE, "test_data", "SNP_21_2020_1.tif")
FOUR = ndimage.generate_binary_structure(2, 1)       # 4-connectivity, as the grower uses by default


def pieces(labels):
    """Number of 4-connected pieces of every adaptel id present."""
    out = []
    for lab, box in enumerate(ndimage.find_objects(labels + 1)):
        if box is not None:
            out.append(ndimage.label(labels[box] == lab, structure=FOUR)[1])
    return np.array(out)


with rasterio.open(INP) as src:
    data = src.read().astype(np.float64)             # plain array: the zeros outside the plot are data

# ---- threshold table: top-left 200 x 200 px ----------------------------------------
corner = data[:, :200, :200]
print("Reference table, top-left 200 x 200 px, plain array:")
for t in (10.0, 30.0, 60.0, 120.0):
    print(f"  minkowski threshold {t:g}: {adaptels_from_array(corner, threshold=t)[1]} adaptels")
for t in (0.002, 0.01, 0.03, 0.2):
    n = adaptels_from_array(corner, threshold=t, distance="angular")[1]
    print(f"  angular threshold {t:g}: {n} adaptels")

# ---- Contiguity: whole 400 x 400 px raster at threshold 60 --------------------------
labels, n = adaptels_from_array(data, threshold=60.0)
p = pieces(labels)
split = int((p > 1).sum())
print("Contiguity, 400 x 400 px, threshold 60, plain array:")
print(f"  {split} of {n} adaptels ({split / n:.1%}) in more than one piece, the worst in {p.max()}")

labels_f, n_f = create_adaptels(INP, None, threshold=60.0, quiet=True)
split_f = int((pieces(labels_f) > 1).sum())
n_fc = enforce_connectivity(labels_f)[1]
print("Contiguity, create_adaptels on the file (nodata masked):")
print(f"  {n_f} adaptels, {split_f} of them split, {n_fc} after enforce_connectivity")

labels_c, n_c = enforce_connectivity(labels)
split_c = int((pieces(labels_c) > 1).sum())
print("Contiguity code example, plain array:")
print(f"  adaptels_from_array: {n}, {split} split")
print(f"  enforce_connectivity: {n_c}, {split_c} split")
print(f"  the count rises by {n_c / n - 1:.1%}")
