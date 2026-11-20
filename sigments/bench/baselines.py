"""The segmenters the prototype is compared with, behind one call each.

Every function returns an int32 label raster with -1 on invalid pixels.
"""
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
import numpy as np
import pygeoadaptels as pa
from pygeoadaptels.sicle import sicle_from_array
from obiastudio.core import mrs


def adaptels(data, valid, threshold, connected=True):
    lab, _ = pa.adaptels_from_array(data, mask=~valid, threshold=float(threshold))
    if connected:
        lab, _ = pa.enforce_connectivity(lab)
    lab = lab.astype(np.int32)
    lab[~valid] = -1
    return lab


def multiresolution(data, valid, scale, shape=0.1, compactness=0.5):
    res = mrs.multiresolution_segmentation(data.astype(np.float32), valid, float(scale), shape=shape, compactness=compactness)
    lab = res.labels.astype(np.int32) - 1                 # objects 1..n, 0 = none
    lab[~valid] = -1
    return lab


def sicle(data, valid, n_segments, n_oversampling=None, n_iterations=5):
    n0 = int(n_oversampling or max(3000, 6 * n_segments))
    lab, _ = sicle_from_array(data, mask=(~valid).astype(np.uint8), n_segments=int(n_segments), n_oversampling=n0,
                              n_iterations=n_iterations, quiet=True)
    lab = lab.astype(np.int32)
    lab[~valid] = -1
    return lab
