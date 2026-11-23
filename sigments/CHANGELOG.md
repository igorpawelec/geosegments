# Changelog


## [0.1.0] — 2026-11-23

First tagged version, a pre-release. sigments is a research prototype: the method is not
published, and its name is a working name until the literature check listed in the README
is done. The tag exists so that software which uses the segmentation can pin the code it
was tested with.

### What it holds
- **`segment(data, k=9.0, valid=None, smooth=0.7, stabilised=True)`**, the segmentation.
  `data` is a `(bands, rows, cols)` array, or `(rows, cols)` for one band. The result is
  `(labels, n)`: `labels` an int32 raster numbered 0 to `n - 1`, with -1 on the pixels that
  `valid` excludes or that are not finite. `tests/test_contract.py` holds this call and this
  result.
- `noise_units`, `tessellate`, `whiten`, `stabilise`: the steps `segment` is made of.
- `hierarchy`, `ladder`, `merge_hierarchy`, `Hierarchy`, `basins`: the first attempt, region
  merging up a ladder of levels, kept for reference.
- `python -m sigments in.tif labels.tif --k 9`, the command line.

### Fixed on the way to the tag
- **A pixel that is not finite is invalid whatever the mask says.** With `valid` given,
  the mask was taken as it was, and a NaN under a True went into the noise estimate: a
  scene that is one segment came out as 451. `valid` is now combined with the finite
  pixels. Results on data without NaN are unchanged.

### Known limits
- `segment` needs something to measure the image's variability on. With no valid pixel,
  or too few to compare neighbours, it stops with an `IndexError`; on an image without any
  pixel-to-pixel variability, a constant one for instance, with
  `numpy.linalg.LinAlgError`. A caller that cuts a raster into tiles has to skip the
  empty and the uniform ones itself. What `segment` should return there is not decided.

### Dependencies
numpy, numba and scipy. The command line also needs rasterio, geopandas and shapely, which
`sigments[io]` adds; importing the package does not load them.
