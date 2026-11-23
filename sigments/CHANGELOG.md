# Changelog


## [0.1.2] — 2026-11-23

A pre-release, as 0.1.0 is.

### Changed
- **The package metadata name the repository.** `pyproject.toml` gives its homepage, its
  source and its issue tracker, https://github.com/igorpawelec/geosegments, and a built
  package carries the three as `Project-URL` entries. Nothing else changed: every image is
  segmented exactly as in 0.1.1.


## [0.1.1] — 2026-11-23

A pre-release, as 0.1.0 is.

### Changed
- **`segment` refuses a `k` that is not a positive number**, with a `ValueError`. Zero and
  negative values were accepted and gave a result that meant nothing.
- **An image with no unit to measure in is refused, with a `ValueError` that says why.**
  `segment` and `noise_units` refuse, whatever the settings: an image with no valid pixel;
  one in which no two valid pixels are neighbours; one in which neighbouring pixels do not
  differ anywhere, a constant image for one; and, with `stabilised`, one with fewer pairs
  of neighbours than the variability curve has bins (24). An estimate that fails further
  on, or does not come out finite, is reported the same way; that is what happens with
  smoothing on when more than half of the neighbouring pixels are identical.
  In 0.1.0 these ended differently by setting. With `stabilised` most stopped with an
  `IndexError` or a `numpy.linalg.LinAlgError` from inside the estimate. Without it an
  image with no valid pixel came back as all -1 with `n = 0`, isolated pixels as one
  segment each, and a constant image without smoothing as one segment.

Every other image is segmented exactly as in 0.1.0.

### Known limits
- An image in which most neighbouring pixels are identical, large flat areas for
  instance, is refused when smoothing is on. With `smooth=0` it is segmented, in a unit
  taken from the few pairs that do differ, and two flat halves with a clear edge between
  them come out as one segment. Mask the flat areas out.


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
