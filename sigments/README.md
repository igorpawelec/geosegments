# sigments: sigma-regulated segments

Image segmentation with one dimensionless parameter. Research prototype.

**Author:** Igor Pawelec, University of Agriculture in Krakow (ORCID 0009-0005-9798-3596).
Started 2026-11-19. **Licence:** GPL-3.0-or-later, see `LICENSE`.

## The question

Adaptels need a threshold in band units and multiresolution segmentation needs a scale, a
shape and a compactness. Is there a segmenter with one dimensionless parameter whose
meaning does not depend on the size, the shape or the number of the objects?

The answer tried here: put the bands into units of the image's own pixel-to-pixel
variability, seed regions at the calmest pixels, and let a region be everything that stays
within `k` of those units from the colour of its seed. The segments are not superpixels:
they have no common size, no compactness and no target count. Whether they are best used as
whole objects or as primitives for a later classification is not decided yet.

## Install and run

Version 0.1.0 is tagged `sigments-v0.1.0`, as a pre-release: the tag pins the code for
software that uses the segmentation, and says nothing more about a research prototype
whose name is still a working name.

```bash
# from the repository, without a clone; numpy, numba and scipy have to be installed
pip install --no-deps "git+https://github.com/igorpawelec/geosegments.git#subdirectory=sigments"
# the same, pinned to a version
pip install --no-deps "git+https://github.com/igorpawelec/geosegments.git@sigments-v0.1.0#subdirectory=sigments"
# from a clone, in this folder
pip install -e .            # numpy, numba, scipy
pip install -e .[io]        # plus rasterio and geopandas for the command line
```

```bash
python -m sigments ortho.tif labels.tif --k 9
python -m sigments ortho.tif labels.tif --k 6 --bands 1 2 3 --polygons segments.gpkg
```

```python
from sigments import segment
labels, n = segment(data, k=9)                                   # data: (bands, rows, cols)
labels, n = segment(data, k=9, smooth=0.0, stabilised=False)     # the fully invariant variant
```

`k` is the tolerance: lower values keep finer differences apart, higher values merge more.
On the scenes tried so far the useful range is about 5 to 13. The label raster is int32
with -1 on invalid pixels.

Tests: `python -m unittest discover -s tests` (pytest works too). They check that known
regions are recovered over a wide range of `k` and that the result is identical after
mixing or rescaling the bands and after rotating or mirroring the raster, and they hold
the call and the result described above (`tests/test_contract.py`), which is what a
pinned version is pinned for.

## What is here

`sigments/`

| module | what it does |
|---|---|
| `noise.py` | `whiten`: puts a band stack into units of its own pixel-to-pixel variability (robust covariance of the differences between adjacent pixels, affine equivariant). `stabilise`: optional variance-stabilising transform per band, estimated from the image. |
| `tessellate.py` | the segmenter. Calm pixels seed regions; a region is every pixel reachable from the seed without stepping on one that differs from the seed colour by more than `k`; then every pixel goes to the seed it is closest to. One pass, no accumulated cost. |
| `__init__.py` | `segment(data, k)`: noise units, optional smoothing, tessellation. |
| `__main__.py` | the command line. |
| `flood.py` | membership growth from one seed, recorded for every tolerance (analysis tool). |
| `grow.py` | adaptel-style growth from one seed, recorded for every threshold (analysis tool). |
| `merge.py`, `basins.py` | the first attempt: region merging by statistical separability. Kept because the negative results below come from it. |

`bench/` holds the experiments, one script per question. The `crop_*` scripts are the path that was
walked on a development window, including the attempts that failed; they are kept as the record.
`source bench/env.sh` finds the interpreter; every script
then locates the repository, the sibling packages used as baselines (pygeoadaptels,
obiastudio) and the data through `bench/paths.py`. `results/` holds their output; the label
rasters and other large files are rebuilt by the scripts and are not in git.

Reference data are the two Krynki windows of the OBIA exercise (crowns from Pawelec et al.
2026) and the SNP_21 test raster of pygeoadaptels. They stay where they are and are not
copied here.

Pictures: `results/gallery_compare.png` (beside adaptels at the same number of segments),
`results/gallery_scales.png` (five values of `k` against five adaptel thresholds),
`results/gallery_snp21.png` (another site, RGB).

## What was found

Numbers are the mean IoU of a reference object with the segment that overlaps it most
(Table 4 of the paper). "Patch" = touching reference crowns joined into one object.
The table of both scenes is in `results/summary.md`.

1. **The adaptel threshold is a size.** Grown from the centre of a dead-crown patch, the
   area of an adaptel is a power law of the threshold with no plateau at the object edge,
   and the best threshold per patch correlates with patch size at r = 0.89 (log-log).
   In membership growth the best tolerance correlates with size at r = 0.03
   (`bench/crop_growth.py`, `bench/crop_flood.py`).
2. **Membership growth delineates better than accumulated growth** when the seed sits in
   the core of the object: per-patch optimum 0.78 against 0.68, best single setting 0.59
   against 0.54 (development window, 38 patches). From a random pixel of the object it
   drops to 0.32-0.40: the seed position is what matters (`bench/crop_rules.py`).
3. **Synthetic blobs, one contrast, eight sizes (8 to 2464 px), each method tuned on
   113 px** (`bench/synthetic_sizes.py`, `results/synthetic_sizes.png`):

   | size (px) | 8 | 19 | 49 | 113 | 255 | 617 | 1254 | 2464 |
   |---|---|---|---|---|---|---|---|---|
   | prototype without smoothing | 0.22 | 0.79 | 0.94 | 0.97 | 0.99 | 0.80 | 0.86 | 0.76 |
   | prototype, smoothing 0.7 px | 0.11 | 0.36 | 0.79 | 0.95 | 0.98 | 0.88 | 0.89 | 0.74 |
   | adaptels | 0.03 | 0.37 | 0.74 | 0.85 | 0.65 | 0.36 | 0.24 | 0.13 |
   | MRS, shape 0.1 | 0.01 | 0.60 | 0.83 | 0.88 | 0.93 | 0.85 | 0.82 | 0.64 |
   | MRS, shape 0.5 | 0.07 | 0.78 | 0.88 | 0.89 | 0.93 | 0.83 | 0.77 | 0.46 |

   The best setting per size spans 3-7 for the prototype without smoothing, 15-360 for
   adaptels and 8-90 for MRS. Smoothing trades the small objects for stability. Nothing
   catches the 8 px blobs at the tuned setting.
4. **Invariance** (`bench/robustness.py`; boundary F1 / adjusted Rand index against the
   unchanged input, every method with the setting that was best on the window):

   | change of the input | prototype, whitened + smoothed | prototype, default (stabilised) | adaptels | MRS 0.5/0.7 |
   |---|---|---|---|---|
   | raster rotated by 90 degrees | 1.00 / 1.00 | 0.98 / 0.93 | 0.73 / 0.51 | 0.93 / 0.67 |
   | all bands brighter by 30 % | 1.00 / 1.00 | 0.95 / 0.62 | 0.75 / 0.49 | 0.93 / 0.63 |
   | bands rescaled one by one | 1.00 / 0.97 | 0.99 / 0.98 | 0.74 / 0.03 | 0.84 / 0.07 |
   | bands mixed by a random matrix | 1.00 / 1.00 | 0.76 / 0.19 | 0.74 / 0.38 | 0.87 / 0.40 |
   | noise of one grey level | 0.81 / 0.33 | 0.79 / 0.29 | 0.73 / 0.52 | 0.90 / 0.55 |

   Without the stabilising transform the result does not depend on the units, the mixing
   or the orientation of the bands. The stabilising transform is per band: it keeps the
   invariance to rescaling and gives up the one to mixing.
5. **Real crowns** (`results/summary.md`). Krynki 1: prototype 0.484 on patches and 0.450
   on crowns, adaptels 0.476 / 0.478, MRS (0.5/0.7) 0.461 / 0.455. Krynki 2: prototype
   0.32 / 0.32, adaptels 0.449 / 0.456, MRS 0.413 / 0.417.
   On SNP_21 (spruce, RGB, a scene not used while designing; `bench/gallery.py`) `k = 9` puts each
   of the 36 digitised dead trees alone in a segment with 1,066 segments on the plot; adaptels at
   their default threshold need 3,060. There are no reference crowns there, only points.
6. **Why Krynki 2 fails** (`bench/heterogeneity.py`): there 74 % of the reference patches
   vary more inside than they differ from their surroundings (27 % in Krynki 1). A tolerance
   against one colour cannot hold such an object; a size or shape prior can.
7. **Stability under noise is the weak point** (last row of the table above). Boundaries
   move no more than those of adaptels, but the large regions do: membership is a hard
   threshold, and when the tolerance is close to the distance between two populations a
   region covers a random part of its neighbour and leaves holes that seed crumbs
   (`bench/debug_counts.py`). Smoothing by 0.7 px helps (boundary F1 0.71 without it).
8. **Speed** (`bench/timing.py`, 2400 x 2400 px, 4 bands): 15-23 s, of which 11-19 s is the
   estimation of the noise units and 4 s the segmentation; adaptels 4 s, MRS 17 s.
9. **What did not work** (development window): merging by Bhattacharyya distance between
   region Gaussians floods above k of about 3 (a broad region tolerates everything);
   merging on boundary contrast fails on blurred edges (0.19); merging on region means
   reaches 0.36-0.38; watershed basins as the starting regions do not help; picking the
   tolerance from the growth curve (first leak, a fraction of the flood tolerance) gains
   only 0.01-0.05 over a fixed k.

## Open

* Heterogeneous objects (Krynki 2). One size-free idea is already ruled out
  (`bench/edge_sharpness.py`): boundaries inside a crown are not softer than its outline.
  The share of the segment contrast that falls within one pixel step is higher inside
  (0.37-0.38) than on outlines (0.28-0.29), the reverse of the hypothesis, and the contrast
  between adjacent segments separates inside from outline at AUC 0.85 on Krynki 1 but only
  0.59 on Krynki 2. Such objects probably need a shape or size prior, or context rules.
* Objects without a calm core (below about 15 px after blur) are absorbed.
* Noise stability: test membership on an edge-preserving average instead of a Gaussian.
* The noise-unit estimate uses every adjacent pair; a symmetric subsample would cut most
  of the run time.
* Literature check before any claim of novelty: seeded region growing (Adams & Bischof
  1994), the image foresting transform with root-anchored cost (Falcão et al. 2004),
  statistical region merging (Nock & Nielsen 2004), MSER (Matas et al. 2002).
