# sigments – notes for development sessions

* What it is: a research prototype of an image segmentation with one dimensionless tolerance `k`
  ("sigma-regulated segments"). Findings, dead ends and open questions are in `README.md`; read its
  "What was found" and "Open" sections before proposing a change to the algorithm.
* Environment: a Python 3.12 environment with numpy, numba and scipy; the benchmarks also need rasterio,
  geopandas, scikit-image and matplotlib. `source bench/env.sh` finds the interpreter and sets `$PY`
  (on Windows a conda interpreter imports numpy only with the environment's `Library/bin` on PATH).
  Per-computer paths go in `CLAUDE.local.md`.
* Tests: `"$PY" -m unittest discover -s tests`; must pass before a commit. After a change to
  `sigments/noise.py` or `sigments/tessellate.py` also run `"$PY" bench/regression.py` from `bench/`
  where the saved label rasters exist: it must report identical rasters, or the change in results is
  deliberate and `results/summary.md` is rebuilt (`bench/final_eval.py`, then `bench/summary.py`).
* Layout: this folder is `sigments/` of the `geosegments` repository, next to `adaptels/` and `qgis/`;
  commands below are run from here. `sigments/` inside it is the package (pure computation; rasterio
  only in `__main__.py`); `bench/` the experiments, one script per question, all paths through
  `bench/paths.py`; `results/` their output; `tests/` the fast checks, which `sigments-tests.yml` at the
  repository root also runs.
* Baselines are never copied here: `pygeoadaptels` (adaptels, SICLE) comes from `adaptels/python` of
  this repository and `obiastudio` (`core/mrs.py`, multiresolution segmentation) from its own checkout
  beside the repository; `bench/paths.py` finds both.
* Data stay out of the repository: the Krynki exercise (orthophotos and reference crowns, in the personal
  OneDrive under `5_DYDAKTYKA/_B4/B4_OBIA/ZADANIE_KRYNKI`) and the test raster of pygeoadaptels. Label
  rasters in `results/labels_*` and `results/*.npz` are large and rebuilt by the scripts; they are
  gitignored.
* Numbers quoted in `README.md` come from a named script in `bench/`; a number without its script does
  not go in. Report negative results as plainly as positive ones.
* sigments has no licence yet (`LICENSE` in this folder; the adaptels packages and the plugin are
  GPL-3.0-or-later). The `geosegments` repository is private for that reason: do not make it public or
  publish the package without Igor's decision and a licence chosen for this folder. The name of the
  method is a working name until the literature check listed in `README.md` is done.
* Code, comments, commit messages and docs in English.
* Authorship and agent scopes: `.claude/rules/` at the repository root.
