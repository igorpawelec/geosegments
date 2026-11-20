"""Command line: segment a raster file and write the label raster.

    python -m sigments input.tif labels.tif --k 9
    python -m sigments input.tif labels.tif --k 6 --bands 1 2 3 --polygons segments.gpkg

The label raster is int32 with -1 as nodata and keeps the grid and the reference system of
the input. Needs rasterio; ``--polygons`` also needs geopandas.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import segment


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sigments", description="Sigma-regulated segments of a raster.")
    ap.add_argument("input", help="raster with one or more bands")
    ap.add_argument("output", help="label raster to write (GeoTIFF, int32, -1 = nodata)")
    ap.add_argument("--k", type=float, default=9.0, help="tolerance in units of the image's own pixel-to-pixel variability (default 9)")
    ap.add_argument("--bands", type=int, nargs="+", help="1-based band numbers to use (default: all)")
    ap.add_argument("--smooth", type=float, default=0.7, help="Gaussian smoothing in pixels after the noise units, 0 = none (default 0.7)")
    ap.add_argument("--no-stabilise", action="store_true", help="skip the per-band variance-stabilising transform")
    ap.add_argument("--polygons", help="also write the segments as polygons to this file")
    args = ap.parse_args(argv)

    import rasterio
    with rasterio.open(args.input) as src:
        data = src.read(args.bands).astype(np.float64) if args.bands else src.read().astype(np.float64)
        profile = src.profile
        valid = src.dataset_mask() > 0
        valid &= np.isfinite(data).all(axis=0)
    t = time.perf_counter()
    labels, n = segment(data, k=args.k, valid=valid, smooth=args.smooth, stabilised=not args.no_stabilise)
    print(f"{n} segments from {int(valid.sum())} pixels in {time.perf_counter() - t:.1f} s (k = {args.k:g})")
    profile.update(driver="GTiff", count=1, dtype="int32", nodata=-1, compress="deflate", predictor=2)
    for key in ("photometric", "jpeg_quality"):
        profile.pop(key, None)
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(labels.astype(np.int32), 1)
    if args.polygons:
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape
        recs = [(int(v), shape(g)) for g, v in shapes(labels.astype(np.int32), mask=labels >= 0, transform=profile["transform"])]
        gdf = gpd.GeoDataFrame({"segment": [r[0] for r in recs]}, geometry=[r[1] for r in recs], crs=profile.get("crs"))
        gdf = gdf.dissolve(by="segment", as_index=False)
        gdf["area"] = gdf.geometry.area
        gdf.to_file(args.polygons)
        print(f"{len(gdf)} polygons written to {args.polygons}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
