"""area_m2 and perimeter are measured on each polygon's own geometry.

Until 0.10.5 ``area_m2`` was the pixel count of the whole label, written on
every piece the label was left in, and ``perimeter`` was the circumference of
a circle of that area. On the test orthophoto the area attribute summed to
8,580 m2 over 7,851 m2 of polygons, and the perimeter of a crown read
10.26 m where its outline was 16.50 m long. A field called perimeter that is
not the perimeter is worse than no field.

Most expectations below are exact by construction: the pixel is 0.25 m and
the origin a whole number, so every coordinate, area and perimeter is a
dyadic rational, representable in float64 whatever the order of the ring or
the GDAL version. That also means those tests cannot see whether the
shoelace sum is taken on coordinates moved to a local origin. The last test
is the one that can: 0.1 m pixels on an origin that is not representable.

Copyright (C) 2026 Igor Pawelec. Licence: GPLv3.
"""

import os

import numpy as np
import pytest

fiona = pytest.importorskip("fiona")
pytest.importorskip("rasterio")
from affine import Affine                                    # noqa: E402

from pygeoadaptels import vectorize as V                     # noqa: E402

CRS_WKT = (
    'PROJCS["ETRS89 / Poland CS92",GEOGCS["ETRS89",DATUM["ETRS89",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",19],'
    'PARAMETER["scale_factor",0.9993],'
    'PARAMETER["false_easting",500000],PARAMETER["false_northing",-5300000],'
    'UNIT["metre",1]]'
)
T = Affine(0.25, 0.0, 540123.0, 0.0, -0.25, 810456.0)
PX = 0.0625                      # m2 per pixel
NODATA = -9999

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "test_data", "SNP_21_2020_1.tif")


def _features(tmp_path, labels, connectivity=4):
    out = tmp_path / "polys.gpkg"
    n = V.vectorize_adaptels(np.asarray(labels, dtype=np.int32), T, CRS_WKT,
                             str(out), connectivity=connectivity, quiet=True)
    with fiona.open(str(out)) as src:
        feats = [(f["properties"]["adaptel_id"], f["properties"]["area_m2"],
                  f["properties"]["perimeter"]) for f in src]
    assert n == len(feats)
    return sorted(feats)


def test_a_square_block(tmp_path):
    lab = np.full((6, 6), NODATA)
    lab[1:5, 1:5] = 0                               # 4 x 4 px = 1 m x 1 m
    assert _features(tmp_path, lab) == [(0, 1.0, 4.0)]


def test_one_pixel(tmp_path):
    lab = np.full((3, 3), NODATA)
    lab[1, 1] = 7
    assert _features(tmp_path, lab) == [(7, PX, 1.0)]


def test_a_label_in_two_pieces_carries_the_area_of_each_piece(tmp_path):
    lab = np.full((6, 8), NODATA)
    lab[0:2, 0:2] = 0                               # 2 x 2 block, 4 px
    lab[4, 4:7] = 0                                 # 1 x 3 strip, 3 px
    feats = _features(tmp_path, lab)
    assert feats == [(0, 3 * PX, 2.0), (0, 4 * PX, 2.0)]
    # the old attribute was 7 px on both, 14 px for 7 px of polygons
    assert sum(a for _, a, _ in feats) == 7 * PX


def test_a_hole_leaves_the_area_and_joins_the_perimeter(tmp_path):
    lab = np.full((7, 7), NODATA)
    lab[1:6, 1:6] = 0                               # 5 x 5 block ...
    lab[3, 3] = 1                                   # ... with one pixel of another label inside
    feats = _features(tmp_path, lab)
    # ring: 24 px; outline 20 edges outside and 4 around the hole
    assert feats == [(0, 24 * PX, 5.0 + 1.0), (1, PX, 1.0)]


def test_the_helper_on_a_plain_polygon():
    square = {"type": "Polygon", "coordinates": [
        [(10.0, 10.0), (14.0, 10.0), (14.0, 13.0), (10.0, 13.0), (10.0, 10.0)],
        [(11.0, 11.0), (12.0, 11.0), (12.0, 12.0), (11.0, 12.0), (11.0, 11.0)]]}
    assert V.polygon_area_perimeter(square) == (12.0 - 1.0, 14.0 + 4.0)


def _plain_shoelace(ring):
    """The sum as it would be written without moving to a local origin."""
    return abs(sum(x0 * y1 - x1 * y0
                   for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]))) / 2.0


def test_the_area_does_not_carry_the_size_of_the_map_coordinates(tmp_path):
    """0.1 m pixels at an origin no float64 holds exactly.

    The products in the shoelace sum are of the order of 3e12 here and their
    differences of the order of 0.01, so taken on the map coordinates as they
    are the sum is off by more than the fourth decimal the attribute is
    written with. The test first shows that on these polygons, so that it
    cannot pass by the data being too kind, and then that the attribute is
    exact all the same.
    """
    transform = Affine(0.1, 0.0, 540123.37, 0.0, -0.1, 5810456.71)
    cell = 4                                            # 4 x 4 px = 0.16 m2, 1.6 m around
    n = 12
    lab = np.repeat(np.repeat(np.arange(n * n, dtype=np.int32).reshape(n, n), cell, axis=0),
                    cell, axis=1)
    out = tmp_path / "cells.gpkg"
    V.vectorize_adaptels(lab, transform, CRS_WKT, str(out), quiet=True)
    with fiona.open(str(out)) as src:
        feats = [(f["properties"]["area_m2"], f["properties"]["perimeter"],
                  [tuple(p[:2]) for p in f["geometry"]["coordinates"][0]]) for f in src]
    assert len(feats) == n * n
    worst = max(abs(_plain_shoelace(ring) - 0.16) for _, _, ring in feats)
    assert worst > 5e-5, f"the plain sum is exact here ({worst:.1e}); the test data proves nothing"
    assert {a for a, _, _ in feats} == {0.16}
    assert {p for _, p, _ in feats} == {1.6}


@pytest.mark.parametrize("connectivity", [4, 8])
def test_areas_add_up_to_the_labelled_area_on_the_test_orthophoto(tmp_path, connectivity):
    """Adaptels are left in several pieces often enough (about one in ten)
    for the old attribute to overshoot by 9 % here."""
    if not os.path.exists(DATA):
        pytest.skip("test_data/SNP_21_2020_1.tif is not in this checkout")
    import rasterio
    from pygeoadaptels import create_adaptels

    labels, n = create_adaptels(DATA, None, threshold=60.0, quiet=True)
    with rasterio.open(DATA) as src:
        transform, crs_wkt = src.transform, src.crs.to_wkt()
    out = tmp_path / "adaptels.gpkg"
    n_poly = V.vectorize_adaptels(labels, transform, crs_wkt, str(out),
                                  connectivity=connectivity, quiet=True)
    with fiona.open(str(out)) as src:
        areas = [f["properties"]["area_m2"] for f in src]
        ids = {f["properties"]["adaptel_id"] for f in src}
    assert len(ids) == n and n_poly >= n
    valid_px = int((labels >= 0).sum())
    assert abs(sum(areas) - valid_px * PX) < 1e-6 * n_poly
