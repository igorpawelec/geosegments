"""The SICLE label raster declares the nodata value it holds.

``sicle_from_array`` marks pixels outside the mask with -1. ``create_sicle``
wrote them to a file that declared -9999, ``write_raster``'s default and the
adaptels convention, so no pixel of the file was nodata as far as a GIS could
tell: on the test orthophoto the 34,386 pixels outside the plot were drawn,
counted and polygonised as a superpixel numbered minus one.

Copyright (C) 2026 Igor Pawelec. Licence: GPLv3.
"""

import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "test_data", "SNP_21_2020_1.tif")


def _scene(tmp_path):
    """A small three-band raster with a nodata corner, written to disk."""
    from affine import Affine
    rng = np.random.default_rng(7)
    data = rng.integers(1, 255, size=(3, 40, 50)).astype(np.float32)
    data[:, :8, :10] = -9999.0
    path = tmp_path / "scene.tif"
    with rasterio.open(str(path), "w", driver="GTiff", height=40, width=50, count=3,
                       dtype="float32", nodata=-9999.0,
                       transform=Affine(0.25, 0, 540000.0, 0, -0.25, 810000.0)) as dst:
        dst.write(data)
    return str(path), 8 * 10


def test_the_file_declares_the_value_the_labels_carry(tmp_path):
    from pygeoadaptels.sicle import create_sicle
    src, n_nodata = _scene(tmp_path)
    out = tmp_path / "sicle.tif"
    labels, n = create_sicle(src, str(out), n_segments=20, n_oversampling=200, quiet=True)
    assert int((labels == -1).sum()) == n_nodata
    with rasterio.open(str(out)) as f:
        band = f.read(1)
        assert f.nodata == -1
        assert int((band == f.nodata).sum()) == n_nodata
        assert not (band == -9999).any()
        # read masked, the superpixels are exactly the valid pixels
        assert int(f.read_masks(1).astype(bool).sum()) == 40 * 50 - n_nodata
    assert np.array_equal(band, labels)
    assert set(np.unique(band[band >= 0])) == set(range(n))


def test_on_the_test_orthophoto(tmp_path):
    if not os.path.exists(DATA):
        pytest.skip("test_data/SNP_21_2020_1.tif is not in this checkout")
    from pygeoadaptels.sicle import create_sicle
    out = tmp_path / "sicle.tif"
    labels, n = create_sicle(DATA, str(out), n_segments=200, quiet=True)
    with rasterio.open(DATA) as f:
        outside = int((f.read_masks(1) == 0).sum())
    with rasterio.open(str(out)) as f:
        assert f.nodata == -1
        assert int((f.read_masks(1) == 0).sum()) == outside
