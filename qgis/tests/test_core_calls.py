"""geoadaptels_palette/core.py against the test data of the repository, without a QGIS.

core.py is the layer that calls the packages, and it imports nothing from
qgis, so everything it does can be checked with pygeoadaptels, pygeopalette,
numba, rasterio and fiona installed. Skipped where they are not. What the
dialogs and the Processing runtime add is tested in a live QGIS.

The test data is found inside the repository: adaptels/python/test_data.
"""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("fiona")
pytest.importorskip("numba")
pytest.importorskip("pygeoadaptels")
pytest.importorskip("pygeopalette")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                # qgis/
REPO = os.path.dirname(ROOT)                                # the geosegments repository
for _p in (os.path.join(HERE, "qgis_stub"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geoadaptels_palette import core  # noqa: E402

DATA = os.path.join(REPO, "adaptels", "python", "test_data")
RGB = os.path.join(DATA, "SNP_21_2020_1.tif")
LAB = os.path.join(DATA, "SNP_21_2020_1_lab.tif")
SHP = os.path.join(DATA, "dead_trees_test.shp")
OUTSIDE = 34386                                             # pixels of the 400 x 400 raster outside the circular plot

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (RGB, LAB, SHP)),
    reason=f"the test data is expected in {DATA}")


def _nodata_and_masked(path):
    with rasterio.open(path) as src:
        band = src.read(1)
        return src.nodata, int((src.read_masks(1) == 0).sum()), band


@pytest.fixture(scope="module")
def adaptels(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("adaptels") / "adaptels.tif")
    return out, core.run_adaptels(RGB, out, threshold=60)


def test_list_spaces():
    spaces = core.list_spaces()
    assert len(spaces) == 15 and "lab" in spaces and "cam02" in spaces


def test_convert_colourspace_writes_the_one_file_asked_for(tmp_path):
    out = str(tmp_path / "sub" / "lab.tif")                  # the folder does not exist yet
    assert core.run_convert_colourspace(RGB, "lab", out) == out
    with rasterio.open(out) as src:
        assert src.count == 3 and src.dtypes[0] == "float32"
        assert src.descriptions == ("L", "a", "b")        # named by pygeopalette 0.8.0
        mine = src.read()
    with rasterio.open(LAB) as src:
        assert float(np.nanmax(np.abs(mine - src.read()))) < 1e-3


def test_adaptels(adaptels):
    out, n = adaptels
    assert n == 2792
    nodata, masked, band = _nodata_and_masked(out)
    assert nodata == -9999 and masked == OUTSIDE
    assert len(np.unique(band[band >= 0])) == n


def test_vectorize(adaptels, tmp_path):
    out, n = adaptels
    assert core.run_vectorize(out, str(tmp_path / "c8.gpkg"), connectivity=8) == 2874
    assert core.run_vectorize(out, str(tmp_path / "c4.gpkg"), connectivity=4) == 3060
    import fiona
    with fiona.open(str(tmp_path / "c4.gpkg")) as src:
        area = sum(f["properties"]["area_m2"] for f in src)
    assert abs(area - (400 * 400 - OUTSIDE) * 0.0625) < 1e-3


def test_enforce_connectivity_on_adaptels(adaptels, tmp_path):
    out, n = adaptels
    ec = str(tmp_path / "ec.tif")
    assert core.run_enforce_connectivity(out, ec) == 3060
    nodata, masked, _ = _nodata_and_masked(ec)
    assert nodata == -9999 and masked == OUTSIDE


def test_enforce_connectivity_keeps_the_nodata_value_of_sicle(tmp_path):
    """SICLE marks nodata with -1, and so does Grow seeds. The tool used to
    write -9999 as the nodata value of whatever it was given, so after it the
    34,386 pixels outside the plot were data again, numbered minus one."""
    si = str(tmp_path / "sicle.tif")
    n = core.run_sicle(RGB, si, n_segments=100)
    assert n == 100
    nodata, masked, _ = _nodata_and_masked(si)
    assert nodata == -1 and masked == OUTSIDE

    ec = str(tmp_path / "sicle_ec.tif")
    core.run_enforce_connectivity(si, ec)
    nodata, masked, band = _nodata_and_masked(ec)
    assert nodata == -1 and masked == OUTSIDE
    assert not (band == -9999).any()


def test_grow_seeds_recipe(tmp_path):
    labels, crowns = str(tmp_path / "g.tif"), str(tmp_path / "g.gpkg")
    core.run_grow_seeds(LAB, SHP, out_labels=labels, out_polys=crowns,
                        max_cost=15, band_weights=[0.5, 2.5, 1.0], max_radius=20,
                        fill_holes=True)
    nodata, masked, band = _nodata_and_masked(labels)
    assert nodata == -1
    assert int((band >= 0).sum()) == 6195                   # the anchor of the README
    assert set(np.unique(band[band >= 0])) == set(range(36))
    import fiona
    with fiona.open(crowns) as src:
        assert len(src) == 36

    ec = str(tmp_path / "g_ec.tif")
    core.run_enforce_connectivity(labels, ec)
    assert _nodata_and_masked(ec)[0] == -1


def test_a_band_the_raster_does_not_have_leaves_no_temp_file(tmp_path, monkeypatch):
    pytest.importorskip("osgeo.gdal")
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with core.band_subset(RGB, [1, 2, 9], 3) as src:
        assert src == RGB                                    # the VRT could not be built; the source is used
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".vrt"] == []
