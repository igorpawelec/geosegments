"""What software that pins a version of sigments relies on: the call, the result, the
dependencies of a plain install, and one version number in every place that states it.

Runs with the standard library (``python -m unittest discover -s tests``) and with pytest.
"""
import inspect
import os
import re
import subprocess
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sigments  # noqa: E402
from sigments import segment  # noqa: E402


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


class TheCall(unittest.TestCase):
    def test_the_signature_of_segment(self):
        params = inspect.signature(segment).parameters
        self.assertEqual(list(params), ["data", "k", "valid", "smooth", "stabilised", "kw"])
        self.assertEqual(params["k"].default, 9.0)
        self.assertIsNone(params["valid"].default)
        self.assertEqual(params["smooth"].default, 0.7)
        self.assertIs(params["stabilised"].default, True)
        self.assertEqual(params["kw"].kind, inspect.Parameter.VAR_KEYWORD)

    def test_the_result_is_a_label_raster_and_a_count(self):
        rng = np.random.default_rng(3)
        data = rng.normal(0, 2.0, size=(3, 60, 80))
        data[:, :, 40:] += 40.0                       # two regions
        valid = np.ones((60, 80), dtype=bool)
        valid[:10, :15] = False
        for dtype in (np.float64, np.float32):
            with self.subTest(dtype=dtype.__name__):
                labels, n = segment(data.astype(dtype), k=9, valid=valid, smooth=0.7, stabilised=True)
                self.assertIsInstance(labels, np.ndarray)
                self.assertEqual(labels.dtype, np.int32)
                self.assertEqual(labels.shape, (60, 80))
                self.assertIsInstance(n, int)
                self.assertTrue((labels[~valid] == -1).all())
                self.assertTrue(np.array_equal(np.unique(labels[valid]), np.arange(n)))
                self.assertGreaterEqual(n, 2)

    def test_one_band_may_come_without_the_band_axis(self):
        rng = np.random.default_rng(4)
        band = rng.normal(0, 1.0, size=(40, 50))
        band[:, 25:] += 30.0
        flat, n_flat = segment(band, k=9)
        stacked, n_stacked = segment(band[None], k=9)
        self.assertEqual(flat.shape, (40, 50))
        self.assertEqual(n_flat, n_stacked)
        self.assertTrue(np.array_equal(flat, stacked))

    def test_pixels_that_are_not_finite_are_invalid_with_or_without_a_mask(self):
        """A mask that is True on a NaN used to let the NaN into the noise estimate:
        this scene, one segment, came out as 451."""
        rng = np.random.default_rng(5)
        data = rng.normal(0, 1.0, size=(2, 30, 30))
        data[0, 5:8, 5:8] = np.nan
        everywhere = np.ones((30, 30), dtype=bool)
        careful = np.isfinite(data).all(axis=0)
        results = [segment(data, k=9), segment(data, k=9, valid=everywhere),
                   segment(data, k=9, valid=careful)]
        for labels, n in results:
            self.assertTrue((labels[5:8, 5:8] == -1).all())
            self.assertEqual(int((labels == -1).sum()), 9)
            self.assertEqual(n, results[0][1])
            self.assertTrue(np.array_equal(labels, results[0][0]))


class ThePackage(unittest.TestCase):
    def test_a_plain_install_needs_numpy_numba_and_scipy_only(self):
        deps = re.search(r"^dependencies = \[(.*?)\]", read("pyproject.toml"), re.M).group(1)
        self.assertEqual(sorted(re.findall(r'"([^"]+)"', deps)), ["numba", "numpy", "scipy"])
        code = ("import sys; sys.path.insert(0, %r); import sigments; "
                "bad = [m for m in ('rasterio', 'geopandas', 'shapely', 'fiona') if m in sys.modules]; "
                "sys.exit(', '.join(bad) or 0)" % ROOT)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, "importing sigments loaded: " + done.stderr.strip())

    def test_one_version_everywhere(self):
        manifest = re.search(r'^version = "([^"]+)"', read("pyproject.toml"), re.M).group(1)
        citation = re.search(r'^version: "([^"]+)"', read("CITATION.cff"), re.M).group(1)
        changelog = re.search(r"^## \[([^\]]+)\]", read("CHANGELOG.md"), re.M).group(1)
        self.assertEqual(sigments.__version__, manifest)
        self.assertEqual(citation, manifest)
        self.assertEqual(changelog, manifest)


if __name__ == "__main__":
    unittest.main()
