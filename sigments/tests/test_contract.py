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


class WhatIsRefused(unittest.TestCase):
    """A ValueError with the reason in it, in place of a result that means nothing or of
    an IndexError from inside the estimate."""

    SETTINGS = [dict(smooth=sm, stabilised=st) for sm in (0.7, 0.0) for st in (True, False)]

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(6)
        cls.data = rng.normal(0, 1.0, size=(3, 40, 40))
        cls.data[:, :, 20:] += 30.0

    def refused(self, reason, data, **kw):
        with self.assertRaisesRegex(ValueError, "cannot estimate that unit here: .*" + reason):
            segment(data, k=9, **kw)

    def test_k_has_to_be_a_positive_number(self):
        for k in (0, 0.0, -1, -9.0, float("nan"), float("inf"), None, "nine", [5, 9]):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "k must be a positive number"):
                    segment(self.data, k=k)
        labels, n = segment(self.data, k=0.5)       # small is fine, it is only strict
        self.assertGreaterEqual(n, 2)

    def test_k_may_be_any_kind_of_number(self):
        from decimal import Decimal
        from fractions import Fraction
        ref, n_ref = segment(self.data, k=9)
        for k in (9.0, np.float32(9), np.int64(9), np.array(9.0), Fraction(9), Decimal("9")):
            with self.subTest(k=repr(k)):
                labels, n = segment(self.data, k=k)
                self.assertEqual(n, n_ref)
                self.assertTrue(np.array_equal(labels, ref))

    def test_an_image_with_no_unit_is_refused_whatever_the_settings(self):
        nobody = np.zeros((40, 40), dtype=bool)
        one = nobody.copy(); one[5, 5] = True
        apart = nobody.copy(); apart[5, 5] = apart[20, 30] = True
        cases = [
            ("no valid pixel", self.data, nobody, "no valid pixel"),
            ("all NaN", np.full((3, 40, 40), np.nan), None, "no valid pixel"),
            ("one valid pixel", self.data, one, "no two of its 1 valid pixels"),
            ("two pixels apart", self.data, apart, "no two of its 2 valid pixels"),
            ("a constant image", np.ones((3, 40, 40)), None, "do not differ anywhere"),
            ("a constant band", np.full((40, 40), 7.5), None, "do not differ anywhere"),
        ]
        for name, data, valid, reason in cases:
            for kw in self.SETTINGS:
                with self.subTest(case=name, **kw):
                    self.refused(reason, data, valid=valid, **kw)

    def test_too_few_neighbours_for_the_variability_curve(self):
        few = np.zeros((40, 40), dtype=bool)
        few[10, 10:14] = True                        # four pixels in a row: three pairs
        for smooth in (0.7, 0.0):
            with self.subTest(smooth=smooth):
                self.refused("3 pairs of neighbours are too few for the variability curve",
                             self.data, valid=few, smooth=smooth, stabilised=True)
        block = np.zeros((40, 40), dtype=bool)
        block[10:14, 18:22] = True                   # 16 pixels across the edge: 24 pairs, enough
        labels, n = segment(self.data, k=9, valid=block, smooth=0.0, stabilised=True)
        self.assertTrue((labels[~block] == -1).all())
        self.assertTrue(np.array_equal(np.unique(labels[block]), np.arange(n)))

    def test_a_failed_estimate_gives_its_reason(self):
        halves = np.zeros((20, 20))
        halves[:, 10:] = 5.0                         # 20 of 760 pairs differ
        for stabilised in (True, False):
            with self.subTest(case="two flat halves, smoothed", stabilised=stabilised):
                self.refused("more than half of its neighbouring pixels are identical",
                             halves, smooth=0.7, stabilised=stabilised)
        huge = np.random.default_rng(8).normal(size=(2, 20, 20)) * 1e307
        with self.subTest(case="values too large to square"):
            self.refused("", huge, smooth=0.7, stabilised=False)


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
