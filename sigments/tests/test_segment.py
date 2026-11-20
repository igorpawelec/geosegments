"""Fast checks of ``sigments.segment`` on a synthetic scene with known, well separated regions.

Rotating or mirroring the raster repeats the same arithmetic and must give the identical
partition. Mixing or rescaling the bands changes the arithmetic, so equality holds up to
rounding: on this scene the partitions are identical as well.

Runs with the standard library (``python -m unittest discover -s tests``) and with pytest.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sigments import segment  # noqa: E402

KW = dict(smooth=0.0, stabilised=False)          # the variant whose invariances are exact
BLOBS = [(40, 40, 12), (60, 180, 6), (190, 60, 3), (200, 200, 2)]


def same_partition(a, b):
    key = a.astype(np.int64) * (int(b.max()) + 1) + b
    return len(np.unique(key)) == len(np.unique(a)) == len(np.unique(b))


def purity(lab, truth):
    key = lab.astype(np.int64) * (int(truth.max()) + 1) + truth
    uk, cnt = np.unique(key, return_counts=True)
    best = np.zeros(int(lab.max()) + 1, dtype=np.int64)
    np.maximum.at(best, uk // (int(truth.max()) + 1), cnt)
    return best.sum() / lab.size


class SegmentOnKnownRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        h = w = 240
        truth = np.zeros((h, w), dtype=np.int32)
        truth[:, 120:] = 1
        truth[150:, :] += 2
        yy, xx = np.mgrid[0:h, 0:w]
        for i, (r, c, rad) in enumerate(BLOBS):
            truth[(yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2] = 4 + i
        means = np.array([[0, 0, 0], [40, -30, 20], [-35, 25, 45], [30, 50, -40]], dtype=float)[np.arange(truth.max() + 1) % 4]
        means[4:] = means[[0, 1, 2, 3]] + rng.choice([-1, 1], size=(len(BLOBS), 3)) * 35
        cls.truth = truth
        cls.data = means[truth].transpose(2, 0, 1) + rng.normal(0, 3.0, size=(3, h, w))
        cls.mix = rng.normal(size=(3, 3)) + 2 * np.eye(3)
        cls.ref, _ = segment(cls.data, k=6, **KW)

    def test_recovers_the_regions_over_a_wide_range_of_k(self):
        for k in (4, 6, 8, 10):
            with self.subTest(k=k):
                lab, n = segment(self.data, k=k, **KW)
                self.assertEqual(n, self.truth.max() + 1)
                self.assertGreater(purity(lab, self.truth), 0.999)
                for i in range(len(BLOBS)):          # every blob, down to 13 pixels, is one segment of its own
                    inside = np.unique(lab[self.truth == 4 + i])
                    self.assertEqual(inside.size, 1)
                    self.assertLessEqual((lab == inside[0]).sum(), 1.5 * (self.truth == 4 + i).sum() + 4)

    def test_labels_are_dense_and_invalid_pixels_are_minus_one(self):
        valid = np.ones(self.data.shape[1:], dtype=bool)
        valid[:20, :30] = False
        lab, n = segment(self.data, k=6, valid=valid, **KW)
        self.assertTrue((lab[~valid] == -1).all())
        self.assertTrue(np.array_equal(np.unique(lab[valid]), np.arange(n)))

    def test_same_result_when_bands_are_mixed_or_rescaled(self):
        mixed, _ = segment(np.einsum("ij,jrc->irc", self.mix, self.data) + 50.0, k=6, **KW)
        scaled, _ = segment(self.data * np.array([1.0, 257.0, 0.01])[:, None, None], k=6, **KW)
        self.assertTrue(same_partition(self.ref, mixed))
        self.assertTrue(same_partition(self.ref, scaled))

    def test_same_result_when_the_raster_is_rotated_or_mirrored(self):
        rot, _ = segment(np.ascontiguousarray(np.rot90(self.data, 1, axes=(1, 2))), k=6, **KW)
        flip, _ = segment(np.ascontiguousarray(self.data[:, :, ::-1]), k=6, **KW)
        self.assertTrue(same_partition(self.ref, np.ascontiguousarray(np.rot90(rot, -1))))
        self.assertTrue(same_partition(self.ref, np.ascontiguousarray(flip[:, ::-1])))

    def test_default_pipeline_runs_and_keeps_the_large_regions(self):
        lab, n = segment(self.data)                  # stabilised and smoothed, k = 9
        self.assertGreaterEqual(n, 4)
        self.assertGreater(purity(lab, self.truth), 0.98)


if __name__ == "__main__":
    unittest.main()
