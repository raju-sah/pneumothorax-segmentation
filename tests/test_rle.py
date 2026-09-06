"""Unit tests for Run-Length Encoding (RLE) utilities."""

import unittest
import numpy as np
from src.utils.rle import rle_decode, rle_encode, aggregate_rle_masks


class TestRLE(unittest.TestCase):
    def test_rle_empty_mask(self):
        """Test sentinel value '-1' produces an empty mask."""
        mask = rle_decode("-1", shape=(100, 100))
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(np.sum(mask), 0)
        self.assertEqual(rle_encode(mask), "-1")

    def test_rle_roundtrip(self):
        """Test encoding and decoding a synthetic binary mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 20:40] = 1
        mask[50:60, 70:80] = 1

        encoded = rle_encode(mask)
        decoded = rle_decode(encoded, shape=(100, 100))

        np.testing.assert_array_equal(mask, decoded)

    def test_aggregate_rle_masks(self):
        """Test combining multiple RLE masks via bitwise logical OR."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10:20, 10:20] = 1

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[15:25, 15:25] = 1

        rle1 = rle_encode(mask1)
        rle2 = rle_encode(mask2)

        composite = aggregate_rle_masks([rle1, rle2, "-1"], shape=(100, 100))
        expected = np.bitwise_or(mask1, mask2)

        np.testing.assert_array_equal(composite, expected)


if __name__ == "__main__":
    unittest.main()
