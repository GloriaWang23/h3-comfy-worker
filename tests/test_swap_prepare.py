"""Unit tests for swap_prepare.py (character-swap template preprocessing). Needs numpy + Pillow.

Run: python tests/test_swap_prepare.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import swap_prepare as sp  # noqa: E402


def solid(n, rgb, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    f = np.zeros((n, 48, 32, 3), np.int16) + np.array(rgb, np.int16)
    if noise:
        f += rng.integers(-noise, noise + 1, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


class Shots(unittest.TestCase):
    def test_detect_shots_single(self):
        self.assertEqual(sp.detect_shots(solid(60, (120, 80, 60), noise=6)), [(0, 60)])

    def test_detect_shots_cut(self):
        f = np.concatenate([solid(30, (200, 30, 30)), solid(30, (30, 30, 200))])
        self.assertEqual(sp.detect_shots(f), [(0, 30), (30, 60)])

    def test_detect_shots_merges_short(self):
        f = np.concatenate([solid(30, (200, 30, 30)), solid(5, (30, 30, 200)), solid(30, (200, 30, 30))])
        self.assertEqual(sp.detect_shots(f), [(0, 65)])


class Masks(unittest.TestCase):
    def test_best_frame(self):
        m = np.zeros((4, 10, 10), bool)
        m[1, :3, :3] = True
        m[2, :6, :6] = True
        self.assertEqual(sp.best_frame(m), 2)

    def test_crop_box_margin_and_clip(self):
        m = np.zeros((100, 80), bool)
        m[0:20, 0:10] = True  # top-left corner
        x0, y0, x1, y1 = sp.crop_box(m, margin=0.15)
        self.assertEqual((x0, y0), (0, 0))
        self.assertTrue(x1 <= 80 and y1 <= 100)
        self.assertEqual((x1 - x0) % 2, 0)
        self.assertEqual((y1 - y0) % 2, 0)
        self.assertGreaterEqual(x1, 10)
        self.assertGreaterEqual(y1, 20)

    def test_concat_masks_zero_fill(self):
        a, b = np.ones((10, 4, 6), bool), np.ones((3, 4, 6), bool)
        out = sp.concat_masks([a, 5, b], 18)
        self.assertEqual(out.shape, (18, 4, 6))
        self.assertFalse(out[10:15].any())
        self.assertTrue(out[:10].all() and out[15:].all())

    def test_concat_masks_length_mismatch(self):
        with self.assertRaises(ValueError) as e:
            sp.concat_masks([np.ones((10, 4, 6), bool), 5], 18)
        self.assertIn("!= source_frames", str(e.exception))


if __name__ == "__main__":
    unittest.main()
