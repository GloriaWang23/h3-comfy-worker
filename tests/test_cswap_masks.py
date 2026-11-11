"""Unit tests for the seed-track mask helpers (comfy_nodes/cswap_nodes/masks.py). Needs numpy.

Run: python tests/test_cswap_masks.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "comfy_nodes", "cswap_nodes"))
import masks as mk  # noqa: E402


class Seeds(unittest.TestCase):
    def test_person_points(self):
        pos, neg = mk.seed_points([0.4, 0.1, 0.6, 0.3], "person", (600, 800))
        self.assertEqual(pos[0], {"x": 300, "y": 160})                 # face centre
        self.assertEqual(pos[1], {"x": 300, "y": 480})                 # torso: 1.5 face heights below the chin
        self.assertEqual(neg, [])

    def test_head_points(self):
        pos, neg = mk.seed_points([0.4, 0.1, 0.6, 0.3], "head", (600, 800))
        self.assertEqual(pos, [{"x": 300, "y": 160}])
        self.assertEqual(neg, [{"x": 300, "y": 480}])                  # the torso is not the head

    def test_points_stay_inside_the_frame(self):
        pos, _ = mk.seed_points([0.4, 0.8, 0.6, 0.99], "person", (600, 800))
        self.assertTrue(all(0 <= p["x"] < 600 and 0 <= p["y"] < 800 for p in pos))


class Stitch(unittest.TestCase):
    def test_stitch_forward_and_backward(self):
        fwd = np.ones((3, 2, 2), np.float32) * 0.9      # frames 2,3,4
        bwd = np.ones((3, 2, 2), np.float32) * 0.4      # frames 2,1,0 (reversed time)
        bwd[2] = 0.7                                    # this is frame 0
        out = mk.stitch(fwd, bwd, seed=2, total=5)
        self.assertEqual(out.shape, (5, 2, 2))
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.7, places=5)
        self.assertAlmostEqual(float(out[1, 0, 0]), 0.4, places=5)
        self.assertAlmostEqual(float(out[2, 0, 0]), 0.9, places=5)     # the seed frame comes from the forward pass
        self.assertAlmostEqual(float(out[4, 0, 0]), 0.9, places=5)

    def test_stitch_seed_at_first_frame(self):
        out = mk.stitch(np.ones((4, 2, 2), np.float32), None, seed=0, total=4)
        self.assertEqual(out.shape, (4, 2, 2))


class Lost(unittest.TestCase):
    def test_open_frees_lost_frames(self):
        m = np.zeros((3, 4, 4), bool)
        m[0, :2, :2] = True
        out, lost = mk.fill_lost(m, "open")
        self.assertEqual(lost, [1, 2])
        self.assertTrue(out[1].all() and out[2].all())                 # no restriction where tracking lost the person
        self.assertEqual(int(out[0].sum()), 4)

    def test_closed_keeps_lost_frames_empty(self):
        m = np.zeros((2, 4, 4), bool)
        m[0, 0, 0] = True
        out, lost = mk.fill_lost(m, "closed")
        self.assertEqual(lost, [1])
        self.assertFalse(out[1].any())                                 # head not visible: nothing to edit


if __name__ == "__main__":
    unittest.main()
