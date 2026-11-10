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


class Graphs(unittest.TestCase):
    def test_prepare_graph_nodes(self):
        g = sp.build_prepare_graph("tpl.mp4", "480", 0.0)
        types_ = {n["class_type"] for n in g.values()}
        self.assertTrue({"LoadVideo", "H3RelayPrepareSource", "SaveVideo"} <= types_)
        prep = next(n for n in g.values() if n["class_type"] == "H3RelayPrepareSource")
        self.assertEqual(prep["inputs"]["resolution"], "480")
        self.assertEqual(prep["inputs"]["duration_seconds"], 0.0)
        self.assertEqual(prep["inputs"]["missing_audio"], "silence")

    def test_shot_graph_slice_and_one_save_per_object(self):
        g = sp.build_shot_graph("cs_prepared.mp4", 30, 90, "cs_s1", max_objects=3)
        sl = [n for n in g.values() if n["class_type"] == "ImageFromBatch"]
        self.assertIn({"batch_index": 30, "length": 90}, [{k: n["inputs"][k] for k in ("batch_index", "length")} for n in sl])
        saves = [n for n in g.values() if n["class_type"] == "H3RelaySaveMask"]
        self.assertEqual(sorted(n["inputs"]["filename_prefix"] for n in saves), ["cs_s1_o0", "cs_s1_o1", "cs_s1_o2"])
        self.assertTrue(all(n["inputs"]["frames"] == 90 for n in saves))
        ids = set(g)
        dangling = [v for n in g.values() for v in n["inputs"].values() if isinstance(v, list) and v[0] not in ids]
        self.assertEqual(dangling, [])

    def test_duration_param(self):
        self.assertEqual(sp.duration_param(9.9), 0.0)
        self.assertEqual(sp.duration_param(10.5), 0.0)
        self.assertEqual(sp.duration_param(14.0), 10.0)


class People(unittest.TestCase):
    def test_people_skip_empty_and_crop(self):
        frames = np.zeros((10, 40, 30, 3), np.uint8)
        m0 = np.zeros((10, 40, 30), bool); m0[4, 5:25, 5:15] = True
        out = sp.people_from_masks({0: m0, 1: np.zeros((10, 40, 30), bool)}, frames)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["obj"], 0)
        self.assertEqual(out[0]["frame"], 4)
        self.assertEqual(out[0]["crop"].shape[2], 3)
        self.assertAlmostEqual(out[0]["area"], 200 / 1200, places=3)

    def test_original_crop_scales_bbox_and_frame(self):
        orig = np.zeros((300, 200, 100, 3), np.uint8)
        orig[250, 20:40, 10:30] = 255  # original frame for idx24=200 at 30 fps
        c = sp.original_crop(orig, 30.0, 200, [5, 10, 15, 20], (50, 100))
        self.assertEqual(c.shape, (20, 20, 3))
        self.assertTrue((c == 255).all())


class MaskPlan(unittest.TestCase):
    def test_mask_plan_build(self):
        d = tempfile.mkdtemp()
        np.savez_compressed(os.path.join(d, "s0.npz"), mask=np.ones((10, 4, 6), bool))
        p = sp.build_mask_file({"name": "mask_p.npz", "total": 15, "parts": [{"media": "s0.npz"}, {"zeros": 5}]}, d)
        with np.load(p) as z:
            self.assertEqual(z["mask"].shape, (15, 4, 6))
            self.assertEqual(z["mask"].dtype, bool)

    def test_mask_plan_bad_total(self):
        d = tempfile.mkdtemp()
        np.savez_compressed(os.path.join(d, "s0.npz"), mask=np.ones((10, 4, 6), bool))
        with self.assertRaises(ValueError) as e:
            sp.build_mask_file({"name": "mask_p.npz", "total": 12, "parts": [{"media": "s0.npz"}]}, d)
        self.assertIn("!= source_frames", str(e.exception))


if __name__ == "__main__":
    unittest.main()
