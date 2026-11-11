"""Unit tests for swap_faces.py (people of a template from face identity). Needs numpy.

Run: python tests/test_swap_faces.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import swap_faces as sf  # noqa: E402


def emb(seed, jitter=0.0, k=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512)
    if jitter:
        v = v + np.random.default_rng(1000 + k).normal(size=512) * jitter
    return (v / np.linalg.norm(v)).astype(np.float32)


def det(frame, seed, sex, bbox=(100, 100, 200, 220), score=0.9, k=0):
    return {"frame": frame, "bbox": list(bbox), "score": score, "emb": emb(seed, 0.25, k), "sex": sex, "age": 28}


class Cluster(unittest.TestCase):
    def test_two_identities(self):
        dets = [det(i * 3, 1, "F", k=i) for i in range(20)] + [det(60 + i * 3, 2, "M", k=50 + i) for i in range(6)]
        people = sf.cluster_faces(dets)
        self.assertEqual(len(people), 2)
        self.assertEqual([len(p) for p in people], [20, 6])

    def test_noise_cluster_dropped(self):
        # one stray detection (a poster, a false positive) is not a person
        dets = [det(i * 3, 1, "F", k=i) for i in range(20)] + [det(30, 9, "M", score=0.62, k=99)]
        self.assertEqual(len(sf.cluster_faces(dets)), 1)

    def test_short_clip_keeps_small_clusters(self):
        # with few detections overall a 2-detection person still counts
        dets = [det(0, 1, "F", k=1), det(3, 1, "F", k=2), det(6, 2, "M", k=3), det(9, 2, "M", k=4)]
        self.assertEqual(len(sf.cluster_faces(dets)), 2)


class Person(unittest.TestCase):
    def test_gender_majority_and_confidence(self):
        c = [det(i, 1, "F", k=i) for i in range(8)] + [det(20 + i, 1, "M", k=20 + i) for i in range(2)]
        p = sf.person_card(c, (608, 800))
        self.assertEqual(p["gender"], "female")
        self.assertAlmostEqual(p["gender_conf"], 0.8)
        self.assertEqual((p["first_frame"], p["last_frame"], p["faces"]), (0, 21, 10))

    def test_best_frame_is_largest_clear_face(self):
        c = [det(0, 1, "F", bbox=(100, 100, 150, 160), k=1), det(30, 1, "F", bbox=(100, 100, 300, 340), k=2),
             det(60, 1, "F", bbox=(100, 100, 320, 360), score=0.61, k=3)]
        p = sf.person_card(c, (608, 800))
        self.assertEqual(p["frame"], 30)

    def test_best_frame_prefers_the_whole_person_in_frame(self):
        # same face size and score: at the bottom edge the upper body is cut off, mid-frame it is whole
        c = [det(10, 1, "F", bbox=(250, 640, 350, 760), k=1), det(40, 1, "F", bbox=(250, 200, 350, 320), k=2)]
        self.assertEqual(sf.person_card(c, (608, 800))["frame"], 40)

    def test_crop_is_head_and_upper_body_inside_frame(self):
        p = sf.person_card([det(0, 1, "F", bbox=(500, 20, 600, 140), k=1)], (608, 800))
        x0, y0, x1, y1 = p["crop_box"]
        self.assertTrue(0 <= x0 < x1 <= 608 and 0 <= y0 < y1 <= 800)
        self.assertGreater(y1 - y0, 2 * (140 - 20))      # below the chin: upper body
        self.assertEqual(((x1 - x0) % 2, (y1 - y0) % 2), (0, 0))

    def test_people_sorted_by_presence_with_ids(self):
        dets = [det(60 + i * 3, 2, "M", k=50 + i) for i in range(6)] + [det(i * 3, 1, "F", k=i) for i in range(20)]
        people = sf.people_from_faces(dets, (608, 800))
        self.assertEqual([p["id"] for p in people], ["p1", "p2"])
        self.assertEqual([p["gender"] for p in people], ["female", "male"])
        self.assertNotIn("emb", people[0])


if __name__ == "__main__":
    unittest.main()
