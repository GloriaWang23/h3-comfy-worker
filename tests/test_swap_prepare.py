"""Unit tests for swap_prepare.py (character-swap template preprocessing). Needs numpy + Pillow.

Run: python tests/test_swap_prepare.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Local test servers; never send them through a developer machine's HTTP proxy.
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(k, None)
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

    def test_shot_graph_sam_runs_without_cudnn(self):
        # SAM3 text detection fails with "cuDNN Frontend error: No valid execution plans" on this
        # image (torch 2.11 cu130, RTX 5090): detect / track go through the cuDNN-off wrappers.
        g = sp.build_shot_graph("cs_prepared.mp4", 0, 48, "cs_s0")
        types_ = [n["class_type"] for n in g.values()]
        self.assertIn("CSwapSAM3Detect", types_)
        self.assertIn("CSwapSAM3VideoTrack", types_)
        self.assertNotIn("SAM3_Detect", types_)
        self.assertNotIn("SAM3_VideoTrack", types_)

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


class Ready(unittest.TestCase):
    def test_wait_ready_up(self):
        import http.server, threading
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
            def log_message(self, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.assertTrue(sp.wait_ready(f"http://127.0.0.1:{srv.server_address[1]}", timeout_s=5))

    def test_wait_ready_down(self):
        self.assertFalse(sp.wait_ready("http://127.0.0.1:9", timeout_s=1.5, every_s=0.5))

    def test_comfy_error_names_node_and_trace(self):
        st = {"status_str": "error", "messages": [["execution_error", {"node_id": "detect", "node_type": "CSwapSAM3Detect",
              "exception_message": "cuDNN Frontend error", "traceback": ["a", "File x.py, line 3, in forward\n", "  attn(q, k, v)\n"]}]]}
        msg = sp.comfy_error(st)
        self.assertIn("CSwapSAM3Detect", msg)
        self.assertIn("cuDNN Frontend error", msg)
        self.assertIn("attn(q, k, v)", msg)


class ManifestV2(unittest.TestCase):
    def test_key_frames_even_and_include_ends(self):
        self.assertEqual(sp.key_frames(240, 8), [0, 34, 68, 102, 137, 171, 205, 239])
        self.assertEqual(sp.key_frames(5, 8), [0, 1, 2, 3, 4])

    def test_trim_command(self):
        self.assertIsNone(sp.trim_command("in.mp4", "out.mp4", 0, 8.0))          # nothing to cut
        cmd = sp.trim_command("in.mp4", "out.mp4", 4.5, 20.0)
        self.assertEqual(cmd[:2], ["ffmpeg", "-hide_banner"])
        self.assertEqual(cmd[cmd.index("-ss") + 1], "4.500")
        self.assertEqual(cmd[cmd.index("-t") + 1], "10.000")                       # at most 10 s kept
        self.assertEqual(cmd[-1], "out.mp4")
        self.assertEqual(sp.trim_command("in.mp4", "out.mp4", 0, 15.0)[sp.trim_command("in.mp4", "out.mp4", 0, 15.0).index("-t") + 1], "10.000")

    def test_manifest_source_and_start(self):
        urls = {"cs_prepared.mp4": "u/p.mp4", "cs_thumb.jpg": "u/t.jpg", "cs_f0.jpg": "u/f0.jpg", "cs_source.mp4": "u/s.mp4"}
        m = sp.build_manifest([], urls, frames=240, wh=(608, 800), seconds=10.0, dur=0.0, had_audio=True, resolution="480", key=[0], start=4.5)
        self.assertEqual((m["source"], m["start"]), ("u/s.mp4", 4.5))
        m0 = sp.build_manifest([], {k: v for k, v in urls.items() if k != "cs_source.mp4"}, frames=240, wh=(608, 800), seconds=10.0, dur=0.0, had_audio=True, resolution="480", key=[0])
        self.assertEqual((m0["source"], m0["start"]), (None, 0))

    def test_manifest_people_from_faces(self):
        people = [{"id": "p1", "gender": "female", "gender_conf": 1.0, "age": 28, "faces": 55, "first_frame": 0,
                   "last_frame": 171, "frame": 18, "face_box": [284, 122, 437, 325], "crop_box": [130, 20, 590, 800]}]
        urls = {"cs_prepared.mp4": "u/p.mp4", "cs_thumb.jpg": "u/t.jpg", "cs_p1_crop.png": "u/c1.png",
                "cs_f0.jpg": "u/f0.jpg", "cs_f1.jpg": "u/f1.jpg"}
        m = sp.build_manifest(people, urls, frames=240, wh=(608, 800), seconds=10.03, dur=0.0, had_audio=True,
                              resolution="480", key=[0, 239])
        self.assertEqual(m["version"], 2)
        self.assertEqual(m["people"][0]["crop"], "u/c1.png")
        self.assertEqual(m["people"][0]["gender"], "female")
        self.assertEqual(m["frames"], [{"frame": 0, "url": "u/f0.jpg"}, {"frame": 239, "url": "u/f1.jpg"}])
        self.assertEqual((m["thumb"], m["prepared_video"], m["source_frames"]), ("u/t.jpg", "u/p.mp4", 240))
        self.assertNotIn("shots", m)


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
