"""Unit tests for h3_handler.py with the stock handler, boto3 and requests stubbed out.

Run: python3 tests/test_h3_handler.py
"""
import http.server
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- stubs for modules that only exist inside the worker image ---------------------------
uploads = []
fail_uploads = {"n": 0}


class FakeS3:
    def upload_file(self, path, bucket, key, ExtraArgs=None):
        if fail_uploads["n"] > 0:
            fail_uploads["n"] -= 1
            raise ConnectionError("cos down")
        uploads.append(key)


sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: FakeS3())
botocore = types.ModuleType("botocore")
botocore_config = types.ModuleType("botocore.config")
botocore_config.Config = lambda **k: k
sys.modules["botocore"] = botocore
sys.modules["botocore.config"] = botocore_config
comfy_alive = {"v": True}
sys.modules["requests"] = types.SimpleNamespace(
    get=lambda *a, **k: types.SimpleNamespace(ok=comfy_alive["v"], status_code=200, json=lambda: {}),
    post=lambda *a, **k: None,
    RequestException=Exception,
)
base_result = {"v": {"images": []}}
base = types.ModuleType("handler")
base.rp_upload = types.SimpleNamespace(upload_image=None)
base.handler = lambda job: base_result["v"]
history = {"v": {}}
base.get_history = lambda prompt_id: history["v"]
base.runpod = types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda cfg: None))
sys.modules["handler"] = base

# The test server is local; never send it through a developer machine's HTTP proxy.
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(k, None)
os.environ.update(COS_BUCKET="b", COS_REGION="r", COS_DOMAIN="https://cdn", COS_PREFIX="p",
                  COS_SECRET_ID="x", COS_SECRET_KEY="y")
sys.path.insert(0, ROOT)
import h3_handler as h  # noqa: E402

h.time.sleep = lambda s: None  # no real backoff waits in tests


class Server(http.server.BaseHTTPRequestHandler):
    hits = {}

    def do_GET(self):
        n = Server.hits[self.path] = Server.hits.get(self.path, 0) + 1
        if self.path == "/flaky.png" and n == 1:
            self.send_error(503)
        elif self.path == "/missing.png":
            self.send_error(404)
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * 1024)

    def log_message(self, *a):
        pass


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.tmp = tempfile.mkdtemp()
        h.INPUT_DIR = os.path.join(cls.tmp, "input")
        h.OUTPUT_DIR = os.path.join(cls.tmp, "output")
        h.MODELS_DIR = os.path.join(cls.tmp, "models")
        os.makedirs(os.path.join(h.MODELS_DIR, "diffusion_models"))
        open(os.path.join(h.MODELS_DIR, "diffusion_models", "unet.safetensors"), "w").close()
        h.watch_cancel = lambda job_id, done: None

    def setUp(self):
        uploads.clear()
        comfy_alive["v"] = True
        base_result["v"] = {"images": []}
        os.makedirs(h.OUTPUT_DIR, exist_ok=True)

    def job(self, media=(), unet="unet.safetensors"):
        return {"id": "j1", "input": {"workflow": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet}}}, "media": list(media)}}

    def test_cos_key_default_unchanged(self):
        with mock.patch.object(h.time, "strftime", return_value="20261110"):
            self.assertEqual(h.cos_key("abc", "/x/out/whatever.mp4", False), "p/20261110/abc.mp4")

    def test_cos_key_keep_names(self):
        with mock.patch.object(h.time, "strftime", return_value="20261110"):
            self.assertEqual(h.cos_key("abc", "/x/out/cs_s0_o1.npz", True), "p/20261110/abc/cs_s0_o1.npz")

    def test_upload_files_returns_urls(self):
        f = os.path.join(self.tmp, "cs_s0.png"); open(f, "wb").write(b"x")
        with mock.patch.object(h, "UPLOAD_KEEP_NAMES", True), mock.patch.object(h.time, "strftime", return_value="20261110"):
            self.assertEqual(h.upload_files("j9", [f]), ["https://cdn/p/20261110/j9/cs_s0.png"])

    def test_prepare_task_routes_to_swap_prepare(self):
        fake = types.ModuleType("swap_prepare")
        seen = {}
        fake.run_prepare = lambda job_input, job_id, **ctx: seen.update(media=job_input.get("media")) or {"source_frames": 240, "shots": []}
        with mock.patch.dict(sys.modules, {"swap_prepare": fake}):
            r = h.h3_handler({"id": "jp", "input": {"task": "prepare", "media": [{"name": "cs_src.mp4", "url": f"{self.url}/v.mp4"}]}})
        self.assertEqual(r["prepare"]["source_frames"], 240)
        self.assertEqual([m["name"] for m in seen["media"]], ["cs_src.mp4"])
        self.assertIn("timings", r)

    def test_swap_mask_plan_built_before_run(self):
        calls = []
        fake = types.ModuleType("swap_prepare")
        fake.build_mask_file = lambda plan, d: calls.append(plan["name"]) or os.path.join(d, plan["name"])
        base_result["v"] = {"images": []}
        with mock.patch.dict(sys.modules, {"swap_prepare": fake}):
            h.h3_handler({"id": "js", "input": {"task": "swap", "workflow": {}, "media": [],
                                                 "mask_plan": {"name": "mask_p.npz", "total": 1, "parts": []}}})
        self.assertEqual(calls, ["mask_p.npz"])

    def test_bad_mask_plan_fails_job(self):
        fake = types.ModuleType("swap_prepare")
        def boom(plan, d):
            raise ValueError("mask frames 3 != source_frames 4")
        fake.build_mask_file = boom
        with mock.patch.dict(sys.modules, {"swap_prepare": fake}):
            r = h.h3_handler({"id": "jb", "input": {"task": "swap", "workflow": {}, "media": [],
                                                     "mask_plan": {"name": "m.npz", "total": 4, "parts": []}}})
        self.assertIn("!= source_frames", r["error"])

    def test_missing_model_fails_fast(self):
        r = h.h3_handler(self.job(unet="nope.safetensors"))
        self.assertIn("diffusion_models/nope.safetensors", r["error"])

    def test_execution_error_fails_the_job(self):
        base_result["v"] = {"images": [{"type": "s3_url", "data": "u"}], "errors": ["Workflow execution error: Node Type: KSampler, " + "x" * 5000 + " nvrtc: error"]}
        r = h.h3_handler(self.job())
        self.assertNotIn("images", r)
        self.assertTrue(r["error"].startswith("Workflow execution error: Node Type: KSampler"))
        self.assertTrue(r["error"].endswith("nvrtc: error"))
        self.assertLess(len(r["error"]), 1500)

    def test_warnings_alone_keep_the_result(self):
        base_result["v"] = {"images": [{"type": "s3_url", "data": "u"}], "errors": ["Node 9 produced unhandled output keys"]}
        self.assertEqual(h.h3_handler(self.job())["images"][0]["data"], "u")

    def test_input_previews_not_uploaded(self):
        history["v"] = {"p": {"outputs": {"1": {"images": [{"filename": "source.mp4", "type": "input"}]},
                                          "2": {"images": [{"filename": "out.mp4", "type": "output"}]}}}}
        out = h.base.get_history("p")["p"]["outputs"]
        self.assertEqual(out["1"]["images"], [])
        self.assertEqual(out["2"]["images"][0]["filename"], "out.mp4")

    def test_flashvsr_loaders_checked_and_none_skipped(self):
        wf = {"1": {"class_type": "FlashVSR_SM_Model", "inputs": {"dit": "turbo_dit_plus.safetensors"}},
              "2": {"class_type": "FlashVSR_SM_Vae", "inputs": {"vae": "none", "tcd_encoder": "none"}}}
        self.assertEqual(h.missing_models(wf), ["FlashVSR/turbo_dit_plus.safetensors"])

    def test_download_retries_5xx_then_succeeds_and_cleans_up(self):
        r = h.h3_handler(self.job([{"name": "first.png", "url": self.url + "/flaky.png"}]))
        self.assertNotIn("error", r)
        self.assertEqual(Server.hits["/flaky.png"], 2)
        self.assertFalse(os.path.exists(os.path.join(h.INPUT_DIR, "first.png")), "input removed after the job")
        self.assertEqual(r["timings"]["worker_job"] >= 1, True)
        self.assertIn("total_s", r["timings"])

    def test_404_is_not_retried(self):
        r = h.h3_handler(self.job([{"name": "first.png", "url": self.url + "/missing.png"}]))
        self.assertIn("HTTP 404", r["error"])
        self.assertEqual(Server.hits["/missing.png"], 1)

    def test_upload_retries(self):
        fail_uploads["n"] = 2
        path = os.path.join(h.OUTPUT_DIR, "v.mp4")
        open(path, "w").close()
        url = h.upload_to_cos("j1", path)
        self.assertEqual(url, "https://cdn/p/" + uploads[0].split("p/", 1)[1])
        self.assertEqual(len(uploads), 1)

    def test_dead_comfy_requests_fresh_worker(self):
        base_result["v"] = {"error": "ComfyUI server not reachable"}
        comfy_alive["v"] = False
        r = h.h3_handler(self.job())
        self.assertTrue(r.get("refresh_worker"))

    def test_error_with_live_comfy_keeps_worker(self):
        base_result["v"] = {"error": "Workflow execution error"}
        r = h.h3_handler(self.job())
        self.assertNotIn("refresh_worker", r)

    def test_outputs_cleaned(self):
        sub = os.path.join(h.OUTPUT_DIR, "video")
        os.makedirs(sub)
        open(os.path.join(sub, "a.mp4"), "w").close()
        h.h3_handler(self.job())
        self.assertEqual(os.listdir(h.OUTPUT_DIR), [])


class RefMediaTests(unittest.TestCase):
    """prepare_refs with real ffmpeg / ffprobe on generated clips."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def clip(self, name, seconds, fps=30, size="320x240", audio=True, extra=()):
        import subprocess
        path = os.path.join(self.dir, name)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=s={size}:d={seconds}:r={fps}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", f"sine=d={seconds}", "-c:a", "libopus" if name.endswith(".webm") else "aac", "-shortest"]
        w, hgt = (int(x) for x in size.split("x"))
        vcodec = "libvpx-vp9" if name.endswith(".webm") else "libx264" if (w % 2 == 0 and hgt % 2 == 0) else "libx264rgb"
        cmd += ["-c:v", vcodec, *extra, path]
        subprocess.run(cmd, check=True)
        return path

    def fps(self, path):
        return h.probe(path)["fps"]

    def test_30_and_60_fps_become_24(self):
        for fps in (30, 60):
            p = self.clip(f"refvideo{fps}.mp4", 3, fps=fps)
            h.prepare_refs([p])
            self.assertAlmostEqual(self.fps(p), 24, delta=0.05)
            self.assertAlmostEqual(h.probe(p)["duration"], 3, delta=0.15)
            self.assertTrue(h.probe(p)["audio"], "soundtrack kept")

    def test_24_and_23976_untouched(self):
        for fps in ("24", "24000/1001"):
            p = self.clip(f"refvideo_{fps.replace('/', '_')}.mp4", 3, fps=fps)
            before = os.path.getmtime(p), os.path.getsize(p)
            h.prepare_refs([p])
            self.assertEqual((os.path.getmtime(p), os.path.getsize(p)), before)

    def test_odd_size_and_no_audio(self):
        p = self.clip("refvideo0.mp4", 3, size="321x241", audio=False)
        h.prepare_refs([p])
        info = h.probe(p)
        self.assertAlmostEqual(info["fps"], 24, delta=0.05)
        self.assertEqual(info["audio"], [])

    def test_webm(self):
        p = self.clip("refvideo0.webm", 3, extra=())
        h.prepare_refs([p])
        self.assertAlmostEqual(self.fps(p), 24, delta=0.05)

    def test_duration_limits(self):
        ok = self.clip("refvideo0.mp4", 15.08, fps=24)
        h.prepare_refs([ok])
        too_long = self.clip("refvideo1.mp4", 20, fps=24)
        with self.assertRaises(h.MediaError):
            h.prepare_refs([too_long])
        short = self.clip("refvideo2.mp4", 1, fps=24)
        with self.assertRaises(h.MediaError):
            h.prepare_refs([short])

    def test_total_over_15(self):
        a = self.clip("refvideo0.mp4", 9, fps=24)
        b = self.clip("refvideo1.mp4", 9, fps=24)
        with self.assertRaises(h.MediaError):
            h.prepare_refs([a, b])

    def test_guide_video(self):
        p = self.clip("guidevideo.mp4", 4)
        with self.assertRaises(h.MediaError):
            h.prepare_refs([p], [{"name": "guidevideo.mp4", "min_seconds": 5}])
        h.prepare_refs([p], [{"name": "guidevideo.mp4", "min_seconds": 4}])
        self.assertAlmostEqual(self.fps(p), 24, delta=0.05)

    def test_guide_without_audio_gets_silence(self):
        p = self.clip("guidevideo.mp4", 3, fps=24, audio=False)
        h.prepare_refs([p], [{"name": "guidevideo.mp4", "min_seconds": 3}])
        info = h.probe(p)
        self.assertTrue(info["audio"])
        self.assertAlmostEqual(info["duration"], 3, delta=0.15)

    def test_other_files_untouched(self):
        p = self.clip("cs_src.mp4", 20, fps=30)
        h.prepare_refs([p])
        self.assertAlmostEqual(self.fps(p), 30, delta=0.05)

    def test_broken_file_and_no_leftovers(self):
        p = os.path.join(self.dir, "refvideo0.mp4")
        open(p, "wb").write(b"not a video")
        with self.assertRaises(h.MediaError):
            h.prepare_refs([p])
        q = self.clip("refvideo1.mp4", 3)
        with mock.patch.object(h, "CONVERT_TIMEOUT_S", 0.001), self.assertRaises(h.MediaError):
            h.prepare_refs([q])
        self.assertEqual([f for f in os.listdir(self.dir) if f.startswith(".conv_")], [])

    def test_handler_returns_media_error(self):
        with mock.patch.object(h, "fetch_media", return_value=[self.clip("refvideo0.mp4", 20, fps=24)]):
            r = h.h3_handler({"id": "j", "input": {"workflow": {}, "media": [{"name": "refvideo0.mp4", "url": "http://x"}]}})
        self.assertIn("2–15", r["error"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
