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

    def test_missing_model_fails_fast(self):
        r = h.h3_handler(self.job(unet="nope.safetensors"))
        self.assertIn("diffusion_models/nope.safetensors", r["error"])

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


if __name__ == "__main__":
    unittest.main(verbosity=1)
