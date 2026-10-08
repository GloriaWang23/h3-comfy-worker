"""Wrapper around runpod/worker-comfyui's handler.

Adds three things without touching the stock handler:
  - input.media: [{"name": "first.png", "url": "https://..."}] is downloaded into
    ComfyUI's input directory before the workflow runs, so LoadImage / LoadVideo /
    LoadAudio can reference it by name.
  - outputs are uploaded to Tencent COS (S3-compatible API) under COS_PREFIX and
    returned as COS_DOMAIN URLs, by replacing the stock handler's S3 upload hook.
  - a cancelled job stops generating: Runpod's cancel only changes the job status, so a
    watcher polls it and interrupts ComfyUI instead of letting the GPU finish (billed)
    work nobody will receive.
"""

import os
import re
import sys
import threading
import time
import urllib.request

import boto3
import requests
from botocore.config import Config

sys.path.insert(0, "/")
import handler as base  # noqa: E402  (the stock /handler.py)

INPUT_DIR = "/comfyui/input"
MAX_MEDIA_BYTES = 500 * 1024 * 1024
COMFY_URL = "http://127.0.0.1:8188"
CANCEL_POLL_SECONDS = 5

COS_BUCKET = os.environ.get("COS_BUCKET", "")
COS_REGION = os.environ.get("COS_REGION", "")
COS_DOMAIN = os.environ.get("COS_DOMAIN", "").rstrip("/")
COS_PREFIX = os.environ.get("COS_PREFIX", "").strip("/")

_cos = None


def cos_client():
    global _cos
    if _cos is None:
        opts = dict(s3={"addressing_style": "virtual"}, signature_version="s3v4")
        try:
            # botocore >= 1.36 sends uploads as aws-chunked with a trailing checksum by
            # default; COS then stores "Content-Encoding: aws-chunked" on the object.
            config = Config(**opts, request_checksum_calculation="when_required", response_checksum_validation="when_required")
        except TypeError:  # older botocore: never used aws-chunked
            config = Config(**opts)
        _cos = boto3.client(
            "s3",
            endpoint_url=f"https://cos.{COS_REGION}.myqcloud.com",
            region_name=COS_REGION,
            aws_access_key_id=os.environ["COS_SECRET_ID"],
            aws_secret_access_key=os.environ["COS_SECRET_KEY"],
            config=config,
        )
    return _cos


CONTENT_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".png": "image/png", ".jpg": "image/jpeg"}


def upload_to_cos(job_id, file_location, *args, **kwargs):
    """Drop-in for runpod's rp_upload.upload_image(job_id, file_location) -> url."""
    ext = os.path.splitext(file_location)[1].lower()
    key = f"{COS_PREFIX}/{time.strftime('%Y%m%d')}/{job_id}{ext}"
    cos_client().upload_file(
        file_location, COS_BUCKET, key,
        ExtraArgs={"ContentType": CONTENT_TYPES.get(ext, "application/octet-stream")},
    )
    return f"{COS_DOMAIN}/{key}"


def fetch_media(items):
    os.makedirs(INPUT_DIR, exist_ok=True)
    for item in items:
        name = os.path.basename(item["name"])
        if not re.fullmatch(r"[\w.\-]+", name):
            raise ValueError(f"bad media name: {item['name']!r}")
        req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0 (h3-comfy-worker)"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r, open(os.path.join(INPUT_DIR, name), "wb") as f:
            size = 0
            while chunk := r.read(1 << 20):
                size += len(chunk)
                if size > MAX_MEDIA_BYTES:
                    raise ValueError(f"{name} exceeds {MAX_MEDIA_BYTES >> 20} MB")
                f.write(chunk)
        print(f"h3-handler - fetched {name} ({size >> 10} KB) in {time.time() - t0:.1f}s")


def watch_cancel(job_id, done):
    """Interrupt ComfyUI once Runpod reports this job as cancelled."""
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID")
    key = os.environ.get("RUNPOD_AI_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not (endpoint and key):
        print("h3-handler - cancel watcher off (no endpoint id / api key in env)")
        return
    url = f"https://api.runpod.ai/v2/{endpoint}/status/{job_id}"
    while not done.wait(CANCEL_POLL_SECONDS):
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=10)
        except requests.RequestException:
            continue
        if r.status_code in (401, 403):
            print(f"h3-handler - cancel watcher off (status API answered {r.status_code})")
            return
        if r.ok and r.json().get("status") == "CANCELLED":
            print(f"h3-handler - job {job_id} cancelled, interrupting ComfyUI")
            try:
                requests.post(f"{COMFY_URL}/interrupt", timeout=10)
                requests.post(f"{COMFY_URL}/queue", json={"clear": True}, timeout=10)
            except requests.RequestException as e:
                print(f"h3-handler - interrupt failed: {e}")
            return


def h3_handler(job):
    media = job["input"].pop("media", None) or []
    try:
        fetch_media(media)
    except Exception as e:  # surfaced to the caller like the stock handler's errors
        return {"error": f"media download failed: {e}"}
    done = threading.Event()
    threading.Thread(target=watch_cancel, args=(job["id"], done), daemon=True).start()
    try:
        return base.handler(job)
    finally:
        done.set()


if COS_BUCKET:
    # The stock handler only takes its upload branch when BUCKET_ENDPOINT_URL is set.
    os.environ.setdefault("BUCKET_ENDPOINT_URL", f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com")
    base.rp_upload.upload_image = upload_to_cos

if __name__ == "__main__":
    print(f"h3-handler - starting (COS upload {'on' if COS_BUCKET else 'off'})")
    base.runpod.serverless.start({"handler": h3_handler})
