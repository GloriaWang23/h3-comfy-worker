"""Wrapper around runpod/worker-comfyui's handler.

Adds these things without touching the stock handler:
  - input.media: [{"name": "first.png", "url": "https://..."}] is downloaded into
    ComfyUI's input directory before the workflow runs, so LoadImage / LoadVideo /
    LoadAudio can reference it by name.
  - outputs are uploaded to Tencent COS (S3-compatible API) under COS_PREFIX and
    returned as COS_DOMAIN URLs, by replacing the stock handler's S3 upload hook.
  - a cancelled job stops generating: Runpod's cancel only changes the job status, so a
    watcher polls it and interrupts ComfyUI instead of letting the GPU finish (billed)
    work nobody will receive.
  - model files are checked before the workflow is queued, so a wrong file name fails in
    milliseconds instead of after ComfyUI has started loading.
  - downloads and COS uploads retry, since a lost upload throws away a finished, billed job.
  - a dead ComfyUI process asks Runpod for a fresh worker instead of failing every later job.
  - per-job inputs and outputs are deleted afterwards, and stage timings are returned with
    the result (worker logs are gone once the container stops).
"""

import glob
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

import boto3
import requests
from botocore.config import Config

sys.path.insert(0, "/")
import handler as base  # noqa: E402  (the stock /handler.py)

INPUT_DIR = "/comfyui/input"
OUTPUT_DIR = "/comfyui/output"
MODELS_DIR = "/h3-models"
MAX_MEDIA_BYTES = 500 * 1024 * 1024
COMFY_URL = "http://127.0.0.1:8188"
CANCEL_POLL_SECONDS = 5
# Loader input -> model folder under MODELS_DIR (see extra_model_paths.yaml).
MODEL_INPUTS = {
    "unet_name": "diffusion_models",
    "clip_name": "text_encoders",
    "vae_name": "vae",
    "lora_name": "loras",
}

JOBS_DONE = 0

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


class Permanent(ValueError):
    """A failure that retrying cannot fix (bad link, file too large)."""


def retry(what, fn, attempts, first_delay=2.0):
    """Call fn(); on an exception wait 2 s, 4 s, ... and try again, re-raising the last one."""
    for i in range(attempts):
        try:
            return fn()
        except Permanent:
            raise
        except Exception as e:
            if i == attempts - 1:
                raise
            delay = first_delay * 2 ** i
            print(f"h3-handler - {what} failed ({e}), retry {i + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)


TIMINGS = {}


def upload_to_cos(job_id, file_location, *args, **kwargs):
    """Drop-in for runpod's rp_upload.upload_image(job_id, file_location) -> url."""
    ext = os.path.splitext(file_location)[1].lower()
    key = f"{COS_PREFIX}/{time.strftime('%Y%m%d')}/{job_id}{ext}"
    t0 = time.time()
    retry(
        f"COS upload {key}",
        lambda: cos_client().upload_file(
            file_location, COS_BUCKET, key,
            ExtraArgs={"ContentType": CONTENT_TYPES.get(ext, "application/octet-stream")},
        ),
        attempts=4,
    )
    TIMINGS["upload_s"] = round(TIMINGS.get("upload_s", 0) + time.time() - t0, 1)
    return f"{COS_DOMAIN}/{key}"


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (h3-comfy-worker)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        size = 0
        while chunk := r.read(1 << 20):
            size += len(chunk)
            if size > MAX_MEDIA_BYTES:
                raise Permanent(f"{os.path.basename(path)} exceeds {MAX_MEDIA_BYTES >> 20} MB")
            f.write(chunk)
    return size


def fetch_media(items):
    """Download each media URL into INPUT_DIR; returns the written paths."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    paths = []
    for item in items:
        name = os.path.basename(item["name"])
        if not re.fullmatch(r"[\w.\-]+", name):
            raise ValueError(f"bad media name: {item['name']!r}")
        path = os.path.join(INPUT_DIR, name)
        paths.append(path)
        t0 = time.time()

        def once():
            try:
                return download(item["url"], path)
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:  # a bad link stays bad
                    raise Permanent(f"{name}: HTTP {e.code}") from e
                raise

        size = retry(f"download {name}", once, attempts=3)
        print(f"h3-handler - fetched {name} ({size >> 10} KB) in {time.time() - t0:.1f}s")
    return paths


def missing_models(workflow):
    """Model files the workflow's loaders name that are not in MODELS_DIR."""
    missing = []
    for node in workflow.values():
        for key, folder in MODEL_INPUTS.items():
            name = node.get("inputs", {}).get(key)
            if isinstance(name, str) and not os.path.isfile(os.path.join(MODELS_DIR, folder, name)):
                missing.append(f"{folder}/{name}")
    return missing


def comfy_alive():
    try:
        return requests.get(f"{COMFY_URL}/system_stats", timeout=5).ok
    except requests.RequestException:
        return False


def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except OSError:
            pass


def cleanup(paths):
    """Delete this job's inputs and everything ComfyUI wrote to OUTPUT_DIR."""
    for p in paths:
        remove(p)
    for entry in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        if os.path.basename(entry) != "_output_images_will_be_put_here":
            remove(entry)


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
    global JOBS_DONE
    t_start = time.time()
    TIMINGS.clear()
    timings = {"worker_job": JOBS_DONE + 1}  # 1 = first job on this worker (models not loaded yet)
    JOBS_DONE += 1
    workflow = job["input"].get("workflow") or {}
    missing = missing_models(workflow)
    if missing:
        return {"error": f"model files not found: {', '.join(missing)}"}
    media = job["input"].pop("media", None) or []
    paths = []
    try:
        t0 = time.time()
        try:
            paths = fetch_media(media)
        except Exception as e:  # surfaced to the caller like the stock handler's errors
            return {"error": f"media download failed: {e}"}
        timings["media_s"] = round(time.time() - t0, 1)
        done = threading.Event()
        threading.Thread(target=watch_cancel, args=(job["id"], done), daemon=True).start()
        t0 = time.time()
        try:
            result = base.handler(job)
        finally:
            done.set()
        timings["comfy_s"] = round(time.time() - t0 - TIMINGS.get("upload_s", 0), 1)
        timings.update(TIMINGS)
        timings["total_s"] = round(time.time() - t_start, 1)
        if isinstance(result, dict):
            result["timings"] = timings
            if result.get("error") and not comfy_alive():
                # Every later job on this worker would fail the same way; let Runpod replace it.
                print("h3-handler - ComfyUI is not responding, requesting a fresh worker")
                result["refresh_worker"] = True
        print(f"h3-handler - timings {timings}")
        return result
    finally:
        cleanup(paths)


if COS_BUCKET:
    # The stock handler only takes its upload branch when BUCKET_ENDPOINT_URL is set.
    os.environ.setdefault("BUCKET_ENDPOINT_URL", f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com")
    base.rp_upload.upload_image = upload_to_cos

if __name__ == "__main__":
    print(f"h3-handler - starting (COS upload {'on' if COS_BUCKET else 'off'})")
    base.runpod.serverless.start({"handler": h3_handler})
