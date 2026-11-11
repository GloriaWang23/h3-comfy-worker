"""Character-swap template preprocessing (endpoint character-swap, task "prepare").

Pure helpers (numpy / Pillow, CPU) used by the prepare flow and by the swap mask plan:
shot detection on the prepared 24 fps frames, best frame / crop box of a person mask, and
concatenation of per-shot masks into one timeline mask.
"""
import numpy as np
from PIL import Image

HIST_BINS = (16, 4, 4)  # H, S, V


def _hsv_hist(frame):
    hsv = np.asarray(Image.fromarray(frame).resize((64, 64), Image.Resampling.BILINEAR).convert("HSV"))
    idx = [(hsv[..., c].astype(np.int32) * n) // 256 for c, n in enumerate(HIST_BINS)]
    flat = (idx[0] * HIST_BINS[1] + idx[1]) * HIST_BINS[2] + idx[2]
    h = np.bincount(flat.ravel(), minlength=int(np.prod(HIST_BINS))).astype(np.float64)
    return h / h.sum()


def detect_shots(frames, threshold=0.35, min_len=12):
    """Half-open [start, end) shots covering all frames. A cut is a Bhattacharyya distance between
    consecutive HSV histograms above `threshold`; shots shorter than `min_len` join the previous one
    (or the next one, for a short first shot)."""
    n = len(frames)
    if n == 0:
        return []
    hists = [_hsv_hist(f) for f in frames]

    def dist(a, b):
        return np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(hists[a] * hists[b])))))

    cuts = [i for i in range(1, n) if dist(i - 1, i) > threshold]
    bounds = [0] + cuts + [n]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    merged = []
    i = 0
    while i < len(shots):
        s, e = shots[i]
        if merged and e - s < min_len:
            # A short flash joins the previous shot; if the scene after it matches the one
            # before it, that next shot is the same shot too.
            if i + 1 < len(shots) and dist(merged[-1][1] - 1, shots[i + 1][0]) <= threshold:
                e = shots[i + 1][1]
                i += 1
            merged[-1] = (merged[-1][0], e)
        elif merged and merged[-1][1] - merged[-1][0] < min_len:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
        i += 1
    return merged


def best_frame(mask):
    """Index of the frame where the mask covers the largest area."""
    return int(np.argmax(np.asarray(mask).reshape(len(mask), -1).sum(axis=1)))


def crop_box(mask2d, margin=0.15):
    """(x0, y0, x1, y1) around the mask, grown by `margin` of its size, clipped to the frame,
    with even width and height."""
    m = np.asarray(mask2d) > 0.5
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return (0, 0, w - w % 2, h - h % 2)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = int(round((x1 - x0) * margin)), int(round((y1 - y0) * margin))
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    if (x1 - x0) % 2:
        x1 = x1 + 1 if x1 < w else x1 - 1
    if (y1 - y0) % 2:
        y1 = y1 + 1 if y1 < h else y1 - 1
    return (int(x0), int(y0), int(x1), int(y1))


def concat_masks(parts, total):
    """Join per-shot masks ([n, H, W] arrays) and gaps (ints = that many empty frames) into one
    bool [total, H, W] mask."""
    shape = next((p.shape[1:] for p in parts if not isinstance(p, (int, np.integer))), None)
    if shape is None:
        raise ValueError("mask plan has no mask part")
    out = [np.zeros((int(p), *shape), bool) if isinstance(p, (int, np.integer)) else np.asarray(p) > 0.5
           for p in parts]
    joined = np.concatenate(out) if out else np.zeros((0, *shape), bool)
    if len(joined) != total:
        raise ValueError(f"mask frames {len(joined)} != source_frames {total}")
    return joined


# ----------------------------------------------------------------------------------------------
# prepare flow: ComfyUI graphs + manifest

# MiniMax H3 generates at most 362 frames (17n+5 grid) = 15.08 s at 24 fps in one go; templates use up
# to that. Longer videos keep 15 s (from the chosen start).
KEEP_SECONDS = 15.0
MAX_SECONDS = 15.5
MAX_OBJECTS = 3
SAM_FILE = "sam3.1_multiplex_fp16.safetensors"


def duration_param(seconds):
    """H3RelayPrepareSource duration_seconds for a template: 0 = whole clip, else the first 15 s."""
    return 0.0 if seconds <= MAX_SECONDS else KEEP_SECONDS


def build_prepare_graph(video, resolution, duration_seconds):
    """Prepare the template exactly as the swap graph does (24 fps, resolution) and save it."""
    return {
        "source": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "prepare": {"class_type": "H3RelayPrepareSource", "inputs": {
            "video": ["source", 0], "resolution": resolution, "duration_seconds": float(duration_seconds),
            "missing_audio": "silence"}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["prepare", 8], "filename_prefix": "cs_prepared", "format": "mp4", "format.codec": "h264",
            "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16}},
    }


def build_shot_graph(prepared, start, length, prefix, max_objects=MAX_OBJECTS):
    """One shot: SAM3 detects people on its first frame, tracks them through the shot (re-detecting
    by text every frame), and saves one mask sequence per object slot (empty when unused)."""
    g = {
        "video": {"class_type": "LoadVideo", "inputs": {"file": prepared}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["video", 0]}},
        "shot": {"class_type": "ImageFromBatch", "inputs": {"image": ["parts", 0], "batch_index": start, "length": length}},
        "first": {"class_type": "ImageFromBatch", "inputs": {"image": ["shot", 0], "batch_index": 0, "length": 1}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_FILE}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": "person"}},
        # cuDNN-off wrappers (comfy_nodes/cswap_nodes): SAM3 text detection fails with cuDNN on this image
        "detect": {"class_type": "CSwapSAM3Detect", "inputs": {
            "model": ["sam", 0], "image": ["first", 0], "conditioning": ["text", 0], "threshold": 0.35,
            "refine_iterations": 2, "individual_masks": True}},
        "track": {"class_type": "CSwapSAM3VideoTrack", "inputs": {
            "model": ["sam", 0], "images": ["shot", 0], "initial_mask": ["detect", 0], "conditioning": ["text", 0],
            "detection_threshold": 0.5, "max_objects": max_objects, "detect_interval": 1}},
    }
    for i in range(max_objects):
        g[f"mask{i}"] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["track", 0], "object_indices": str(i)}}
        g[f"save{i}"] = {"class_type": "H3RelaySaveMask", "inputs": {
            "masks": [f"mask{i}", 0], "filename_prefix": f"{prefix}_o{i}", "frames": length}}
    return g


def people_from_masks(masks_by_obj, frames):
    """Non-empty objects of one shot -> [{obj, frame, bbox, area, crop (H,W,3 uint8)}]."""
    out = []
    for obj, m in sorted(masks_by_obj.items()):
        m = np.asarray(m) > 0.5
        if not m.any():
            continue
        f = best_frame(m)
        x0, y0, x1, y1 = crop_box(m[f])
        out.append({"obj": int(obj), "frame": f, "bbox": [x0, y0, x1, y1],
                    "area": round(float(m[f].mean()), 4), "crop": np.asarray(frames[f])[y0:y1, x0:x1]})
    return out


def build_mask_file(plan, input_dir):
    """swap input.mask_plan {name, total, parts: [{media} | {zeros}]} -> <input_dir>/<name> (key 'mask')."""
    import os
    parts = []
    for p in plan["parts"]:
        if "zeros" in p:
            parts.append(int(p["zeros"]))
        else:
            with np.load(os.path.join(input_dir, os.path.basename(p["media"])), allow_pickle=False) as z:
                parts.append(z["mask"])
    mask = concat_masks(parts, int(plan["total"]))
    path = os.path.join(input_dir, os.path.basename(plan["name"]))
    np.savez_compressed(path, mask=mask)
    return path


def wait_ready(url, timeout_s=180, every_s=2.0):
    """True once ComfyUI answers /system_stats. The prepare task bypasses the stock handler, which
    otherwise does this wait, and is often the first job on a cold worker."""
    import time, urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if urllib.request.urlopen(f"{url}/system_stats", timeout=5).status == 200:
                return True
        except Exception:
            pass
        time.sleep(every_s)
    return False


def comfy_error(status):
    """'ComfyUI error in <node_type> (<node_id>): <message> | <last traceback lines>'."""
    e = next((m[1] for m in status.get("messages", []) if m[0] == "execution_error"), {})
    tail = " ".join(x.strip() for x in (e.get("traceback") or [])[-2:])
    return (f"ComfyUI error in {e.get('node_type', '?')} ({e.get('node_id', '?')}): "
            f"{str(e.get('exception_message', '')).strip()[:400]} | {tail[-600:]}")


def comfy_run(graph, url, timeout_s=900):
    """Queue a graph on the local ComfyUI and wait; raises on an execution error."""
    import json, time, urllib.request, uuid
    pid = str(uuid.uuid4())
    req = urllib.request.Request(f"{url}/prompt", json.dumps({"prompt": graph, "prompt_id": pid}).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(1)
        h = json.load(urllib.request.urlopen(f"{url}/history/{pid}", timeout=30))
        st = h.get(pid, {}).get("status", {})
        if st.get("status_str") == "error":
            raise RuntimeError(comfy_error(st))
        if st.get("completed"):
            return h[pid]
    raise TimeoutError("ComfyUI prepare step timed out")


def _decode(path):
    import av
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def _fps(path):
    import av
    with av.open(path) as c:
        return float(c.streams.video[0].average_rate or 24)


def original_crop(orig, orig_fps, idx24, bbox, prepared_wh):
    """Crop a person from the original video (sharper than the prepared frames): prepared frame
    idx24 at 24 fps -> nearest original frame, bbox scaled from the prepared size."""
    i = min(len(orig) - 1, int(round(idx24 * orig_fps / 24.0)))
    oh, ow = orig.shape[1:3]
    sx, sy = ow / prepared_wh[0], oh / prepared_wh[1]
    x0, y0, x1, y1 = bbox
    return np.ascontiguousarray(orig[i, int(y0 * sy):int(round(y1 * sy)), int(x0 * sx):int(round(x1 * sx))])


def _probe(path):
    import av
    with av.open(path) as c:
        v = c.streams.video[0]
        secs = float(c.duration / 1e6) if c.duration else float(v.duration * v.time_base)
        return secs, len(c.streams.audio) > 0


def key_frames(n, count=None):
    """`count` evenly spaced frame indices including the first and the last (all frames if fewer).
    Default: 8 stills up to ~10 s, 12 for longer clips (the VLM takes at most 16 images with crops)."""
    if count is None:
        count = 8 if n <= 250 else 12
    if n <= count:
        return list(range(n))
    return [int(round(i * (n - 1) / (count - 1))) for i in range(count)]


def trim_command(src, dst, start, seconds, keep=KEEP_SECONDS, end=None):
    """ffmpeg command that keeps [start, end) — at most `keep` seconds — re-encoded near-losslessly
    with audio; without `end`, `keep` seconds from `start`. None when that is the whole clip already."""
    start = max(0.0, start)
    if end is not None and end > start:
        if start <= 0 and end >= seconds - 0.05 and seconds <= MAX_SECONDS:
            return None
        keep = min(keep, end - start)
    elif start <= 0 and seconds <= MAX_SECONDS:
        return None
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", src,
            "-t", f"{keep:.3f}", "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst]


def build_manifest(people, urls, *, frames, wh, seconds, dur, had_audio, resolution, key, start=0, end=None):
    """Manifest v2: people come from face identity (swap_faces), each with a head + upper-body crop;
    `frames` are evenly spaced stills for the semantic (VLM) pass and the editor. No shots, no masks:
    the default designations (role / description / crop) need neither."""
    return {
        "version": 2,
        "source_frames": int(frames), "fps": 24, "width": int(wh[0]), "height": int(wh[1]),
        "duration_s": round(seconds, 2), "duration_param": dur, "trimmed": dur > 0, "had_audio": had_audio,
        "resolution": resolution, "prepared_video": urls["cs_prepared.mp4"], "thumb": urls["cs_thumb.jpg"],
        # when the template was cut (longer than 15 s, or a start offset), swaps run on this clip
        "source": urls.get("cs_source.mp4"), "start": start, "end": end,
        "frames": [{"frame": int(f), "url": urls[f"cs_f{i}.jpg"]} for i, f in enumerate(key)],
        "people": [{**p, "crop": urls[f"cs_{p['id']}_crop.png"]} for p in people],
    }


def run_prepare(job_input, job_id, *, input_dir, output_dir, upload, comfy_url, resolution="480"):
    """task "prepare": template video (input.media[0]) -> manifest v2 (people by face identity)."""
    import glob, os
    from PIL import Image
    import swap_faces
    if not wait_ready(comfy_url):
        raise RuntimeError("ComfyUI did not come up")
    import subprocess
    name = os.path.basename(job_input["media"][0]["name"])
    seconds, had_audio = _probe(os.path.join(input_dir, name))
    start = float(job_input.get("start") or 0)
    end = float(job_input.get("end") or 0) or None
    files = []
    cut = trim_command(os.path.join(input_dir, name), os.path.join(input_dir, "cs_source.mp4"), start, seconds, end=end)
    if cut:
        subprocess.run(cut, check=True, timeout=300)
        name = "cs_source.mp4"
        files.append(os.path.join(input_dir, name))
        seconds, had_audio = _probe(os.path.join(input_dir, name))
    dur = duration_param(seconds)
    comfy_run(build_prepare_graph(name, resolution, dur), comfy_url)
    saved = sorted(glob.glob(os.path.join(output_dir, "**", "cs_prepared*.mp4"), recursive=True))[-1]
    prepared = os.path.join(input_dir, "cs_prepared.mp4")
    os.replace(saved, prepared)
    frames = _decode(prepared)
    n, h, w = frames.shape[:3]
    people = swap_faces.people_from_faces(swap_faces.detect_faces(frames), (w, h))
    orig, orig_fps = _decode(os.path.join(input_dir, name)), _fps(os.path.join(input_dir, name))
    files.append(prepared)

    def save(arr, fname, **kw):
        path = os.path.join(output_dir, fname)
        Image.fromarray(np.ascontiguousarray(arr)).save(path, **kw)
        files.append(path)

    save(frames[0], "cs_thumb.jpg", quality=88)
    key = key_frames(n)
    for i, f in enumerate(key):
        save(frames[f], f"cs_f{i}.jpg", quality=90)
    for p in people:   # crops come from the original video: sharper than the prepared frames
        save(original_crop(orig, orig_fps, p["frame"], p["crop_box"], (w, h)), f"cs_{p['id']}_crop.png")
    urls = dict(zip([os.path.basename(f) for f in files], upload(job_id, files)))
    return build_manifest(people, urls, frames=n, wh=(w, h), seconds=seconds, dur=dur, had_audio=had_audio,
                          resolution=resolution, key=key, start=start, end=end)
