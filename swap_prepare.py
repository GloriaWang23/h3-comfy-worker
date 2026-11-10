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
