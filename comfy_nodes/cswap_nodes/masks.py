"""Pure helpers of the seed-track mask node (numpy only, unit-tested outside ComfyUI).

A person of a character-swap template is known by face identity: the frame where the face is clearest
and its box there. A region mask starts from that seed (points for SAM3), is tracked forward and
backward from the seed frame (BFS's key-frame approach), and frames where tracking lost the person are
either freed ("open": no restriction, the prompt still names the person) or left empty ("closed").
"""
import numpy as np


def seed_points(face_box, kind, wh):
    """SAM3 point prompts from a normalised face box [x0, y0, x1, y1]: the face centre, and a torso
    point 1.5 face heights below the chin: positive for the whole person, negative for the head."""
    w, h = wh
    x0, y0, x1, y1 = face_box
    cx = min(w - 1, max(0, int(round((x0 + x1) / 2 * w))))
    face = {"x": cx, "y": min(h - 1, max(0, int(round((y0 + y1) / 2 * h))))}
    torso = {"x": cx, "y": min(h - 1, max(0, int(round((y1 + 1.5 * (y1 - y0)) * h))))}
    return ([face, torso], []) if kind == "person" else ([face], [torso])


def stitch(forward, backward, seed, total):
    """Masks tracked forward from the seed frame ([seed..]) and backward ([seed, seed-1, .. 0], i.e.
    reversed time) -> one [total, H, W] sequence in time order."""
    out = np.zeros((total, *forward.shape[1:]), forward.dtype)
    out[seed:seed + len(forward)] = forward[:total - seed]
    if backward is not None and seed > 0:
        out[:seed] = backward[1:seed + 1][::-1]
    return out


def fill_lost(mask, lost):
    """Frames with an empty mask: "open" frees them (whole frame editable), "closed" keeps them
    empty. Returns (mask, lost frame indices)."""
    m = np.asarray(mask) > 0.5
    idx = [int(i) for i in np.flatnonzero(m.reshape(len(m), -1).sum(1) == 0)]
    if lost == "open":
        m = m.copy()
        m[idx] = True
    return m, idx
