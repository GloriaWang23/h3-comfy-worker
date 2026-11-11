"""People of a character-swap template, from face identity.

Borrowed from FaceFusion's approach (face_selector / face_recognizer): detect faces frame by frame,
embed them (ArcFace) and decide "same person" by embedding similarity, so identity survives camera
moves, close-ups and cuts — a region tracker alone (SAM) loses or swaps ids there. Body parts with
no face are not people. Gender comes from the face classifier, by majority over the person's faces.

Pure helpers work on plain detections {frame, bbox [x0,y0,x1,y1], score, emb (unit vector), sex
'F'|'M', age}; detect_faces() wraps insightface (buffalo_l: SCRFD + ArcFace + genderage) on CPU.
"""
import numpy as np

SIMILARITY = 0.35      # cosine similarity to a cluster's mean face to join it (FaceFusion's reference distance idea)
MIN_SCORE = 0.6        # detector score kept
FRAME_STEP = 3         # analyse every 3rd prepared frame (8 per second at 24 fps)
MIN_FACES = 3          # a cluster smaller than this is noise ...
MIN_SHARE = 0.04       # ... unless it holds this share of all detections (short clips, brief appearances)


def cluster_faces(dets, similarity=SIMILARITY):
    """Greedy clustering by cosine similarity to each cluster's mean embedding; returns clusters
    (lists of detections), most-seen first, without noise clusters."""
    clusters = []
    for d in sorted(dets, key=lambda x: x["frame"]):
        best, best_sim = None, similarity
        for c in clusters:
            mean = c["sum"] / np.linalg.norm(c["sum"])
            sim = float(np.dot(d["emb"], mean))
            if sim > best_sim:
                best, best_sim = c, sim
        if best is None:
            clusters.append({"items": [d], "sum": np.array(d["emb"], np.float64)})
        else:
            best["items"].append(d)
            best["sum"] += d["emb"]
    total = len(dets)
    keep = [c["items"] for c in clusters
            if len(c["items"]) >= MIN_FACES or (total and len(c["items"]) / total >= MIN_SHARE and len(c["items"]) >= 2)]
    return sorted(keep, key=len, reverse=True)


def _crop_box(face, frame_wh):
    """Head and upper body around a face box: one face width each side, half a face above,
    three face heights below; clipped, even size."""
    w, h = frame_wh
    x0, y0, x1, y1 = face
    fw, fh = x1 - x0, y1 - y0
    cx0, cy0 = max(0, int(x0 - fw)), max(0, int(y0 - fh * 0.5))
    cx1, cy1 = min(w, int(x1 + fw)), min(h, int(y1 + fh * 3))
    if (cx1 - cx0) % 2:
        cx1 = cx1 - 1 if cx1 - cx0 > 2 else cx1 + 1
    if (cy1 - cy0) % 2:
        cy1 = cy1 - 1 if cy1 - cy0 > 2 else cy1 + 1
    return [cx0, cy0, min(w, cx1), min(h, cy1)]


def person_card(cluster, frame_wh):
    """One person from a face cluster: gender by majority, the frame with the largest clear face,
    a head + upper-body crop box on that frame."""
    sexes = [d["sex"] for d in cluster]
    female = sexes.count("F")
    gender = "female" if female * 2 >= len(sexes) else "male"
    best = max(cluster, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) * (d["score"] ** 4))
    frames = [d["frame"] for d in cluster]
    return {
        "gender": gender,
        "gender_conf": round(max(female, len(sexes) - female) / len(sexes), 2),
        "age": int(np.median([d.get("age", 0) for d in cluster])),
        "faces": len(cluster),
        "first_frame": int(min(frames)),
        "last_frame": int(max(frames)),
        "frame": int(best["frame"]),
        "face_box": [int(v) for v in best["bbox"]],
        "crop_box": _crop_box(best["bbox"], frame_wh),
    }


def people_from_faces(dets, frame_wh):
    """Detections -> person cards p1, p2, ... (most-seen first)."""
    return [{"id": f"p{i + 1}", **person_card(c, frame_wh)} for i, c in enumerate(cluster_faces(dets))]


_APP = None


def detect_faces(frames_rgb, step=FRAME_STEP, min_score=MIN_SCORE):
    """Faces on every `step`-th frame ([N,H,W,3] uint8 RGB) -> detections for cluster_faces."""
    global _APP
    if _APP is None:
        import os
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name="buffalo_l", root=os.environ.get("INSIGHTFACE_HOME", "/opt/insightface"),
                            providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
    dets = []
    for i in range(0, len(frames_rgb), step):
        for f in _APP.get(np.ascontiguousarray(frames_rgb[i][:, :, ::-1])):
            if f.det_score < min_score:
                continue
            dets.append({"frame": i, "bbox": [float(v) for v in f.bbox], "score": float(f.det_score),
                         "emb": np.asarray(f.normed_embedding, np.float32), "sex": f.sex, "age": int(f.age)})
    return dets
