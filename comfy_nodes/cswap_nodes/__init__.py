"""character-swap helper nodes: ComfyUI's SAM3 detect / video track without cuDNN.

On the character-swap worker image (torch 2.11 cu130, RTX 5090) SAM3's text-prompted detection
fails with "cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built". cuDNN is
used both by convolutions and by scaled_dot_product_attention's cuDNN backend; these wrappers turn
both off around the stock nodes and restore them afterwards, so the H3 graphs on the same worker
keep cuDNN.
"""
import contextlib

import torch


@contextlib.contextmanager
def no_cudnn():
    """cuDNN off for convolutions and attention. ComfyUI's attention wraps every SDPA call in
    sdpa_kernel(comfy.ops.SDPA_BACKEND_PRIORITY, set_priority=True), which lists CUDNN_ATTENTION and
    overrides the global cudnn-sdp switch, so that list is filtered for the duration too."""
    import comfy.ops
    prio = getattr(comfy.ops, "SDPA_BACKEND_PRIORITY", None)
    saved_prio = list(prio) if prio is not None else None
    sdp = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(False)
    if prio is not None:
        from torch.nn.attention import SDPBackend
        prio[:] = [b for b in prio if b != SDPBackend.CUDNN_ATTENTION]
    try:
        with torch.backends.cudnn.flags(enabled=False):
            yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(sdp)
        if saved_prio is not None:
            prio[:] = saved_prio

from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack


class CSwapSAM3Detect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "image": ("IMAGE",), "conditioning": ("CONDITIONING",),
                             "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5}),
                             "individual_masks": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "character-swap"

    def run(self, model, image, conditioning, threshold, refine_iterations, individual_masks):
        with no_cudnn():
            out = SAM3_Detect.execute(model=model, image=image, conditioning=conditioning, threshold=threshold,
                                      refine_iterations=refine_iterations, individual_masks=individual_masks)
        return (out[0],)


class CSwapSAM3VideoTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "model": ("MODEL",), "initial_mask": ("MASK",), "conditioning": ("CONDITIONING",),
                             "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "max_objects": ("INT", {"default": 3, "min": 0, "max": 64}),
                             "detect_interval": ("INT", {"default": 1, "min": 1})}}
    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    FUNCTION = "run"
    CATEGORY = "character-swap"

    def run(self, images, model, initial_mask, conditioning, detection_threshold, max_objects, detect_interval):
        with no_cudnn():
            out = SAM3_VideoTrack.execute(images=images, model=model, initial_mask=initial_mask, conditioning=conditioning,
                                          detection_threshold=detection_threshold, max_objects=max_objects,
                                          detect_interval=detect_interval)
        return (out[0],)


class CSwapSeedTrack:
    """Region mask of one known person: SAM3 point prompts on the seed frame (from the face box), then
    tracking forward and backward from it. No text re-detection (that mixed up ids). The mask is grown
    by `expansion_px` and frames where the person was lost are handled by `lost`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "model": ("MODEL",),
                             "frame": ("INT", {"default": 0, "min": 0}),
                             "face_box": ("STRING", {"default": "[0.4, 0.1, 0.6, 0.3]", "tooltip": "normalised [x0, y0, x1, y1] of the face on the seed frame"}),
                             "kind": (["person", "head"],),
                             "expansion_px": ("INT", {"default": 192, "min": 0, "max": 256}),
                             "lost": (["open", "closed"],)}}
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "run"
    CATEGORY = "character-swap"

    def _track(self, images, model, seed_mask):
        from comfy_extras.nodes_sam3 import SAM3_TrackToMask
        track = SAM3_VideoTrack.execute(images=images, model=model, initial_mask=seed_mask, detection_threshold=0.5,
                                        max_objects=1, detect_interval=1)[0]
        return SAM3_TrackToMask.execute(track_data=track, object_indices="0")[0]

    def run(self, images, model, frame, face_box, kind, expansion_px, lost):
        import json
        import numpy as np
        import torch.nn.functional as F
        from .masks import fill_lost, seed_points, stitch
        n, h, w = images.shape[:3]
        frame = min(frame, n - 1)
        pos, neg = seed_points(json.loads(face_box), kind, (w, h))

        def work():
            seed = SAM3_Detect.execute(model=model, image=images[frame:frame + 1], positive_coords=json.dumps(pos),
                                       negative_coords=json.dumps(neg) if neg else None, threshold=0.35,
                                       refine_iterations=2, individual_masks=True)[0][:1]
            fwd = self._track(images[frame:], model, seed)
            bwd = self._track(torch.flip(images[:frame + 1], dims=[0]), model, seed) if frame > 0 else None
            return fwd, bwd

        try:
            fwd, bwd = work()
        except RuntimeError as e:      # the cuDNN frontend error seen with SAM3 text detection on this image
            if "cudnn" not in str(e).lower():
                raise
            with no_cudnn():
                fwd, bwd = work()
        mask, lost_frames = fill_lost(stitch(fwd.float().cpu().numpy(), None if bwd is None else bwd.float().cpu().numpy(), frame, n), lost)
        out = torch.from_numpy(mask.astype(np.float32))
        if expansion_px:
            k = 2 * int(expansion_px) + 1
            out = F.max_pool2d(out[:, None], kernel_size=k, stride=1, padding=int(expansion_px))[:, 0]
        report = json.dumps({"seed_frame": frame, "kind": kind, "lost_frames": len(lost_frames), "lost": lost,
                             "first_lost": lost_frames[:10]})
        return {"result": (out, report), "ui": {"text": [report]}}


NODE_CLASS_MAPPINGS = {"CSwapSAM3Detect": CSwapSAM3Detect, "CSwapSAM3VideoTrack": CSwapSAM3VideoTrack, "CSwapSeedTrack": CSwapSeedTrack}
NODE_DISPLAY_NAME_MAPPINGS = {"CSwapSAM3Detect": "SAM3 Detect (cuDNN off)", "CSwapSAM3VideoTrack": "SAM3 Video Track (cuDNN off)", "CSwapSeedTrack": "Seed Track Mask (face seed, both directions)"}
