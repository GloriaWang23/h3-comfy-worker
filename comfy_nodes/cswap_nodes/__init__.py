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
    sdp = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        with torch.backends.cudnn.flags(enabled=False):
            yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(sdp)

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


NODE_CLASS_MAPPINGS = {"CSwapSAM3Detect": CSwapSAM3Detect, "CSwapSAM3VideoTrack": CSwapSAM3VideoTrack}
NODE_DISPLAY_NAME_MAPPINGS = {"CSwapSAM3Detect": "SAM3 Detect (cuDNN off)", "CSwapSAM3VideoTrack": "SAM3 Video Track (cuDNN off)"}
