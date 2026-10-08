# h3-comfy-worker

RunPod Serverless worker image for H3 Studio (MiniMax H3 video generation through ComfyUI).

Built `FROM runpod/worker-comfyui:5.10.0-base`, following that project's customization guide, with:

- **torch 2.11.0+cu130** (comfy-kitchen's CUDA int8 kernels need CUDA 13; the endpoint only uses hosts with CUDA >= 13.0)
- **ComfyUI v0.39.2** (the base image ships 0.34.0, which has no MiniMax H3 nodes)
- **`h3_handler.py`**: downloads `input.media` URLs into ComfyUI's input folder, uploads outputs to Tencent COS, and interrupts ComfyUI when the Runpod job is cancelled
- **`h3_start.sh`**: links `/h3-models` to the endpoint's cached Hugging Face snapshot, then runs the stock `/start.sh`

Model weights are not in the image. The endpoint's cached model (`GloriaWang23/h3-comfy-runtime`) provides them under `/runpod-volume/huggingface-cache`.

## Image

`ghcr.io/gloriawang23/h3-comfy-worker`, built by `.github/workflows/build.yml`:

| Trigger | Tags |
|---|---|
| push to `main` | `main`, `sha-<short>` |
| tag `vX.Y.Z` | `vX.Y.Z` (pin the RunPod template to these) |

## Worker environment

| Variable | Purpose |
|---|---|
| `H3_MODEL_SHA` | commit of the cached model; must match the endpoint's model reference |
| `H3_MODEL_REPO` | optional, defaults to `GloriaWang23/h3-comfy-runtime` |
| `COS_BUCKET`, `COS_REGION`, `COS_DOMAIN`, `COS_PREFIX`, `COS_SECRET_ID`, `COS_SECRET_KEY` | output upload |
| `COMFY_LOG_LEVEL`, `REFRESH_WORKER` | stock worker-comfyui settings |

Secrets live only in the RunPod template, never in this repository.
