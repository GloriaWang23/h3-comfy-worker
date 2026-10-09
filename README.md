# h3-comfy-worker

RunPod Serverless worker image for H3 Studio (MiniMax H3 video generation through ComfyUI).

Built `FROM runpod/worker-comfyui:5.10.0-base`, following that project's customization guide, with:

- **torch 2.11.0+cu130** (comfy-kitchen's CUDA int8 kernels need CUDA 13; the endpoint only uses hosts with CUDA >= 13.0)
- **ComfyUI v0.39.2** (the base image ships 0.34.0, which has no MiniMax H3 nodes)
- **`h3_handler.py`**, wrapped around the stock handler:
  - downloads `input.media` URLs into ComfyUI's input folder (retries; a 4xx or oversize file fails at once)
  - checks that every model file the loaders name exists before queueing
  - uploads outputs to Tencent COS (retries with backoff; a lost upload throws away a finished, billed job)
  - interrupts ComfyUI when the Runpod job is cancelled
  - returns `refresh_worker` when ComfyUI stops responding, so Runpod replaces the worker
  - deletes the job's inputs and outputs afterwards and returns stage `timings` (`media_s`, `comfy_s`, `upload_s`, `total_s`, `worker_job`)
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

## Models

The cached Hugging Face repo holds only weights (official H3 defaults, Lightning LoRAs,
10Eros, FastH3, community LoRAs). Add or remove files with the `sync models` workflow
(`.github/workflows/sync-models.yml`, needs the repository secret `HF_TOKEN`); it prints the
new commit. Endpoints pin a commit, so nothing changes until production is switched to it.
A new commit makes each host download the cache again before its first worker starts
(not billed, about 10 minutes for ~108 GB).

## Switch or roll back production

```bash
ops/switch-prod.sh <image> <model-commit> [start-cmd-json]
# current
ops/switch-prod.sh ghcr.io/gloriawang23/h3-comfy-worker:sha-7d41bbb cae2ddd279b97ab3e229bce284ae561849538bdd
# back to the pre-image setup (stock image + boot script from the model repo)
ops/switch-prod.sh runpod/worker-comfyui:5.10.0-base cb4f9269d859d57a87989699a8ce068913e00a02 "$(cat ops/rollback-cmd.json)"
```

The script refuses while jobs are running or queued, keeps the template's other env vars,
and moves the template's `H3_MODEL_SHA` and the endpoint's model reference together.

## Tests

`python3 tests/test_h3_handler.py` (stock handler, boto3 and requests stubbed).
