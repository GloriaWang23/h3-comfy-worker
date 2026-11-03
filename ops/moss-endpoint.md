# moss-image endpoint (MOSS-Image image generation)

Created 2026-11-03. Design: `h3-studio/docs/superpowers/specs/2026-11-03-image-moss-design.md`.

| item | value |
|---|---|
| endpoint | `moss-image` — `966x5j6ruo9f2m` |
| template | `moss-image-template` — `c43hf6faum` |
| image | `ghcr.io/gloriawang23/h3-comfy-worker:sha-0454363` (same as h3-hf; MOSS nodes are built into ComfyUI v0.43.1) |
| GPU | RTX 5090 (pool `ADA_32_PRO`), CUDA ≥ 13.0 |
| workers | min 0 / max 1, FlashBoot on, scale by queue delay 4 s |
| timeouts | execution 180 s (each job: 3× estimate within 60–180 s), idle 15 s |
| model cache | `https://huggingface.co/GloriaWang23/moss-image-runtime:fa59edf063a12ad5545fa3b882151ed8c160c9cc` (44.8 GB, `sync-moss-models.yml`) |
| template env | `H3_MODEL_REPO=GloriaWang23/moss-image-runtime`, `H3_MODEL_SHA=<model commit>`, `COS_PREFIX=images/moss`, COS credentials as the vsr template |
| studio | Vercel env `RUNPOD_MOSS_ENDPOINT_ID`; `endpoint_config` row `moss` |

Moving to a new model commit (after re-running `sync-moss-models.yml`): set the template's
`H3_MODEL_SHA` to the new commit (REST `PATCH /v1/templates/c43hf6faum`), then
`runpodctl serverless update 966x5j6ruo9f2m --model-reference "https://huggingface.co/GloriaWang23/moss-image-runtime:<sha>"`.
