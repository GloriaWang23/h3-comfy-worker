# character-swap endpoint (MiniMax H3 character swap)

Created 2026-11-10. Design: `h3-studio/docs/superpowers/specs/2026-11-10-character-swap-design.md`;
experiments: `character-swap/RESULTS.md`; smoke: `character-swap/tools/smoke_swap.mts`.

| item | value |
|---|---|
| endpoint | `character-swap` — `sxzztytrk1vudj` |
| template | `character-swap-template` — `lxtihcgb1l` |
| image | `ghcr.io/gloriawang23/character-swap-worker:sha-aebe5f8` (Dockerfile.swap: H3 image sha-0454363 + h3-relay e2a7665 + cswap_nodes + swap_prepare.py) |
| GPU | RTX 5090, CUDA ≥ 13.0 |
| workers | min 0 / max 1, FlashBoot, queue delay 4 s |
| timeouts | execution 1800 s (studio: prepare 1500 s, swap pass 2× estimate within 300–1800 s), idle 15 s |
| model cache | `https://huggingface.co/GloriaWang23/character-swap-runtime:31ababcd929ae891060282b05a8d86f634dac9c3` (69.6 GB, `sync-swap-models.yml`) |
| template env | `H3_MODEL_REPO`, `H3_MODEL_SHA`, `COS_PREFIX=swap`, `UPLOAD_KEEP_NAMES=1`, COS credentials as moss/vsr |
| studio | Vercel env `RUNPOD_SWAP_ENDPOINT_ID`; `endpoint_config` row `swap`; watchdog row 6 |

Tasks: `input.task = "prepare"` (template video → manifest of shots / SAM3 people / crops / masks) and
`"swap"` (a full workflow; optional `mask_plan` joined into one mask file before the run).

**Switching image or model commit: `ops/switch-swap.sh <image-tag> [model-commit]`.** A template PATCH
alone left a warm worker on the old image; the script also releases the endpoint and drains workers.

Known: SAM3 text detection fails with "cuDNN Frontend error: No valid execution plans" on this image;
`cswap_nodes` run SAM3 with cuDNN off (convolutions, SDPA and ComfyUI's SDPA priority list), which makes
prepare slower (147–458 s per 8–10 s template).
