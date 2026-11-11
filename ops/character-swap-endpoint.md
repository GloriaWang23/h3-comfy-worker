# character-swap endpoint (MiniMax H3 character swap)

Created 2026-11-10. Design: `h3-studio/docs/superpowers/specs/2026-11-10-character-swap-design.md`;
experiments: `character-swap/RESULTS.md`; smoke: `character-swap/tools/smoke_swap.mts`.

| item | value |
|---|---|
| endpoint | `character-swap` — `sxzztytrk1vudj` |
| template | `character-swap-template` — `lxtihcgb1l` |
| image | `ghcr.io/gloriawang23/character-swap-worker:sha-de36bc1` (Dockerfile.swap: H3 image sha-0454363 + h3-relay e2a7665 + cswap_nodes incl. CSwapSeedTrack + insightface buffalo_l + swap_prepare.py / swap_faces.py) |
| GPU | RTX 5090, CUDA ≥ 13.0 |
| workers | min 0 / max 1, FlashBoot, queue delay 4 s |
| timeouts | execution 1800 s (studio: prepare 1500 s, swap pass 2× estimate within 300–1800 s), idle 15 s |
| model cache | `https://huggingface.co/GloriaWang23/character-swap-runtime:31ababcd929ae891060282b05a8d86f634dac9c3` (69.6 GB, `sync-swap-models.yml`) |
| template env | `H3_MODEL_REPO`, `H3_MODEL_SHA`, `COS_PREFIX=swap`, `UPLOAD_KEEP_NAMES=1`, COS credentials as moss/vsr |
| studio | Vercel env `RUNPOD_SWAP_ENDPOINT_ID`; `endpoint_config` row `swap`; watchdog row 6 |

Tasks: `input.task = "prepare"` (template video -> manifest v2: people by face identity with head +
upper-body crops, 8 key frames; ~57 s per 10 s template) and `"swap"` (a full workflow).

Prepare v2 (2026-11-11) replaced the SAM3 text-detect + whole-clip tracking prepare, which mixed up
ids on a zoom to close-ups, counted a spurious third person and cropped faceless close-ups. Identity
now comes from face embeddings (insightface SCRFD + ArcFace + genderage on CPU, FaceFusion's approach);
no masks are produced and the studio names the person to replace in the prompt. `cswap_nodes` (SAM3
with cuDNN off) and `build_shot_graph` / `mask_plan` stay in the image for a later optional mask mode.

**Switching image or model commit: `ops/switch-swap.sh <image-tag> [model-commit]`.** A template PATCH
alone left a warm worker on the old image; the script also releases the endpoint and drains workers.

## vlm-qwen38 endpoint (semantic pass of templates)

| item | value |
|---|---|
| endpoint | `vlm-qwen38` — `ujkh2ffe37szir` |
| template | `vlm-qwen38-template` — `7dotaavt6m` |
| image | `ghcr.io/gloriawang23/h3-vlm-worker:sha-69deb78…` (same as vlm-qwen36-vl; worker-vllm) |
| GPU | RTX 5090 (weights 18.7 GB + ~3.7 GB activations), CUDA ≥ 13.0, 0–1 workers, idle 30 s, execution 600 s |
| model cache | `https://huggingface.co/GloriaWang23/vlm-qwen38-27b-abl-nvfp4:5e1fda8f60a1f91403d9afc7478ee57aff42a594` (copy of `lyf/Qwen3.8-27B-Huihui-Abliterated-NVFP4-MTP-VL@3e2cce29`, `sync-vlm38-model.yml`) |
| template env | as vlm-qwen36-vl, with `MODEL_NAME` / `MODEL_REVISION` above, served name `vl38`, `MAX_MODEL_LEN=24576`, `MAX_NUM_SEQS=2`, `VLLM_EXTRA_ARGS` = images only (16), fp8 KV cache, `max_pixels` 1003520, trust remote code |
| studio | Vercel env `RUNPOD_VL38_ENDPOINT_ID`; `endpoint_config` row `vl38`; watchdog row 7; called from `api/swap.ts` only |

First smoke 2026-11-11: cold start ~3 min, 11–12 s per template (8 frames + 2 crops, ~4.5k prompt tokens).

Round 3 (2026-11-11, image sha-b3772aa): optional edit regions come from `CSwapSeedTrack` inside the
swap graph (SAM3 point prompts on the person's clearest face frame, tracked forward and backward; a
person region opens up where tracking loses the person, a head region stays closed), used by the
`box` designation and by head mode through akatz. Prepare accepts `input.start` (10 s from that
offset; the cut clip is uploaded as `cs_source.mp4` and swaps run on it). Smoke: box 367 s (first
model load), akatz head 212 s, two target pictures 204 s, all at 480.
Prompt lesson: naming tattoos or jewelry in the prompt — even as "do not add tattoos" — makes the
model draw them; the studio prompt mentions neither.
