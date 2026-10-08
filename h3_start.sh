#!/bin/bash
# Link /h3-models to this endpoint's cached model snapshot, then hand over to the stock
# /start.sh (GPU pre-flight, ComfyUI, handler).
#
# H3_MODEL_SHA must equal the commit in the endpoint's model reference: Runpod mounts
# that exact snapshot under /runpod-volume/huggingface-cache. The wait is short on
# purpose, since the container is already billed here: a missing snapshot means the
# cache mount failed, and exiting lets Runpod mark the worker unhealthy.
set -e
REPO="${H3_MODEL_REPO:-GloriaWang23/h3-comfy-runtime}"
: "${H3_MODEL_SHA:?H3_MODEL_SHA must be set to the cached model commit}"
SNAP="/runpod-volume/huggingface-cache/hub/models--${REPO/\//--}/snapshots/${H3_MODEL_SHA}"

for i in $(seq 1 24); do
  [ -d "$SNAP/diffusion_models" ] && break
  echo "h3_start: waiting for cached model ${SNAP} (${i}/24)"
  sleep 5
done
if [ ! -d "$SNAP/diffusion_models" ]; then
  echo "h3_start: cached model not found at ${SNAP}" >&2
  exit 1
fi

ln -sfn "$SNAP" /h3-models
echo "h3_start: models from ${SNAP}, ComfyUI $(cat /comfyui/.h3_version)"
exec /start.sh
