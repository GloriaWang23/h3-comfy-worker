#!/bin/bash
# Point the character-swap endpoint at a worker image (and model commit) and make sure no worker
# keeps running the old image: patch the template, release the endpoint (serverless update), then
# drain workers (max 0 -> wait -> max 1). A template PATCH alone left a warm worker on the old
# image (2026-11-10). Refuses while jobs are running or queued. Secrets stay in memory.
#
#   ops/switch-swap.sh <image-tag> [model-commit]
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
TAG=$1 SHA=${2:-31ababcd929ae891060282b05a8d86f634dac9c3}
TEMPLATE=lxtihcgb1l ENDPOINT=sxzztytrk1vudj REPO=GloriaWang23/character-swap-runtime
KEY=$(awk -F'"' '/apikey/{print $2}' ~/.runpod/config.toml)
health() { curl -s "https://api.runpod.ai/v2/$ENDPOINT/health" -H "Authorization: Bearer $KEY"; }

health | python3 -c '
import sys, json
j = json.load(sys.stdin)["jobs"]
busy = j.get("inProgress", 0) + j.get("inQueue", 0)
sys.exit("refusing: jobs are running or queued" if busy else 0)'

curl -s "https://rest.runpod.io/v1/templates/$TEMPLATE" -H "Authorization: Bearer $KEY" \
  | IMAGE="ghcr.io/gloriawang23/character-swap-worker:$TAG" SHA=$SHA python3 -c '
import sys, json, os
e = json.load(sys.stdin)["env"]
e["H3_MODEL_SHA"] = os.environ["SHA"]
print(json.dumps({"imageName": os.environ["IMAGE"], "env": e}))' \
  | curl -s -X PATCH "https://rest.runpod.io/v1/templates/$TEMPLATE" -H "Authorization: Bearer $KEY" \
      -H 'Content-Type: application/json' --data-binary @- \
  | python3 -c 'import sys, json; d = json.load(sys.stdin); print("template:", d["imageName"], "| H3_MODEL_SHA:", d["env"]["H3_MODEL_SHA"][:10])'

runpodctl serverless update "$ENDPOINT" --model-reference "https://huggingface.co/$REPO:$SHA" 2>/dev/null \
  | python3 -c 'import sys, json; print("model cache:", json.load(sys.stdin).get("modelReferences"))'
runpodctl serverless update "$ENDPOINT" --workers-max 0 >/dev/null 2>&1
for i in $(seq 1 30); do
  n=$(health | python3 -c 'import sys, json; w = json.load(sys.stdin)["workers"]; print(sum(v for k, v in w.items() if k != "unhealthy"))')
  [ "$n" = "0" ] && break
  sleep 10
done
runpodctl serverless update "$ENDPOINT" --workers-max 1 >/dev/null 2>&1
sleep 60
echo "done: workers drained, max 1; the next job cold-starts on the new image."
