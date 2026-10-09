#!/bin/bash
# Point the production endpoint at a worker image + cached model commit, keeping the
# template's other env vars (COS keys etc.) as they are. Secrets are read and written
# in memory only.
#
#   ops/switch-prod.sh <image> <model-commit> [start-cmd-json]
# Rollback example (the pre-image setup):
#   ops/switch-prod.sh runpod/worker-comfyui:5.10.0-base cb4f9269d859d57a87989699a8ce068913e00a02 "$(cat rollback-cmd.json)"
set -euo pipefail
IMAGE=$1 SHA=$2 CMD=${3:-[]}
TEMPLATE=iueurexwjj ENDPOINT=n30uxilrpybvau REPO=GloriaWang23/h3-comfy-runtime
KEY=$(awk -F'"' '/apikey/{print $2}' ~/.runpod/config.toml)

curl -s "https://api.runpod.ai/v2/$ENDPOINT/health" -H "Authorization: Bearer $KEY" | python3 -c '
import sys, json
j = json.load(sys.stdin)["jobs"]
busy = j.get("inProgress", 0) + j.get("inQueue", 0)
print(f"endpoint jobs running/queued: {busy}")
sys.exit("refusing: jobs are running or queued" if busy else 0)'

curl -s "https://rest.runpod.io/v1/templates/$TEMPLATE" -H "Authorization: Bearer $KEY" \
  | IMAGE=$IMAGE SHA=$SHA CMD=$CMD python3 -c '
import sys, json, os
e = json.load(sys.stdin)["env"]
e["H3_MODEL_SHA"] = os.environ["SHA"]
print(json.dumps({"imageName": os.environ["IMAGE"], "dockerStartCmd": json.loads(os.environ["CMD"]), "env": e}))' \
  | curl -s -X PATCH "https://rest.runpod.io/v1/templates/$TEMPLATE" -H "Authorization: Bearer $KEY" \
      -H 'Content-Type: application/json' --data-binary @- \
  | python3 -c 'import sys, json; d = json.load(sys.stdin); print("template:", d["imageName"], "| start cmd:", "custom" if d.get("dockerStartCmd") else "image default", "| H3_MODEL_SHA:", d["env"]["H3_MODEL_SHA"][:10])'

runpodctl serverless update "$ENDPOINT" --model-reference "https://huggingface.co/$REPO:$SHA" \
  | python3 -c 'import sys, json; print("model cache:", json.load(sys.stdin).get("modelReferences"))'
echo "done. Workers redeploy and download the model cache (not billed) before taking jobs."
