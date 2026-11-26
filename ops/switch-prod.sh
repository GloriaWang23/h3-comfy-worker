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

# Template image + env through the v2 API (rest.runpod.io does not resolve everywhere). The whole
# env is read and written back in memory; the result is read again and compared.
KEY=$KEY IMAGE=$IMAGE SHA=$SHA CMD=$CMD TEMPLATE=$TEMPLATE python3 - <<'PY'
import json, os, urllib.error, urllib.request
url = f"https://api.runpod.io/v2/templates/{os.environ['TEMPLATE']}"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def call(method="GET", body=None):
    r = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None, headers={
        "Authorization": "Bearer " + os.environ["KEY"], "Content-Type": "application/json", "User-Agent": "curl/8"})
    try:
        return json.loads(op.open(r, timeout=60).read() or b"{}")
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")
        for v in (body or {}).get("env", {}).values():  # never print env values
            if len(str(v)) > 8:
                text = text.replace(str(v), "<redacted>")
        raise SystemExit(f"{method} template: HTTP {e.code} {text[:500]}")
old = call()
env = {**old["env"], "H3_MODEL_SHA": os.environ["SHA"]}
body = {"image": os.environ["IMAGE"], "env": env}
cmd = json.loads(os.environ["CMD"])
# v2 `args` is a JSON string ("{}" = the image's own command).
if cmd:
    body["args"] = json.dumps({"cmd": cmd})
elif old.get("args") not in (None, "", "{}"):
    body["args"] = "{}"
call("PATCH", body)
new = call()
assert new["image"] == os.environ["IMAGE"], f"image is {new['image']}"
assert new["env"] == env, "env differs after the update: " + ", ".join(k for k in {**env, **new["env"]} if env.get(k) != new["env"].get(k))
for k in ("disk", "ports", "serverless", "registry", "mounts"):
    assert new.get(k) == old.get(k), f"{k} changed"
print("template:", new["image"], "| args:", new.get("args") or "image default", "| H3_MODEL_SHA:", new["env"]["H3_MODEL_SHA"][:10], "| env keys:", len(new["env"]))
PY

runpodctl serverless update "$ENDPOINT" --model-reference "https://huggingface.co/$REPO:$SHA" \
  | python3 -c 'import sys, json; print("model cache:", json.load(sys.stdin).get("modelReferences"))'
echo "done. Workers redeploy and download the model cache (not billed) before taking jobs."

# Verify instead of assuming: an idle worker on the old image can linger (workers-max 0 did not
# always evict it, 2026-11-26). Poll the worker logs every 30 s (model-status is rate limited per
# hour, so it is read once at the end, for information only) for up to 20 minutes.
TAG=${IMAGE##*:}
for i in $(seq 1 40); do
  # the handler prints "image=sha-<full sha>" at start (images built after 2026-11-26)
  if runpodctl serverless logs "$ENDPOINT" --since 30m 2>/dev/null | grep -q "image=$TAG"; then
    echo "verified: a worker pulled image $TAG"
    runpodctl serverless model-status "$ENDPOINT" 2>/dev/null | head -c 600; echo
    exit 0
  fi
  sleep 30
done
echo "WARNING: no worker log mentions image $TAG after 20 min. Workers start only when a job arrives and may wait for a GPU or the model cache; submit a small job and grep the logs for \"image=$TAG\" before trusting the switch." >&2
exit 2
