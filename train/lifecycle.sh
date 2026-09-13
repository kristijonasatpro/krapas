#!/bin/bash
# Rent a GPU, train, fetch the checkpoints, and make sure the GPU is gone. One command, no leftovers.
#
#   DATA=clips.tgz ./lifecycle.sh
#
# Environment: RUNPOD_API_KEY in ~/.config/runpod/env (see runpod_v2.py), POD_SSH_KEY = the private key whose
# public half RunPod injects, PY = a python with `requests` (runpod_v2 needs nothing else).
#
# The pod is terminated on every exit path — success, failure, Ctrl-C — and the termination is verified
# against the API instead of trusted. That matters: we once had a terminate call return success while the
# pod kept running and billing for ten hours. HARD_STOP_MIN is the second line of defence.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
DATA=${DATA:-$HERE/clips.tgz}              # tar.gz of the clips dir: clips/wav/*.wav + clips/train.jsonl + clips/dev.jsonl
DATA_DIR=${DATA_DIR:-clips}                # must match the audio_path prefix inside the manifests
KEY=${POD_SSH_KEY:-$HOME/.ssh/id_ed25519}
RESULTS=${RESULTS:-$HERE/results}
GPU=${GPU:-NVIDIA A100 80GB PCIe}
IMAGE=${IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}
POD_NAME=${POD_NAME:-tts-finetune}
HARD_STOP_MIN=${HARD_STOP_MIN:-120}
LOG=$RESULTS/lifecycle.log
mkdir -p "$RESULTS"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
DAY=$(date -u +%Y-%m-%dT00:00:00Z)

read -r POD IP PORT <<< "$($PY - <<PY
import sys, os; sys.path.insert(0, "$HERE")
from runpod_v2 import create_pod, wait_ssh
pid = os.environ.get("POD_ID") or create_pod("$POD_NAME", "$GPU", "$IMAGE", disk_gb=80)["id"]
ip, port = wait_ssh(pid)
print(pid, ip, port)
PY
)"
[ -z "${POD:-}" ] && { log "create failed"; exit 1; }
log "pod $POD ssh $IP:$PORT"

kill_pod(){ $PY -c "
import sys, time; sys.path.insert(0, '$HERE')
from runpod_v2 import terminate_verified, api
now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
r = terminate_verified('$POD', '$DAY', now, settle_s=30)
print('verified_dead:', r['verified_dead'])
print('pods left:', api('GET', '/v2/pods')[1])
print('billing:', {k: r[k] for k in r if k != 'verified_dead'})" 2>&1 | tee -a "$LOG"; }
trap 'log "trap: terminating"; kill_pod' EXIT

S="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=25 root@$IP"
get(){ scp -i "$KEY" -P "$PORT" -q -o StrictHostKeyChecking=no root@$IP:"$1" "$2" 2>>"$LOG"; }

$S "mkdir -p /workspace/ft" || { log "ssh failed"; exit 1; }
scp -i "$KEY" -P "$PORT" -q -o StrictHostKeyChecking=no \
    "$DATA" "$HERE/pod_train.sh" "$HERE/train_config.json" "$HERE/data_config.json" \
    root@$IP:/workspace/ft/ || { log "scp failed"; exit 1; }
$S "cd /workspace/ft && tar -xzf $(basename "$DATA") && rm $(basename "$DATA") && ls $DATA_DIR/wav | wc -l && DATA_DIR=$DATA_DIR nohup bash pod_train.sh > pod_train.out 2>&1 &"
log "upload done, training launched"

T0=$(date +%s)
while true; do
  ST=$($S "cat /workspace/ft/STATUS 2>/dev/null" 2>/dev/null)
  LAST=$($S "grep -oE '(step|Step)[^|]*loss[^|]*' /workspace/ft/train.log 2>/dev/null | tail -1" 2>/dev/null | tr '\n' ' ')
  log "status=$ST | $LAST"
  case "$ST" in
    COMPLETE) break;;
    INSTALL_FAILED|TOKENIZE_FAILED|TRAIN_FAILED)
      get /workspace/ft/pod_train.out "$RESULTS/"; get /workspace/ft/train.log "$RESULTS/"
      get "/workspace/ft/tokenize_*.log" "$RESULTS/"; log "FAILED $ST"; exit 1;;
  esac
  if [ $(( $(date +%s) - T0 )) -gt $(( HARD_STOP_MIN * 60 )) ]; then
    get /workspace/ft/train.log "$RESULTS/"; log "HARD STOP after $HARD_STOP_MIN min"; exit 1
  fi
  sleep 120
done

get /workspace/ft/train.log "$RESULTS/"; get /workspace/ft/pod_train.out "$RESULTS/"
get "/workspace/ft/tokenize_*.log" "$RESULTS/"
CKS=$($S "ls -d /workspace/ft/exp/checkpoint-* | sort -t- -k2 -n")
log "checkpoints: $(echo $CKS | tr '\n' ' ')"
# Model weights only (no optimizer state) for every saved step: you pick the one that sounds best by ear,
# and it is usually not the last one.
for CK in $CKS; do
  n=$(basename "$CK"); mkdir -p "$RESULTS/$n"
  for f in config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json \
           special_tokens_map.json vocab.json merges.txt added_tokens.json chat_template.jinja; do
    get "$CK/$f" "$RESULTS/$n/"
  done
done
du -sh "$RESULTS"/* | tee -a "$LOG"
log "FETCH_DONE"
