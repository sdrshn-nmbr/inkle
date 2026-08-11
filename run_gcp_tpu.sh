#!/usr/bin/env bash

set -euo pipefail

PROJECT="${INKLING_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${INKLING_GCP_ZONE:-us-central1-a}"
TPU_NAME="${INKLING_TPU_NAME:-inkling-v5e-16}"
ACCELERATOR_TYPE="${INKLING_ACCELERATOR_TYPE:-v5litepod-16}"
RESULTS_DIRECTORY="${INKLING_RESULTS_DIRECTORY:-/tmp/inkling-v5e-results-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORACLE_DIRECTORY="${INKLING_ORACLE_DIRECTORY:-/tmp/inkling-cpu-reference-full}"
EXPERT_CACHE_DIRECTORY="${INKLING_EXPERT_CACHE_DIRECTORY:-/tmp/inkling-expert-cache}"
REUSE_TPU="${INKLING_REUSE_TPU:-0}"
RUN_QWEN_CONTROL="${INKLING_RUN_QWEN_CONTROL:-1}"
REUSE_STAGED_DATA="${INKLING_REUSE_STAGED_DATA:-0}"
PREFIX_LAYERS="${INKLING_PREFIX_LAYERS:-6}"
DIAGNOSE_SPLIT="${INKLING_DIAGNOSE_SPLIT:-0}"
STAGING_DIRECTORY="$(mktemp -d -t inkling-v5e.XXXXXX)"
MANAGE_TPU=0
DELETE_TPU_ON_EXIT=0
ARTIFACTS_COLLECTED=0

gcloud_tpu() {
  gcloud compute tpus tpu-vm "$@" --project="$PROJECT" --zone="$ZONE"
}

remote_command() {
  local worker=$1
  local command=$2
  gcloud_tpu ssh "$TPU_NAME" --worker="$worker" --command="$command" --quiet
}

stop_remote_processes() {
  if [[ "$MANAGE_TPU" != "1" ]]; then
    return
  fi
  for worker in 0 1 2 3; do
    remote_command "$worker" \
      'if test -f "$HOME/inkle-results/active.pid"; then kill -TERM "$(cat "$HOME/inkle-results/active.pid")" 2>/dev/null || true; fi; pkill -TERM -f "[s]gl_jax.launch_server|[t]pu_inkling_validation.py" 2>/dev/null || true' \
      >/dev/null 2>&1 || true
  done
}

collect_artifacts() {
  if [[ "$MANAGE_TPU" != "1" || "$ARTIFACTS_COLLECTED" == "1" ]]; then
    return
  fi
  ARTIFACTS_COLLECTED=1
  if [[ "$DIAGNOSE_SPLIT" == "1" ]]; then
    echo "INKLING_DIAGNOSTIC_ARTIFACT_COPY_SKIPPED"
    return
  fi
  mkdir -p "$RESULTS_DIRECTORY"
  gcloud_tpu describe "$TPU_NAME" --format=json \
    >"$RESULTS_DIRECTORY/tpu-description.json" 2>/dev/null || true
  for worker in 0 1 2 3; do
    mkdir -p "$RESULTS_DIRECTORY/worker-$worker"
    if remote_command "$worker" 'test -d "$HOME/inkle-results"' >/dev/null 2>&1; then
      gcloud_tpu scp "$TPU_NAME:~/inkle-results" \
        "$RESULTS_DIRECTORY/worker-$worker/" \
        --worker="$worker" \
        --recurse \
        --quiet || true
    fi
  done
}

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  stop_remote_processes
  collect_artifacts
  if [[ "$MANAGE_TPU" == "1" ]]; then
    if [[ "$DELETE_TPU_ON_EXIT" == "1" && "$exit_status" == "0" ]]; then
      gcloud_tpu delete "$TPU_NAME" --quiet || true
      echo "INKLING_TPU_DELETED name=$TPU_NAME zone=$ZONE"
    elif [[ "$exit_status" == "0" ]]; then
      echo "INKLING_TPU_PRESERVED_AFTER_DIAGNOSTIC name=$TPU_NAME zone=$ZONE"
    else
      echo "INKLING_TPU_PRESERVED_AFTER_FAILURE name=$TPU_NAME zone=$ZONE" >&2
    fi
  fi
  rm -rf -- "$STAGING_DIRECTORY"
  exit "$exit_status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
  echo "INKLING_GCP_PROJECT_REQUIRED" >&2
  exit 1
fi
if [[ ! -d "$ORACLE_DIRECTORY" ]]; then
  echo "INKLING_ORACLE_DIRECTORY_MISSING path=$ORACLE_DIRECTORY" >&2
  exit 1
fi
if [[ ! -d "$EXPERT_CACHE_DIRECTORY" ]]; then
  echo "INKLING_EXPERT_CACHE_DIRECTORY_MISSING path=$EXPERT_CACHE_DIRECTORY" >&2
  exit 1
fi
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  echo "INKLING_GCLOUD_AUTH_REQUIRED run='gcloud auth login'" >&2
  exit 1
fi

if gcloud_tpu describe "$TPU_NAME" >/dev/null 2>&1; then
  if [[ "$REUSE_TPU" != "1" ]]; then
    echo "INKLING_TPU_ALREADY_EXISTS name=$TPU_NAME zone=$ZONE" >&2
    exit 1
  fi
  echo "INKLING_TPU_REUSED name=$TPU_NAME zone=$ZONE"
  MANAGE_TPU=1
else
  gcloud_tpu create "$TPU_NAME" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --version=v2-tpuv5-litepod \
    --quiet
  MANAGE_TPU=1
fi

while true; do
  state="$(gcloud_tpu describe "$TPU_NAME" --format='value(state)')"
  echo "INKLING_TPU_STATE name=$TPU_NAME state=$state"
  if [[ "$state" == "READY" ]]; then
    break
  fi
  if [[ "$state" == "TERMINATED" || "$state" == "PREEMPTED" ]]; then
    echo "INKLING_TPU_UNUSABLE name=$TPU_NAME state=$state" >&2
    exit 1
  fi
  sleep 30
done

CODE_ARCHIVE="$STAGING_DIRECTORY/inkle-source.tar.gz"
ORACLE_ARCHIVE="$STAGING_DIRECTORY/inkling-oracle.tar.gz"
tar \
  --exclude=.git \
  --exclude=.venv \
  --exclude=.pytest_cache \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf "$CODE_ARCHIVE" \
  -C "$SOURCE_DIRECTORY" \
  .
if [[ "$REUSE_STAGED_DATA" != "1" ]]; then
  tar -czf "$ORACLE_ARCHIVE" -C "$(dirname "$ORACLE_DIRECTORY")" "$(basename "$ORACLE_DIRECTORY")"
fi

for worker in 0 1 2 3; do
  gcloud_tpu scp "$CODE_ARCHIVE" "$TPU_NAME:~/inkle-source.tar.gz" \
    --worker="$worker" --quiet
  if [[ "$REUSE_STAGED_DATA" == "1" ]]; then
    if ! remote_command "$worker" \
      'test -d /tmp/inkling-cpu-reference-full; test "$(ls /tmp/inkling-expert-cache/layer_0[2-5]_expert_*.npz 2>/dev/null | wc -l)" -ge 256'; then
      echo "INKLING_REUSED_DATA_MISSING worker=$worker" >&2
      exit 1
    fi
  else
    CACHE_ARCHIVE="$STAGING_DIRECTORY/expert-cache-worker-$worker.tar"
    expert_start=$((worker * 64))
    expert_stop=$((expert_start + 64))
    cache_files=()
    for layer in 02 03 04 05; do
      for ((expert_id = expert_start; expert_id < expert_stop; expert_id++)); do
        printf -v cache_name 'layer_%s_expert_%03d.npz' "$layer" "$expert_id"
        if [[ -f "$EXPERT_CACHE_DIRECTORY/$cache_name" ]]; then
          cache_files+=("$cache_name")
        fi
      done
    done
    if [[ "${#cache_files[@]}" -gt 0 ]]; then
      tar -cf "$CACHE_ARCHIVE" -C "$EXPERT_CACHE_DIRECTORY" "${cache_files[@]}"
    else
      tar -cf "$CACHE_ARCHIVE" --files-from /dev/null
    fi
    gcloud_tpu scp "$ORACLE_ARCHIVE" "$TPU_NAME:~/inkling-oracle.tar.gz" \
      --worker="$worker" --quiet
    gcloud_tpu scp "$CACHE_ARCHIVE" "$TPU_NAME:~/inkling-expert-cache.tar" \
      --worker="$worker" --quiet
  fi
done

setup_pids=()
for worker in 0 1 2 3; do
  if [[ "$REUSE_STAGED_DATA" == "1" ]]; then
    setup_command='set -euo pipefail; if test -e "$HOME/inkle"; then mv "$HOME/inkle" "$HOME/inkle.previous.$(date +%s)"; fi; mkdir -p "$HOME/inkle" "$HOME/inkle-results"; tar -xzf "$HOME/inkle-source.tar.gz" -C "$HOME/inkle"; cd "$HOME/inkle"; "$HOME/.local/bin/uv" sync --extra tpu'
  else
    setup_command='set -euo pipefail; if test -e "$HOME/inkle"; then mv "$HOME/inkle" "$HOME/inkle.previous.$(date +%s)"; fi; mkdir -p "$HOME/inkle" "$HOME/inkle-results" /tmp/inkling-expert-cache; tar -xzf "$HOME/inkle-source.tar.gz" -C "$HOME/inkle"; tar -xzf "$HOME/inkling-oracle.tar.gz" -C /tmp; tar -xf "$HOME/inkling-expert-cache.tar" -C /tmp/inkling-expert-cache; curl -LsSf https://astral.sh/uv/install.sh | sh; cd "$HOME/inkle"; "$HOME/.local/bin/uv" sync --extra tpu'
  fi
  remote_command "$worker" "$setup_command" >"$STAGING_DIRECTORY/setup-worker-$worker.log" 2>&1 &
  setup_pids+=("$!")
done
for worker in 0 1 2 3; do
  if ! wait "${setup_pids[$worker]}"; then
    cat "$STAGING_DIRECTORY/setup-worker-$worker.log" >&2
    echo "INKLING_WORKER_SETUP_FAILED worker=$worker" >&2
    exit 1
  fi
done
echo "INKLING_WORKERS_READY count=4"

RANK_ZERO_IP="$(remote_command 0 "hostname -I | awk '{print \$1}'" | tail -n 1 | tr -d '[:space:]')"
if [[ -z "$RANK_ZERO_IP" ]]; then
  echo "INKLING_RANK_ZERO_IP_MISSING" >&2
  exit 1
fi

launch_mode() {
  local mode=$1
  local coordinator=$2
  for worker in 0 1 2 3; do
    remote_command "$worker" \
      "rm -f \"\$HOME/inkle-results/${mode}-worker-${worker}.status\"; nohup env INKLING_WORKER_STATUS=\"\$HOME/inkle-results/${mode}-worker-${worker}.status\" INKLING_PREFIX_LAYERS='$PREFIX_LAYERS' INKLING_DIAGNOSE_SPLIT='$DIAGNOSE_SPLIT' bash \"\$HOME/inkle/tpu_worker_entrypoint.sh\" '$mode' '$worker' '$coordinator' >\"\$HOME/inkle-results/${mode}-worker-${worker}.log\" 2>&1 < /dev/null & echo \$! >\"\$HOME/inkle-results/active.pid\""
  done
}

if [[ "$RUN_QWEN_CONTROL" == "1" ]]; then
  launch_mode qwen-control "$RANK_ZERO_IP:10011"
  qwen_ready=0
  for _ in $(seq 1 360); do
    if remote_command 0 'curl -fsS http://127.0.0.1:30000/health >/dev/null' >/dev/null 2>&1; then
      qwen_ready=1
      break
    fi
    for worker in 0 1 2 3; do
      if remote_command "$worker" "test -f \"\$HOME/inkle-results/qwen-control-worker-${worker}.status\"" >/dev/null 2>&1; then
        echo "INKLING_QWEN_CONTROL_EXITED_EARLY worker=$worker" >&2
        exit 1
      fi
    done
    sleep 20
  done
  if [[ "$qwen_ready" != "1" ]]; then
    echo "INKLING_QWEN_CONTROL_NOT_READY" >&2
    exit 1
  fi

  QWEN_RESPONSE="$(remote_command 0 \
    'curl -fsS http://127.0.0.1:30000/generate -H "Content-Type: application/json" -d '\''{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}'\'' | tee "$HOME/inkle-results/qwen-control-response.json"')"
  if [[ "$QWEN_RESPONSE" != *Paris* && "$QWEN_RESPONSE" != *paris* ]]; then
    echo "INKLING_QWEN_CONTROL_INCOHERENT response=$QWEN_RESPONSE" >&2
    exit 1
  fi
  echo "INKLING_QWEN_CONTROL_ACCEPTED response=$QWEN_RESPONSE"
  stop_remote_processes
else
  echo "INKLING_QWEN_CONTROL_SKIPPED reason=previously-accepted"
fi

launch_mode inkling-prefix "$RANK_ZERO_IP:10021"
while true; do
  all_done=1
  for worker in 0 1 2 3; do
    status_path="\$HOME/inkle-results/inkling-prefix-worker-${worker}.status"
    if remote_command "$worker" "test -f \"$status_path\"" >/dev/null 2>&1; then
      status="$(remote_command "$worker" "cat \"$status_path\"")"
      if [[ "$status" != "0" ]]; then
        echo "INKLING_PREFIX_WORKER_FAILED worker=$worker status=$status" >&2
        exit 1
      fi
    else
      all_done=0
    fi
  done
  if [[ "$all_done" == "1" ]]; then
    break
  fi
  sleep 30
done

if ! remote_command 0 'test -s "$HOME/inkle-results/inkling-prefix-result.json"'; then
  echo "INKLING_PREFIX_RESULT_MISSING" >&2
  exit 1
fi
echo "INKLING_PREFIX_ACCEPTED"

if [[ "$DIAGNOSE_SPLIT" == "1" ]]; then
  echo "INKLING_DIAGNOSTIC_ACCEPTED"
  exit 0
fi

DELETE_TPU_ON_EXIT=1
echo "INKLING_RESULTS_DIRECTORY path=$RESULTS_DIRECTORY"
