#!/usr/bin/env bash

set -euo pipefail

MODE="${1:?mode is required}"
NODE_RANK="${2:?node rank is required}"
COORDINATOR_ADDRESS="${3:?coordinator address is required}"
STATUS_FILE="${INKLING_WORKER_STATUS:-$HOME/inkle-results/${MODE}-worker-${NODE_RANK}.status}"
CHILD_PID=""

cleanup() {
  local exit_status=$?
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  mkdir -p "$(dirname "$STATUS_FILE")"
  printf '%s\n' "$exit_status" >"$STATUS_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cd "$HOME/inkle"
mkdir -p "$HOME/inkle-results" /tmp/jit_cache
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
export PATH="$HOME/.local/bin:$PATH"

if [[ "$MODE" == "qwen-control" ]]; then
  uv run python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --trust-remote-code \
    --tp-size 16 \
    --ep-size 16 \
    --moe-backend epmoe \
    --device tpu \
    --dtype bfloat16 \
    --attention-backend native \
    --mem-fraction-static 0.82 \
    --context-length 2048 \
    --chunked-prefill-size 512 \
    --max-prefill-tokens 512 \
    --max-running-requests 4 \
    --max-total-tokens 2048 \
    --page-size 128 \
    --download-dir /tmp/huggingface \
    --skip-server-warmup \
    --nnodes 4 \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$COORDINATOR_ADDRESS" \
    --host 0.0.0.0 \
    --port 30000 &
elif [[ "$MODE" == "inkling-prefix" ]]; then
  DIAGNOSTIC_ARGS=()
  if [[ "${INKLING_DIAGNOSE_SPLIT:-0}" == "1" ]]; then
    DIAGNOSTIC_ARGS+=(--diagnose-split)
  fi
  uv run python tpu_inkling_validation.py \
    --coordinator-address "$COORDINATOR_ADDRESS" \
    --num-processes 4 \
    --process-id "$NODE_RANK" \
    --prefix-layers "${INKLING_PREFIX_LAYERS:-6}" \
    --reference-directory /tmp/inkling-cpu-reference-full \
    --route-reference-directory /tmp/inkling-cpu-reference-full \
    --expert-cache-directory /tmp/inkling-expert-cache \
    --profile-directory "$HOME/inkle-results/profile" \
    --output "$HOME/inkle-results/inkling-prefix-result.json" \
    "${DIAGNOSTIC_ARGS[@]}" &
else
  echo "INKLING_WORKER_MODE_UNSUPPORTED mode=$MODE" >&2
  exit 2
fi

CHILD_PID=$!
wait "$CHILD_PID"
