#!/usr/bin/env bash

set -euo pipefail

# Set INKLING_GCP_PROJECT to your GCP project (e.g. export it in your shell profile).
PROJECT="${INKLING_GCP_PROJECT:-your-gcp-project}"
ZONE="${INKLING_GCP_ZONE:-us-central1-a}"
TPU_NAME="${INKLING_TPU_NAME:-inkling-v5e-4}"
ACCELERATOR_TYPE="${INKLING_ACCELERATOR_TYPE:-v5litepod-4}"
PROMPT="${INKLING_PROMPT:-The capital of France is}"
RESULTS_DIRECTORY="${INKLING_RESULTS_DIRECTORY:-/tmp/inkling-gcp-results-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELETE_ON_FAILURE="${INKLING_DELETE_ON_FAILURE:-0}"
REUSE_TPU="${INKLING_REUSE_TPU:-0}"
MANAGE_TPU=0
printf -v QUOTED_PROMPT "%q" "$PROMPT"

cleanup() {
  local exit_status=$?
  if [[ "$MANAGE_TPU" == "1" ]]; then
    if [[ "$exit_status" != "0" && "$DELETE_ON_FAILURE" != "1" ]]; then
      echo "INKLING_TPU_PRESERVED_AFTER_FAILURE name=$TPU_NAME zone=$ZONE" >&2
      return
    fi
    if gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
      gcloud compute tpus tpu-vm delete "$TPU_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet || true
    fi
    MANAGE_TPU=0
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  if [[ "$REUSE_TPU" != "1" ]]; then
    echo "INKLING_TPU_ALREADY_EXISTS name=$TPU_NAME zone=$ZONE" >&2
    exit 1
  fi
  echo "INKLING_TPU_REUSED name=$TPU_NAME zone=$ZONE"
  MANAGE_TPU=1
else
  gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --version=v2-tpuv5-litepod \
    --quiet
  MANAGE_TPU=1
fi

mkdir -p "$RESULTS_DIRECTORY"

gcloud compute tpus tpu-vm scp \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --worker=0 \
  --recurse \
  "$SOURCE_DIRECTORY" \
  "$TPU_NAME:~/inkling"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --worker=0 \
  --command="curl -LsSf https://astral.sh/uv/install.sh | sh"

set +e
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --worker=0 \
  --command="~/.local/bin/uv run ~/inkling/streaming_tpu_inference.py --no-initialize-distributed --prompt $QUOTED_PROMPT --output ~/inkling-output/output.npz --validation-directory ~/inkling-validation" \
  2>&1 | tee "$RESULTS_DIRECTORY/inference.log"
INFERENCE_STATUS=${PIPESTATUS[0]}
set -e

for WORKER in 0; do
  mkdir -p "$RESULTS_DIRECTORY/worker-$WORKER"
  if gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --worker="$WORKER" \
    --command="test -d ~/inkling-validation" \
    --quiet; then
    gcloud compute tpus tpu-vm scp \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --worker="$WORKER" \
      --recurse \
      "$TPU_NAME:~/inkling-validation" \
      "$RESULTS_DIRECTORY/worker-$WORKER/"
  fi
  if gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --worker="$WORKER" \
    --command="test -d ~/inkling-output" \
    --quiet; then
    gcloud compute tpus tpu-vm scp \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --worker="$WORKER" \
      --recurse \
      "$TPU_NAME:~/inkling-output" \
      "$RESULTS_DIRECTORY/worker-$WORKER/"
  fi
done

echo "INKLING_RESULTS_DIRECTORY path=$RESULTS_DIRECTORY"
exit "$INFERENCE_STATUS"
