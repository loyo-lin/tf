#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXPORT_ROOT="/root/tf_exports"
if [ -d /root/gpufree-data ]; then
  DEFAULT_EXPORT_ROOT="/root/gpufree-data/tf_exports"
fi
if [ -d /root/autodl-tmp ]; then
  DEFAULT_EXPORT_ROOT="/root/autodl-tmp/tf_exports"
fi
EXPORT_ROOT="${EXPORT_ROOT:-$DEFAULT_EXPORT_ROOT}"
RUN_NAME="${RUN_NAME:-${RESUME_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}}"
EXPORT_DIR="${EXPORT_ROOT}/${RUN_NAME}"
LOG_FILE="${EXPORT_DIR}/train.log"
MODEL_FOLDER="${MODEL_FOLDER:-${EXPORT_DIR}/weights}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPORT_DIR}/runs/tmodel}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
PROGRESS_LOG_INTERVAL_SECONDS="${PROGRESS_LOG_INTERVAL_SECONDS:-1800}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
SHUTDOWN_ON_ERROR="${SHUTDOWN_ON_ERROR:-0}"

mkdir -p "$EXPORT_DIR"
cd "$PROJECT_DIR"
export MODEL_FOLDER EXPERIMENT_NAME CHECKPOINT_INTERVAL PROGRESS_LOG_INTERVAL_SECONDS

exec > >(tee -a "$LOG_FILE") 2>&1
echo
echo "===== Training started at $(date -Is) ====="

echo "Project: $PROJECT_DIR"
echo "Export:  $EXPORT_DIR"
echo "Log:     $LOG_FILE"
echo "Weights: $MODEL_FOLDER"
echo "Runs:    $EXPERIMENT_NAME"
echo "Checkpoint interval: every $CHECKPOINT_INTERVAL steps"
echo "Progress log interval: every $PROGRESS_LOG_INTERVAL_SECONDS seconds"
if [[ "$EXPORT_ROOT" == /root/gpufree-data* ]]; then
  echo "WARNING: /root/gpufree-data is not saved after releasing the instance."
  echo "Download ${EXPORT_ROOT}/${RUN_NAME}.tar.gz or copy it to file storage before release."
fi

if [ -n "${PRELOAD:-}" ] && [ ! -f "${MODEL_FOLDER}/tmodel_${PRELOAD}.pt" ]; then
  echo "ERROR: PRELOAD=${PRELOAD}, but checkpoint was not found:"
  echo "  ${MODEL_FOLDER}/tmodel_${PRELOAD}.pt"
  echo "Use RUN_NAME=<old_run_name> PRELOAD=latest, or set MODEL_FOLDER to the old weights directory."
  exit 2
fi

archive_outputs() {
  mkdir -p "$EXPORT_DIR"

  cp -f config.py dataset.py model.py train.py validation.py attention_visualization.py "$EXPORT_DIR"/
  cp -f requirements.txt "$EXPORT_DIR"/ 2>/dev/null || true

  if compgen -G "tokenizer_*.json" >/dev/null; then
    cp -f tokenizer_*.json "$EXPORT_DIR"/
  fi

  if [ -d "$MODEL_FOLDER" ] && [ "$MODEL_FOLDER" != "${EXPORT_DIR}/weights" ]; then
    rm -rf "${EXPORT_DIR}/weights"
    cp -a "$MODEL_FOLDER" "${EXPORT_DIR}/weights"
  fi

  RUNS_ROOT="$(dirname "$EXPERIMENT_NAME")"
  if [ -d "$RUNS_ROOT" ] && [ "$RUNS_ROOT" != "${EXPORT_DIR}/runs" ]; then
    rm -rf "${EXPORT_DIR}/runs"
    cp -a "$RUNS_ROOT" "${EXPORT_DIR}/runs"
  fi

  tar -czf "${EXPORT_ROOT}/${RUN_NAME}.tar.gz" -C "$EXPORT_ROOT" "$RUN_NAME"
  echo "Saved outputs:"
  echo "  ${EXPORT_DIR}"
  echo "  ${EXPORT_ROOT}/${RUN_NAME}.tar.gz"
}

shutdown_instance() {
  if [ "$AUTO_SHUTDOWN" != "1" ]; then
    echo "AUTO_SHUTDOWN is not 1, skip shutdown."
    return
  fi

  echo "Training script finished. The server will shut down in 20 seconds."
  sync
  (
    sleep 20
    if command -v autodl >/dev/null 2>&1; then
      autodl shutdown >/dev/null 2>&1 || true
    fi
    shutdown -h now >/dev/null 2>&1 || poweroff >/dev/null 2>&1 || halt >/dev/null 2>&1 || true
  ) &
}

if [ "$INSTALL_DEPS" = "1" ]; then
  if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
  fi
fi

mkdir -p "$MODEL_FOLDER" "$(dirname "$EXPERIMENT_NAME")"

set +e
python train.py
TRAIN_STATUS=$?
set -e

archive_outputs

if [ "$TRAIN_STATUS" -eq 0 ] || [ "$SHUTDOWN_ON_ERROR" = "1" ]; then
  shutdown_instance
else
  echo "Training failed with exit code $TRAIN_STATUS. Server is left running for debugging."
fi

exit "$TRAIN_STATUS"
