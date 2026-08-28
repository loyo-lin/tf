#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXPORT_ROOT="/root/autodl-tmp/tf_exports"
if [ -d /root/gpufree-data ]; then
  DEFAULT_EXPORT_ROOT="/root/gpufree-data/tf_exports"
fi
EXPORT_ROOT="${EXPORT_ROOT:-$DEFAULT_EXPORT_ROOT}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
EXPORT_DIR="${EXPORT_ROOT}/${RUN_NAME}"
LOG_FILE="${EXPORT_DIR}/train.log"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
SHUTDOWN_ON_ERROR="${SHUTDOWN_ON_ERROR:-0}"

mkdir -p "$EXPORT_DIR"
cd "$PROJECT_DIR"

echo "Project: $PROJECT_DIR"
echo "Export:  $EXPORT_DIR"
echo "Log:     $LOG_FILE"

archive_outputs() {
  mkdir -p "$EXPORT_DIR"

  cp -f config.py dataset.py model.py train.py validation.py attention_visualization.py "$EXPORT_DIR"/
  cp -f requirements.txt "$EXPORT_DIR"/ 2>/dev/null || true

  if compgen -G "tokenizer_*.json" >/dev/null; then
    cp -f tokenizer_*.json "$EXPORT_DIR"/
  fi

  if [ -d weights ]; then
    cp -a weights "$EXPORT_DIR"/
  fi

  if [ -d runs ]; then
    cp -a runs "$EXPORT_DIR"/
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
  if [ -d wheelhouse ] && [ -f requirements_cloud.txt ]; then
    python -m pip install --no-index --find-links wheelhouse -r requirements_cloud.txt
  elif [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
  fi
fi

mkdir -p weights runs

set +e
python train.py 2>&1 | tee "$LOG_FILE"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

archive_outputs

if [ "$TRAIN_STATUS" -eq 0 ] || [ "$SHUTDOWN_ON_ERROR" = "1" ]; then
  shutdown_instance
else
  echo "Training failed with exit code $TRAIN_STATUS. Server is left running for debugging."
fi

exit "$TRAIN_STATUS"
