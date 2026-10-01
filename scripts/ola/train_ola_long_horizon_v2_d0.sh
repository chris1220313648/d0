#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 all|new [RUN_NAME]}"
case "$MODE" in
  all) STEP_TAG=60k ;;
  new) STEP_TAG=40k ;;
  *) echo "[ERROR] MODE must be all or new" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/root/nas/code/d0}"
CONFIG_FILE="configs/generated/long_horizon_${MODE}_v2_d0.yaml"
RUN_NAME="${2:-long_horizon_${MODE}_v2_d0_${STEP_TAG}_v1}"
DATASET_NAME="$(basename "$CONFIG_FILE" .yaml)"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/$DATASET_NAME/$RUN_NAME"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$DATASET_NAME/$RUN_NAME}"

cd "$PROJECT_ROOT"
if [ -e "$CHECKPOINT_DIR" ] && [ "${ALLOW_EXISTING:-0}" != "1" ]; then
  echo "[ERROR] Checkpoint target already exists: $CHECKPOINT_DIR" >&2
  exit 1
fi

export CONFIG_FILE RUN_NAME OUTPUT_DIR
exec bash scripts/ola/train_ola_lap.sh
