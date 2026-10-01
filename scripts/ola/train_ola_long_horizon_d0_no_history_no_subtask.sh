#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/nas/code/d0}"
CONFIG_FILE="configs/generated/long_horizon_d0_50k_no_history_no_subtask.yaml"
RUN_NAME="${1:-long_horizon_d0_50k_no_history_no_subtask_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/long_horizon_d0_no_history_no_subtask/$RUN_NAME}"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/long_horizon_d0_50k_no_history_no_subtask/$RUN_NAME"

cd "$PROJECT_ROOT"
if [ -e "$CHECKPOINT_DIR" ] && [ "${ALLOW_EXISTING:-0}" != "1" ]; then
    echo "[ERROR] Checkpoint target already exists: $CHECKPOINT_DIR" >&2
    exit 1
fi

export CONFIG_FILE RUN_NAME OUTPUT_DIR
exec bash scripts/ola/train_ola_lap.sh
