#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/nas/code/d0}"
CONFIG_FILE="configs/generated/open_drawer_lap0p1.yaml"
RUN_NAME="${1:-open_drawer_lap0p1_hisinit_future_noise}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/open_drawer_lap0p1/$RUN_NAME}"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/open_drawer_lap0p1/$RUN_NAME"

cd "$PROJECT_ROOT"

if [ -e "$CHECKPOINT_DIR" ] && [ "${ALLOW_EXISTING:-0}" != "1" ]; then
    echo "[ERROR] Checkpoint target already exists: $CHECKPOINT_DIR" >&2
    echo "Choose a new run name, or set ALLOW_EXISTING=1 only when intentional." >&2
    exit 1
fi

export CONFIG_FILE RUN_NAME OUTPUT_DIR
exec bash scripts/ola/train_ola_lap.sh
