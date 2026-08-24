#!/usr/bin/env bash
set -euo pipefail

# Single-node 8-GPU launcher for LAP multi-dataset v2 with lazy LeRobot loaders.
# All settings remain overridable through the environment and are delegated to
# the standard v2 launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASE_LAUNCHER="$SCRIPT_DIR/train_lap_multidataset_v2.sh"

cd "$PROJECT_ROOT"

export CONFIG_FILE="${CONFIG_FILE:-configs/multidataset_lap_v2_lazy.yaml}"
export TASK="${TASK:-multidataset_lap_v2_lazy}"
export RUN_NAME="${RUN_NAME:-${TASK}_lap_8gpu}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Lazy config file not found: $CONFIG_FILE" >&2
    exit 1
fi

if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER" >&2
    exit 1
fi

exec "$BASE_LAUNCHER" "$@"
