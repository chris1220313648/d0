#!/usr/bin/env bash
set -euo pipefail

# HyperTrain launcher for RobotWin LAP v3 finetuning.
# This wrapper reuses the shared RobotWin HyperTrain launcher and only changes
# the default task/config/run naming.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK="${TASK:-robotwin_lap_v3}"
export CONFIG_FILE="${CONFIG_FILE:-configs/robotwin_lap_v3.yaml}"
export RUN_NAME="${RUN_NAME:-${TASK}_hypertrain}"

exec "$SCRIPT_DIR/train_lap_robotwin_hypertrain.sh" "$@"
