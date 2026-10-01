#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-configs/multidataset_lap_robot6_24a_eef_norm.yaml}"
if [[ "${SMOKE:-1}" == 1 ]]; then
    export RUN_NAME="${RUN_NAME:-robot6_eef_delta24_8gpu_bs5_smoke100}"
    export OUTPUT_DIR="${OUTPUT_DIR:-outputs/robot6_eef_delta24_smoke}"
else
    export RUN_NAME="${RUN_NAME:-robot6_eef_delta24_8gpu_bs5}"
    export OUTPUT_DIR="${OUTPUT_DIR:-outputs/robot6_eef_delta24_train}"
fi
exec bash "$SCRIPT_DIR/train_robot6_14a_norm_8gpu.sh"
