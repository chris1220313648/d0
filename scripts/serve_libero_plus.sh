#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CHECKPOINT_PATH [PORT] [CONFIG_PATH]" >&2
    exit 2
fi

MOTUS_ROOT="${MOTUS_ROOT:-/root/nas/code/d0}"
CHECKPOINT_PATH="$1"
PORT="${2:-9883}"
CONFIG_PATH="${3:-${CONFIG_PATH:-configs/libero_raw_osc_lap_h16.yaml}}"

source /opt/conda/etc/profile.d/conda.sh
conda activate motus
cd "$MOTUS_ROOT"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${MOTUS_ROOT}:${PYTHONPATH:-}"

exec python -m examples.libero_plus.policy_server \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --wan-path pretrained_models/Wan2.2-TI2V-5B \
    --vlm-path pretrained_models/Qwen3-VL-2B-Instruct \
    --action-stats /root/nasbak/cjy/robot_raw/libero/motus_cache/raw_osc_action_stats.json \
    --port "$PORT"
