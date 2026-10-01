#!/usr/bin/env bash
# Evaluate the LIBERO raw-OSC H16 gradacc4 step-40000 checkpoint on 8 GPUs.

set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CKPT="${CKPT:-${MOTUS_ROOT}/checkpoints/libero_raw_osc_lap_h16_gradacc4_40k/libero_raw_osc_h16_lap_after_pretrain_gradacc4_step40000/checkpoint_step_40000/pytorch_model}"
CONFIG_PATH="${CONFIG_PATH:-configs/libero_raw_osc_lap_h16_gradacc4_40k.yaml}"
SUITE="${1:-${SUITE:-libero_spatial}}"
if [[ -z "${GPU_IDS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    else
        GPU_IDS="0,1,2,3,4,5,6,7"
    fi
fi
NUM_TRIALS="${NUM_TRIALS:-1}"
VIDEO_MODE="${VIDEO_MODE:-failures}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MOTUS_ROOT}/outputs/libero_plus_8gpu}"
BASE_PORT="${BASE_PORT:-9883}"
MAX_TASKS="${MAX_TASKS:--1}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_libero_plus_eval_step40000_8gpu.sh [libero_spatial|libero_object|libero_goal|libero_10|all]

Environment overrides:
  CKPT=/path/to/checkpoint_step_x/pytorch_model
  CONFIG_PATH=configs/libero_raw_osc_lap_h16_gradacc4_40k.yaml
  GPU_IDS=0,1,2,3,4,5,6,7  # default: all GPUs visible to nvidia-smi
  NUM_TRIALS=1
  VIDEO_MODE=none|failures|all
  OUTPUT_ROOT=/root/nas/code/d0/outputs/libero_plus_8gpu
  BASE_PORT=9883
  MAX_TASKS=-1
  RESUME=1
  DRY_RUN=1
EOF
}

case "${SUITE}" in
    --help|-h) usage; exit 0 ;;
    libero_spatial|libero_object|libero_goal|libero_10|all) ;;
    *) echo "Unsupported suite: ${SUITE}" >&2; usage >&2; exit 2 ;;
esac

[[ -d "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}" >&2; exit 1; }
[[ -f "${MOTUS_ROOT}/${CONFIG_PATH}" || -f "${CONFIG_PATH}" ]] || {
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
}

run_suite() {
    local suite="$1"
    local run_name="step40000_${suite}_8gpu"
    local args=(
        --checkpoint "${CKPT}"
        --task "${suite}"
        --gpu-ids "${GPU_IDS}"
        --base-port "${BASE_PORT}"
        --num-trials "${NUM_TRIALS}"
        --video-mode "${VIDEO_MODE}"
        --output-root "${OUTPUT_ROOT}"
        --run-name "${run_name}"
        --max-tasks "${MAX_TASKS}"
        --mujoco-gl "${MUJOCO_GL}"
    )

    if [[ "${RESUME}" == "1" ]]; then
        args+=(--resume)
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        args+=(--dry-run)
    fi

    echo "Evaluating ${suite} with GPUs ${GPU_IDS}"
    CONFIG_PATH="${CONFIG_PATH}" bash "${MOTUS_ROOT}/scripts/run_libero_plus_eval_multi_gpu.sh" "${args[@]}"
}

if [[ "${SUITE}" == "all" ]]; then
    for suite in libero_spatial libero_object libero_goal libero_10; do
        run_suite "${suite}"
    done
else
    run_suite "${SUITE}"
fi
