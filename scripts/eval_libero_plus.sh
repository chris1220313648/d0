#!/bin/bash
set -euo pipefail

SUITE="${1:-libero_goal}"
PORT="${2:-9883}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/nas/code/LIBERO-plus}"
MOTUS_ROOT="${MOTUS_ROOT:-/root/nas/code/d0}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/opt/conda/envs/liberoplus/bin/python}"
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-$MOTUS_ROOT/outputs/libero_plus}"
RESUME="${RESUME:-0}"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
    resume_args+=(--resume)
fi

case "$SUITE" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *) echo "Unsupported suite: $SUITE" >&2; exit 2 ;;
esac

export LIBERO_HOME="$LIBERO_PLUS_ROOT"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/.libero_plus}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${LIBERO_PLUS_ROOT}:${MOTUS_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

exec "$LIBERO_PLUS_PYTHON" \
    "$MOTUS_ROOT/examples/libero_plus/eval_libero.py" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --task-suite-name "$SUITE" \
    --num-trials-per-task "$NUM_TRIALS" \
    --max-tasks "$MAX_TASKS" \
    --output-dir "$OUTPUT_DIR" \
    "${resume_args[@]}"
