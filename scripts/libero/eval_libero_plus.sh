#!/bin/bash
set -euo pipefail

SUITE="${1:-libero_goal}"
PORT="${2:-9883}"
BENCHMARK_MODE="${BENCHMARK_MODE:-plus}"
case "$BENCHMARK_MODE" in
    plus)
        LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/nas/code/LIBERO-plus}"
        ;;
    original)
        LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/nas/code/LIBERO-original}"
        ;;
    *) echo "Unsupported benchmark mode: $BENCHMARK_MODE" >&2; exit 2 ;;
esac
MOTUS_ROOT="${MOTUS_ROOT:-/root/nas/code/d0}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$MOTUS_ROOT/outputs/.libero_configs/$BENCHMARK_MODE}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/opt/conda/envs/liberoplus/bin/python}"
if [[ "$BENCHMARK_MODE" == "original" ]]; then
    NUM_TRIALS="${NUM_TRIALS:-20}"
else
    NUM_TRIALS="${NUM_TRIALS:-50}"
fi
MAX_TASKS="${MAX_TASKS:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-$MOTUS_ROOT/outputs/libero_${BENCHMARK_MODE}}"
RESUME="${RESUME:-0}"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
    resume_args+=(--resume)
fi

case "$SUITE" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *) echo "Unsupported suite: $SUITE" >&2; exit 2 ;;
esac

if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
    mkdir -p "$LIBERO_CONFIG_PATH"
    printf '%s\n' \
        "assets: $LIBERO_PLUS_ROOT/libero/libero/assets" \
        "bddl_files: $LIBERO_PLUS_ROOT/libero/libero/bddl_files" \
        "benchmark_root: $LIBERO_PLUS_ROOT/libero/libero" \
        "datasets: $LIBERO_PLUS_ROOT/libero/datasets" \
        "init_states: $LIBERO_PLUS_ROOT/libero/libero/init_files" \
        >"$LIBERO_CONFIG_PATH/config.yaml"
    echo "Generated LIBERO config: $LIBERO_CONFIG_PATH/config.yaml"
fi

export LIBERO_HOME="$LIBERO_PLUS_ROOT"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${LIBERO_PLUS_ROOT}:${MOTUS_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

exec "$LIBERO_PLUS_PYTHON" \
    "$MOTUS_ROOT/examples/libero_plus/eval_libero.py" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --benchmark-mode "$BENCHMARK_MODE" \
    --task-suite-name "$SUITE" \
    --num-trials-per-task "$NUM_TRIALS" \
    --max-tasks "$MAX_TASKS" \
    --output-dir "$OUTPUT_DIR" \
    "${resume_args[@]}"
