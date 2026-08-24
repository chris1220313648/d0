#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

SESSION_NAME="${SESSION_NAME:-interndata_a1_lap_v4_14a_all_4gpu}"
TASK="${TASK:-interndata_a1_lap_v4_14a_smoke100_alltasks}"
CONFIG_FILE="${CONFIG_FILE:-configs/interndata_a1_lap_v4_14a_smoke100.yaml}"
RUN_NAME="${RUN_NAME:-${TASK}_4gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
MASTER_PORT="${MASTER_PORT:-29524}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
REPORT_TO="${REPORT_TO:-tensorboard}"
LAUNCHER="$PROJECT_ROOT/scripts/train_lap_multidataset_v4_14a_hypertrain.sh"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is required but was not found" >&2
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $PROJECT_ROOT/$CONFIG_FILE" >&2
    exit 1
fi

if [ ! -f "$LAUNCHER" ]; then
    echo "[ERROR] Launcher not found: $LAUNCHER" >&2
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"

quoted_project_root=$(printf '%q' "$PROJECT_ROOT")
quoted_launcher=$(printf '%q' "$LAUNCHER")
quoted_cuda_devices=$(printf '%q' "$CUDA_VISIBLE_DEVICES")
quoted_task=$(printf '%q' "$TASK")
quoted_config=$(printf '%q' "$CONFIG_FILE")
quoted_run_name=$(printf '%q' "$RUN_NAME")
quoted_output_dir=$(printf '%q' "$OUTPUT_DIR")
quoted_master_port=$(printf '%q' "$MASTER_PORT")
quoted_nproc=$(printf '%q' "$NPROC_PER_NODE")
quoted_report_to=$(printf '%q' "$REPORT_TO")

tmux new-session -d -s "$SESSION_NAME" \
    "cd $quoted_project_root && exec env CUDA_VISIBLE_DEVICES=$quoted_cuda_devices TASK=$quoted_task CONFIG_FILE=$quoted_config RUN_NAME=$quoted_run_name OUTPUT_DIR=$quoted_output_dir MASTER_ADDR=127.0.0.1 MASTER_PORT=$quoted_master_port NPROC_PER_NODE=$quoted_nproc NNODES=1 NODE_RANK=0 REPORT_TO=$quoted_report_to bash $quoted_launcher"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log: $PROJECT_ROOT/$OUTPUT_DIR/train_lap_hypertrain_node0.log"
echo "Stop: tmux kill-session -t $SESSION_NAME"
