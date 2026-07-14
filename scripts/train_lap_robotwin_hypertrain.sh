#!/usr/bin/env bash
set -euo pipefail

# HyperTrain/Kubernetes launcher for RobotWin LAP finetuning.
# Run this same script as the HyperTrain startup command on every pod.
# HyperTrain should inject MASTER_ADDR, MASTER_PORT, NODE_RANK,
# NPROC_PER_NODE and NNODES for multi-node jobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
    source /share/anaconda3/etc/profile.d/conda.sh
else
    echo "[ERROR] Could not find conda.sh" >&2
    exit 1
fi

conda activate motus

TASK="${TASK:-robotwin_lap}"
CONFIG_FILE="${CONFIG_FILE:-configs/robotwin_lap_v1.yaml}"
RUN_NAME="${RUN_NAME:-${TASK}_hypertrain_v1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Defaults for 2 nodes x 8 GPUs. HyperTrain-provided environment variables
# override these values at runtime.
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
DIST_BACKEND="${DIST_BACKEND:-deepspeed}"
WORLD_SIZE="${NNODES}*${NPROC_PER_NODE}"

mkdir -p "$OUTPUT_DIR"

CONFIG_TO_RUN="$CONFIG_FILE"
LOG_SUFFIX="node${NODE_RANK}"
LOG_FILE="$OUTPUT_DIR/train_lap_hypertrain_${LOG_SUFFIX}.log"

exec > >(tee "$LOG_FILE") 2>&1

echo "=========================================="
echo "HyperTrain RobotWin LAP Finetune"
echo "Time: $(date)"
echo "Host: ${HOSTNAME:-$(hostname)}"
echo "Project root: $PROJECT_ROOT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE_RANK: $NODE_RANK"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "DIST_BACKEND: $DIST_BACKEND"
echo "Task: $TASK"
echo "Config: $CONFIG_TO_RUN"
echo "DeepSpeed: $DEEPSPEED_CONFIG"
echo "Run name: $RUN_NAME"
echo "Report to: $REPORT_TO"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=========================================="

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config "$CONFIG_TO_RUN" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO" \
    --log_level "$LOG_LEVEL"
