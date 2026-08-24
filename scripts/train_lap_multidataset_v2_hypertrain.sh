#!/usr/bin/env bash
set -euo pipefail

# HyperTrain/Kubernetes launcher for LAP multi-dataset v2 robot-only training.
# Run this same script as the HyperTrain startup command on every pod.
# This follows the official HyperTrain torchrun example: the platform should
# inject MASTER_ADDR, MASTER_PORT, NODE_RANK, NPROC_PER_NODE and NNODES.

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

TASK="${TASK:-multidataset_lap_v2}"
CONFIG_FILE="${CONFIG_FILE:-configs/multidataset_lap_v2.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_hypertrain}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

HF_HOME="${HF_HOME:-/root/nasbak/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_HUB_OFFLINE HF_DATASETS_OFFLINE

# Keep these defaults aligned with scripts/slurm/train.sh, the official
# HyperTrain example in this repo. For multi-node jobs HyperTrain should
# provide the real values through task environment variables.
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
DIST_BACKEND="${DIST_BACKEND:-deepspeed}"
WORLD_SIZE="${NNODES}*${NPROC_PER_NODE}"

python -c "import peft" >/dev/null 2>&1 || {
    echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"

CONFIG_TO_RUN="$CONFIG_FILE"
LOG_SUFFIX="node${NODE_RANK}"
LOG_FILE="$OUTPUT_DIR/train_lap_hypertrain_${LOG_SUFFIX}.log"

exec > >(tee "$LOG_FILE") 2>&1

echo "=========================================="
echo "HyperTrain LAP Multi-Dataset v2"
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
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "HF_DATASETS_OFFLINE: $HF_DATASETS_OFFLINE"
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
