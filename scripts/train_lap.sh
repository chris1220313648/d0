#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/nas/code/d0}"
cd "$PROJECT_ROOT"

# Define your env settings here
# e.g., nccl, network, proxy, etc.
# export CUDA_VISIBLE_DEVICES=6,7
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
    source /share/anaconda3/etc/profile.d/conda.sh
else
    echo "[ERROR] Could not find conda.sh" >&2
    exit 1
fi

conda activate motus

python -c "import peft" >/dev/null 2>&1 || {
    echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
    exit 1
}

TASK="${TASK:-robotwin}"  # Define your task name here
CONFIG_FILE="${CONFIG_FILE:-configs/robotwin_lap.yaml}"  # Define your dataset config path here
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_d0_v0_after_pretrain}"
REPORT_TO="${REPORT_TO:-wandb}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}" # Define your output directory here

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Single-node training with torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config "$CONFIG_FILE" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO" \
    > "$OUTPUT_DIR/train_lap.log" 2>&1
