#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/nas/code/d0}"
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

python -c "import peft" >/dev/null 2>&1 || {
    echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
    exit 1
}

# This script assumes GPUs have already been allocated by the user.
CONFIG_FILE="${CONFIG_FILE:-configs/ola_lap.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-ola_lap_d0_v_after_pretrain}"
REPORT_TO="${REPORT_TO:-wandb}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-ola-lap}"
export OUTPUT_DIR

mkdir -p "$OUTPUT_DIR"

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

