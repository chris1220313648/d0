#!/bin/bash
set -euo pipefail

# Two-GPU launcher for AgiBot split + EgoVerse LAP multi-dataset training.
# Override any of these from the shell when needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"

TASK="${TASK:-multidataset_agibot_split_humanvideo}"
CONFIG_FILE="${CONFIG_FILE:-configs/multidataset_agibot_split_humanvideo_lap.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_2gpu}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

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
    --log_level "$LOG_LEVEL" \
    > "$OUTPUT_DIR/train_lap_2gpu.log" 2>&1
