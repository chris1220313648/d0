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

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TASK="${TASK:-libero_raw_osc}"
CONFIG_FILE="${CONFIG_FILE:-configs/libero_raw_osc_lap.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_after_pretrain_gradacc4}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}/${RUN_NAME}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

if [ "$NPROC_PER_NODE" -le 0 ]; then
    echo "[ERROR] No visible GPUs. Run inside the allocated GPU container or set NPROC_PER_NODE." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR

echo "Starting LIBERO raw OSC LAP training"
echo "  project=$PROJECT_ROOT"
echo "  config=$CONFIG_FILE"
echo "  run_name=$RUN_NAME"
echo "  nproc_per_node=$NPROC_PER_NODE"
echo "  output=$OUTPUT_DIR"

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
    > "$OUTPUT_DIR/train.log" 2>&1
