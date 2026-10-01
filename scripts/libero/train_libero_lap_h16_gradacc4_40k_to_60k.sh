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

export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

TASK="${TASK:-libero_raw_osc_h16}"
CONFIG_FILE="${CONFIG_FILE:-configs/libero_raw_osc_lap_h16_gradacc4_60k.yaml}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_after_pretrain_gradacc4_step60000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_40k_to_60k.log}"
DATASET_DIR="${DATASET_DIR:-/root/nasbak/cjy/robot_raw/libero}"
CACHE_DIR="${CACHE_DIR:-${DATASET_DIR}/motus_cache}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/root/nas/code/d0/checkpoints/libero_raw_osc_lap_h16_gradacc4_40k/libero_raw_osc_h16_lap_after_pretrain_gradacc4_step40000/checkpoint_step_40000}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29512}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

export RESUME_CHECKPOINT

if [ "$NPROC_PER_NODE" -le 0 ]; then
    echo "[ERROR] No visible GPUs. Run inside the allocated GPU container or set NPROC_PER_NODE." >&2
    exit 1
fi

if [ ! -d "$RESUME_CHECKPOINT" ]; then
    echo "[ERROR] Resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting LIBERO H16-E8 training from 40k to 60k:"
echo "  RUN_NAME=$RUN_NAME"
echo "  CONFIG_FILE=$CONFIG_FILE"
echo "  RESUME_CHECKPOINT=$RESUME_CHECKPOINT"
echo "  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  LOG=$LOG_FILE"

python -m data.libero.generate_action_stats \
    --dataset-dir "$DATASET_DIR" \
    --output "$CACHE_DIR/raw_osc_action_stats.json"

python -m data.libero.generate_language_action \
    --dataset-dir "$DATASET_DIR" \
    --cache-dir "$CACHE_DIR" \
    --horizon 16 \
    --lap-subdir lap_h16

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed configs/zero2.json \
    --config "$CONFIG_FILE" \
    --run_name "$RUN_NAME" \
    --report_to tensorboard \
    > "$LOG_FILE" 2>&1
