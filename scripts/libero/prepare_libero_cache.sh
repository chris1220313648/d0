#!/bin/bash
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

DATASET_DIR="${DATASET_DIR:-/root/nasbak/cjy/robot_raw/libero}"
CACHE_DIR="${CACHE_DIR:-${DATASET_DIR}/motus_cache}"
WAN_PATH="${WAN_PATH:-${PROJECT_ROOT}/pretrained_models}"
T5_DEVICE=${T5_DEVICE:-cuda}

python -m data.libero.generate_action_stats \
    --dataset-dir "$DATASET_DIR" \
    --output "$CACHE_DIR/raw_osc_action_stats.json"

python -m data.libero.generate_language_action \
    --dataset-dir "$DATASET_DIR" \
    --cache-dir "$CACHE_DIR"

python -m data.libero.generate_t5_embeddings \
    --dataset-dir "$DATASET_DIR" \
    --cache-dir "$CACHE_DIR" \
    --wan-path "$WAN_PATH" \
    --device "$T5_DEVICE"

echo "LIBERO cache is ready at $CACHE_DIR"
