#!/usr/bin/env bash
set -euo pipefail

# Train RoboTwin LAP with current-frame/history-qpos flow sources after RB process.
PROJECT_ROOT="/root/nas/code/d0"
CONFIG_FILE="configs/robotwin_lap_history_flow_RBprocess_1.0.yaml"
TASK="robotwin"
OUTPUT_DIR="$PROJECT_ROOT/outputs/motus-${TASK}"

source /opt/conda/etc/profile.d/conda.sh
conda activate motus

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -c "import peft" >/dev/null 2>&1 || {
  echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  train/train.py \
  --deepspeed configs/zero2.json \
  --config "$CONFIG_FILE" \
  --run_name "${TASK}_lap_action" \
  --report_to tensorboard \
  2>&1 | tee "$OUTPUT_DIR/train_lap_history_flow_action.log"
