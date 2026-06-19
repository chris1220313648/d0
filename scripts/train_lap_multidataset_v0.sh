#!/bin/bash
set -euo pipefail

# Four-GPU launcher for LAP multi-dataset v0 training.
# Override any of these from the shell when needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"

TASK="${TASK:-multidataset_lap_v0}"
CONFIG_FILE="${CONFIG_FILE:-configs/multidataset_lap_v0.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_4gpu}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-5}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
SMOKE_REPORT_TO="${SMOKE_REPORT_TO:-none}"
SMOKE_AGIBOT_TASK="${SMOKE_AGIBOT_TASK:-ImitationLearning/CommercialSpaces/task_3405/389111_389369_split}"

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

CONFIG_TO_RUN="$CONFIG_FILE"
LOG_FILE="$OUTPUT_DIR/train_lap_4gpu.log"

if [ "$SMOKE_TEST" = "1" ]; then
    CONFIG_TO_RUN="$OUTPUT_DIR/${TASK}_smoke.yaml"
    LOG_FILE="$OUTPUT_DIR/train_lap_4gpu_smoke.log"
    REPORT_TO="$SMOKE_REPORT_TO"
    python - "$CONFIG_FILE" "$CONFIG_TO_RUN" "$SMOKE_MAX_STEPS" "$SMOKE_BATCH_SIZE" "$SMOKE_NUM_WORKERS" "$SMOKE_REPORT_TO" "$SMOKE_AGIBOT_TASK" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, max_steps, batch_size, num_workers, report_to, agibot_task = sys.argv[1:8]
with open(src, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

datasets = config.get("dataset", {}).get("datasets", [])
agibot_datasets = [item for item in datasets if item.get("type") == "lerobot_agibot"]
if not agibot_datasets:
    raise ValueError("No lerobot_agibot dataset found in smoke source config")

agibot = agibot_datasets[0]
agibot["task_mode"] = "multi"
agibot["task_name"] = agibot_task
agibot["max_episodes"] = None
agibot.setdefault("params", {}).setdefault("task_discovery", {})["enabled"] = False

config.setdefault("training", {})["max_steps"] = int(max_steps)
config["training"]["batch_size"] = int(batch_size)
config.setdefault("system", {})["num_workers"] = int(num_workers)
config["system"]["val_interval"] = int(max_steps) + 1
config["system"]["save_interval"] = int(max_steps) + 1
config["system"]["save_final_checkpoint"] = False
config.setdefault("logging", {})["report_to"] = report_to

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
    echo "[INFO] Smoke test enabled: config=$CONFIG_TO_RUN max_steps=$SMOKE_MAX_STEPS batch_size=$SMOKE_BATCH_SIZE num_workers=$SMOKE_NUM_WORKERS report_to=$SMOKE_REPORT_TO agibot_task=$SMOKE_AGIBOT_TASK"
fi

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config "$CONFIG_TO_RUN" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO" \
    --log_level "$LOG_LEVEL" \
    > "$LOG_FILE" 2>&1
