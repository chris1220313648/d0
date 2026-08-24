#!/usr/bin/env bash
set -euo pipefail

# Independent single-node launcher for eager RoboCOIN-only LAP v2 training.
# All settings can be overridden through environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29514}"

TASK="${TASK:-robocoin_lap_v2}"
CONFIG_FILE="${CONFIG_FILE:-configs/robocoin_lap_v2.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_${NPROC_PER_NODE}gpu}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

HF_HOME="${HF_HOME:-/root/nasbak/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-5}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
SMOKE_REPORT_TO="${SMOKE_REPORT_TO:-none}"
SMOKE_ROBOCOIN_TASK="${SMOKE_ROBOCOIN_TASK:-AI2_Alphabot_2_arrange_teaset}"
SMOKE_MAX_EPISODES="${SMOKE_MAX_EPISODES:-2}"
GENERATE_CONFIG_ONLY="${GENERATE_CONFIG_ONLY:-0}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
    source /share/anaconda3/etc/profile.d/conda.sh
else
    echo "[ERROR] Could not find conda.sh" >&2
    exit 1
fi

conda activate motus
export HF_HOME
export HF_HUB_CACHE
export HF_DATASETS_CACHE
export HF_HUB_OFFLINE
export HF_DATASETS_OFFLINE

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CONFIG_TO_RUN="$CONFIG_FILE"
LOG_FILE="$OUTPUT_DIR/train_lap.log"

if [ "$SMOKE_TEST" = "1" ]; then
    CONFIG_TO_RUN="$OUTPUT_DIR/${TASK}_smoke.yaml"
    LOG_FILE="$OUTPUT_DIR/train_lap_smoke.log"
    REPORT_TO="$SMOKE_REPORT_TO"

    python - \
        "$CONFIG_FILE" \
        "$CONFIG_TO_RUN" \
        "$SMOKE_MAX_STEPS" \
        "$SMOKE_BATCH_SIZE" \
        "$SMOKE_NUM_WORKERS" \
        "$SMOKE_REPORT_TO" \
        "$SMOKE_MAX_EPISODES" \
        "$SMOKE_ROBOCOIN_TASK" <<'PY'
import sys
from pathlib import Path

import yaml

(
    src,
    dst,
    max_steps,
    batch_size,
    num_workers,
    report_to,
    max_episodes,
    robocoin_task,
) = sys.argv[1:9]

with Path(src).open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

dataset = config.get("dataset", {})
if dataset.get("type") != "lerobot_robocoin":
    raise ValueError(
        "RoboCOIN smoke source config must use dataset.type='lerobot_robocoin'"
    )

dataset["task_mode"] = "multi"
dataset["task_name"] = robocoin_task
dataset["max_episodes"] = int(max_episodes)
task_discovery = dataset.setdefault("params", {}).get("task_discovery")
if task_discovery is not None:
    task_discovery["enabled"] = False

max_steps_value = int(max_steps)
config.setdefault("training", {})["max_steps"] = max_steps_value
config["training"]["batch_size"] = int(batch_size)
config.setdefault("system", {})["num_workers"] = int(num_workers)
config["system"]["val_interval"] = max_steps_value + 1
config["system"]["save_interval"] = max_steps_value + 1
config["system"]["save_final_checkpoint"] = False
config.setdefault("logging", {})["report_to"] = report_to
config.setdefault("resume", {})["checkpoint_path"] = None

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, sort_keys=False)
PY

    echo "[INFO] Generated RoboCOIN smoke config: $CONFIG_TO_RUN"
    if [ "$GENERATE_CONFIG_ONLY" = "1" ]; then
        exit 0
    fi
fi

python -c "import peft" >/dev/null 2>&1 || {
    echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
    exit 1
}

"$TORCHRUN_BIN" \
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
