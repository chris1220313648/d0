#!/usr/bin/env bash
set -euo pipefail

# HyperTrain/Kubernetes multi-node smoke test for LAP multi-dataset v2.
# Run this same script on every pod. HyperTrain should inject the distributed
# environment variables listed below.

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

TASK="${TASK:-multidataset_lap_v2_smoke}"
CONFIG_FILE="${CONFIG_FILE:-configs/multidataset_lap_v2.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_hypertrain}"
REPORT_TO="${REPORT_TO:-none}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

HF_HOME="${HF_HOME:-/root/nasbak/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_HUB_OFFLINE HF_DATASETS_OFFLINE

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-4}"
DIST_BACKEND="${DIST_BACKEND:-deepspeed}"
WORLD_SIZE="${NNODES}*${NPROC_PER_NODE}"

SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-100}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
SMOKE_MAX_EPISODES="${SMOKE_MAX_EPISODES:-2}"
SMOKE_AGIBOT_TASK="${SMOKE_AGIBOT_TASK:-ImitationLearning/CommercialSpaces/task_3400/313498_314085_split}"
SMOKE_INTERNDATA_TASK="${SMOKE_INTERNDATA_TASK:-sim_updated/articulation_tasks/franka/close_the_electriccooker}"
SMOKE_ROBOCOIN_TASK="${SMOKE_ROBOCOIN_TASK:-AI2_Alphabot_2_arrange_teaset}"
SMOKE_ROBOTWIN_TASK="${SMOKE_ROBOTWIN_TASK:-blocks_ranking_size}"
SMOKE_BRIDGE_TASK="${SMOKE_BRIDGE_TASK:-move_potato_cloth_table}"
SMOKE_DROID_TASK="${SMOKE_DROID_TASK:-put_the_orange_green_and_yellow_building_blocks_in_the_grey_pot}"
SMOKE_FRACTAL_TASK="${SMOKE_FRACTAL_TASK:-pick_coke_can_from_bottom_shelf_of_fridge}"
GENERATE_CONFIG_ONLY="${GENERATE_CONFIG_ONLY:-0}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

CONFIG_DIR="${TMPDIR:-/tmp}/motus-${TASK}-node${NODE_RANK}"
CONFIG_TO_RUN="$CONFIG_DIR/${TASK}.yaml"
mkdir -p "$CONFIG_DIR"

python - \
    "$CONFIG_FILE" \
    "$CONFIG_TO_RUN" \
    "$SMOKE_MAX_STEPS" \
    "$SMOKE_BATCH_SIZE" \
    "$SMOKE_NUM_WORKERS" \
    "$REPORT_TO" \
    "$SMOKE_MAX_EPISODES" \
    "$CHECKPOINT_DIR" \
    "$SMOKE_AGIBOT_TASK" \
    "$SMOKE_INTERNDATA_TASK" \
    "$SMOKE_ROBOCOIN_TASK" \
    "$SMOKE_ROBOTWIN_TASK" \
    "$SMOKE_BRIDGE_TASK" \
    "$SMOKE_DROID_TASK" \
    "$SMOKE_FRACTAL_TASK" <<'PY'
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
    checkpoint_dir,
    agibot_task,
    interndata_task,
    robocoin_task,
    robotwin_task,
    bridge_task,
    droid_task,
    fractal_task,
) = sys.argv[1:16]

with Path(src).open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

datasets = config.get("dataset", {}).get("datasets", [])
by_name = {item.get("name"): item for item in datasets}
required = {
    "agibot_lerobot_split",
    "bridge_pt_v5",
    "droid_pt_v5",
    "fractal_pt_v5",
    "robocoin_lerobot",
    "interndata_a1",
    "robotwin",
}
missing = sorted(required.difference(by_name))
if missing:
    raise ValueError(f"Smoke source config missing datasets: {missing}")


def disable_discovery(item):
    params = item.setdefault("params", {})
    if "task_discovery" in params:
        params["task_discovery"]["enabled"] = False


def set_lerobot_task(name, task):
    item = by_name[name]
    item["task_mode"] = "multi"
    item["task_name"] = task
    item["max_episodes"] = int(max_episodes)
    disable_discovery(item)


def set_single_task(name, task, data_mode=None):
    item = by_name[name]
    item["task_mode"] = "single"
    item["task_name"] = task
    item["max_episodes"] = int(max_episodes)
    if data_mode is not None:
        item["data_mode"] = data_mode
    disable_discovery(item)


set_lerobot_task("agibot_lerobot_split", agibot_task)
set_lerobot_task("interndata_a1", interndata_task)
set_lerobot_task("robocoin_lerobot", robocoin_task)
set_single_task("robotwin", robotwin_task, data_mode="randomized")
set_single_task("bridge_pt_v5", bridge_task, data_mode="train")
set_single_task("droid_pt_v5", droid_task, data_mode="train")
set_single_task("fractal_pt_v5", fractal_task, data_mode="train")

max_steps_value = int(max_steps)
config.setdefault("training", {})["max_steps"] = max_steps_value
config["training"]["batch_size"] = int(batch_size)
config.setdefault("system", {})["checkpoint_dir"] = checkpoint_dir
config["system"]["num_workers"] = int(num_workers)
config["system"]["val_interval"] = max_steps_value + 1
config["system"]["save_interval"] = max_steps_value
config["system"]["save_final_checkpoint"] = False
config.setdefault("logging", {})["report_to"] = report_to
config.setdefault("resume", {})["checkpoint_path"] = None

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, sort_keys=False)
PY

echo "[INFO] Generated smoke config: $CONFIG_TO_RUN"

if [ "$GENERATE_CONFIG_ONLY" = "1" ]; then
    exit 0
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "[ERROR] DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2
    exit 1
fi

python -c "import peft" >/dev/null 2>&1 || {
    echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train_lap_hypertrain_smoke_node${NODE_RANK}.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "=========================================="
echo "HyperTrain LAP Multi-Dataset v2 Smoke Test"
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
echo "Checkpoint root: $CHECKPOINT_DIR"
echo "Log file: $LOG_FILE"
echo "Max steps: $SMOKE_MAX_STEPS"
echo "Batch size per process: $SMOKE_BATCH_SIZE"
echo "Max episodes per dataset: $SMOKE_MAX_EPISODES"
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
