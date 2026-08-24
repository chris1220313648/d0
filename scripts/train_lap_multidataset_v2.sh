#!/bin/bash
set -euo pipefail

# Eight-GPU launcher for LAP multi-dataset v2 robot-only training.
# Override any of these from the shell when needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29514}"

TASK="${TASK:-multidataset_lap_v2}"
CONFIG_FILE="${CONFIG_FILE:-configs/multidataset_lap_v2.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_8gpu}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
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
SMOKE_AGIBOT_TASK="${SMOKE_AGIBOT_TASK:-ImitationLearning/CommercialSpaces/task_3400/313498_314085_split}"
SMOKE_INTERNDATA_TASK="${SMOKE_INTERNDATA_TASK:-sim_updated/articulation_tasks/franka/close_the_electriccooker}"
SMOKE_ROBOCOIN_TASK="${SMOKE_ROBOCOIN_TASK:-AI2_Alphabot_2_arrange_teaset}"
SMOKE_ROBOTWIN_TASK="${SMOKE_ROBOTWIN_TASK:-blocks_ranking_size}"
SMOKE_BRIDGE_TASK="${SMOKE_BRIDGE_TASK:-move_potato_cloth_table}"
SMOKE_DROID_TASK="${SMOKE_DROID_TASK:-put_the_orange_green_and_yellow_building_blocks_in_the_grey_pot}"
SMOKE_FRACTAL_TASK="${SMOKE_FRACTAL_TASK:-pick_coke_can_from_bottom_shelf_of_fridge}"
SMOKE_MAX_EPISODES="${SMOKE_MAX_EPISODES:-2}"

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

python -c "import peft" >/dev/null 2>&1 || {
    echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"

CONFIG_TO_RUN="$CONFIG_FILE"
LOG_FILE="$OUTPUT_DIR/train_lap_8gpu.log"

if [ "$SMOKE_TEST" = "1" ]; then
    CONFIG_TO_RUN="$OUTPUT_DIR/${TASK}_smoke.yaml"
    LOG_FILE="$OUTPUT_DIR/train_lap_8gpu_smoke.log"
    REPORT_TO="$SMOKE_REPORT_TO"
    python - \
        "$CONFIG_FILE" \
        "$CONFIG_TO_RUN" \
        "$SMOKE_MAX_STEPS" \
        "$SMOKE_BATCH_SIZE" \
        "$SMOKE_NUM_WORKERS" \
        "$SMOKE_REPORT_TO" \
        "$SMOKE_MAX_EPISODES" \
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
    agibot_task,
    interndata_task,
    robocoin_task,
    robotwin_task,
    bridge_task,
    droid_task,
    fractal_task,
) = sys.argv[1:15]

dst_path = Path(dst)
with open(src, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

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

config.setdefault("training", {})["max_steps"] = int(max_steps)
config["training"]["batch_size"] = int(batch_size)
config.setdefault("system", {})["num_workers"] = int(num_workers)
config["system"]["val_interval"] = int(max_steps) + 1
config["system"]["save_interval"] = int(max_steps) + 1
config["system"]["save_final_checkpoint"] = False
config.setdefault("logging", {})["report_to"] = report_to
config.setdefault("resume", {})["checkpoint_path"] = None

dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
    echo "[INFO] Smoke test enabled: config=$CONFIG_TO_RUN max_steps=$SMOKE_MAX_STEPS batch_size=$SMOKE_BATCH_SIZE num_workers=$SMOKE_NUM_WORKERS report_to=$SMOKE_REPORT_TO max_episodes=$SMOKE_MAX_EPISODES"
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
