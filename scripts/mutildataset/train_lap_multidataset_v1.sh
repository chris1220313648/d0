#!/bin/bash
set -euo pipefail

# Four-GPU launcher for LAP multi-dataset v1 training.
# Override any of these from the shell when needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29513}"

TASK="${TASK:-multidataset_lap_v1}"
CONFIG_FILE="${CONFIG_FILE:-configs/multidataset_lap_v1.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_4gpu}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-5}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
SMOKE_REPORT_TO="${SMOKE_REPORT_TO:-none}"
SMOKE_AGIBOT_TASK="${SMOKE_AGIBOT_TASK:-ImitationLearning/CommercialSpaces/task_3405/389111_389369_split}"
SMOKE_EGOVERSE_ROWS="${SMOKE_EGOVERSE_ROWS:-8}"
SMOKE_IMAGEQA_JSON="${SMOKE_IMAGEQA_JSON:-/root/nas/xicheng/d0_handoff_imageqa/json/imageqa_smoke_256.json}"
SMOKE_IMAGEQA_ROOT="${SMOKE_IMAGEQA_ROOT:-/root/nas/qa_data/processed_qwen_vl}"
SMOKE_IMAGEQA_MAX_SAMPLES="${SMOKE_IMAGEQA_MAX_SAMPLES:-16}"

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
    python - \
        "$CONFIG_FILE" \
        "$CONFIG_TO_RUN" \
        "$SMOKE_MAX_STEPS" \
        "$SMOKE_BATCH_SIZE" \
        "$SMOKE_NUM_WORKERS" \
        "$SMOKE_REPORT_TO" \
        "$SMOKE_AGIBOT_TASK" \
        "$SMOKE_EGOVERSE_ROWS" \
        "$SMOKE_IMAGEQA_JSON" \
        "$SMOKE_IMAGEQA_ROOT" \
        "$SMOKE_IMAGEQA_MAX_SAMPLES" <<'PY'
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
    agibot_task,
    egoverse_rows,
    imageqa_json,
    imageqa_root,
    imageqa_max_samples,
) = sys.argv[1:12]

dst_path = Path(dst)
with open(src, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

datasets = config.get("dataset", {}).get("datasets", [])
agibot_datasets = [item for item in datasets if item.get("type") == "lerobot_agibot"]
egoverse_datasets = [item for item in datasets if item.get("type") == "egoverse_trimodal"]
imageqa_datasets = [item for item in datasets if item.get("type") == "image_qa"]
if not agibot_datasets:
    raise ValueError("No lerobot_agibot dataset found in smoke source config")
if not egoverse_datasets:
    raise ValueError("No egoverse_trimodal dataset found in smoke source config")
if not imageqa_datasets:
    raise ValueError("No image_qa dataset found in smoke source config")

agibot = agibot_datasets[0]
agibot["task_mode"] = "multi"
agibot["task_name"] = agibot_task
agibot["max_episodes"] = None
agibot.setdefault("params", {}).setdefault("task_discovery", {})["enabled"] = False

egoverse = egoverse_datasets[0]
source_manifest = Path(egoverse.get("train_manifest", ""))
if not source_manifest.exists():
    raise FileNotFoundError(f"EgoVerse train_manifest not found: {source_manifest}")
short_manifest = dst_path.parent / "egoverse_train_short.jsonl"
rows = []
with source_manifest.open("r", encoding="utf-8") as f:
    for _, line in zip(range(int(egoverse_rows)), f):
        if line.strip():
            rows.append(line)
if not rows:
    raise ValueError(f"No rows found in EgoVerse train_manifest: {source_manifest}")
short_manifest.write_text("".join(rows), encoding="utf-8")
egoverse["train_manifest"] = str(short_manifest.resolve())
egoverse["val_manifest"] = str(short_manifest.resolve())
egoverse["manifest"] = str(short_manifest.resolve())
egoverse["max_samples"] = len(rows)
egoverse["image_aug"] = False

imageqa_json_path = Path(imageqa_json)
imageqa_root_path = Path(imageqa_root)
if not imageqa_json_path.exists():
    raise FileNotFoundError(f"ImageQA smoke json not found: {imageqa_json_path}")
if not imageqa_root_path.exists():
    raise FileNotFoundError(f"ImageQA image_root not found: {imageqa_root_path}")
imageqa = imageqa_datasets[0]
imageqa["json_path"] = str(imageqa_json_path)
imageqa["image_root"] = str(imageqa_root_path)
imageqa["max_samples"] = int(imageqa_max_samples)
imageqa["shuffle"] = False

config.setdefault("training", {})["max_steps"] = int(max_steps)
config["training"]["batch_size"] = int(batch_size)
config.setdefault("system", {})["num_workers"] = int(num_workers)
config["system"]["val_interval"] = int(max_steps) + 1
config["system"]["save_interval"] = int(max_steps) + 1
config["system"]["save_final_checkpoint"] = False
config.setdefault("logging", {})["report_to"] = report_to

dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
    echo "[INFO] Smoke test enabled: config=$CONFIG_TO_RUN max_steps=$SMOKE_MAX_STEPS batch_size=$SMOKE_BATCH_SIZE num_workers=$SMOKE_NUM_WORKERS report_to=$SMOKE_REPORT_TO agibot_task=$SMOKE_AGIBOT_TASK imageqa_json=$SMOKE_IMAGEQA_JSON imageqa_max_samples=$SMOKE_IMAGEQA_MAX_SAMPLES"
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
