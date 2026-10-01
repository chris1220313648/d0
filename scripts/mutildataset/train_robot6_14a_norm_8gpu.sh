#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -o robot6_%j.log
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${ROBOT6_DEPS:-/root/nasbak/cjy/robot_raw/motus_robot6_python_deps}:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export NNODES="${NNODES:-1}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ "$NNODES" != 1 ]]; then
    : "${MASTER_ADDR:?HyperTrain must provide MASTER_ADDR for multi-node training}"
    : "${NODE_RANK:?HyperTrain must provide NODE_RANK (0..NNODES-1)}"
    if [[ "$MASTER_ADDR" == localhost || "$MASTER_ADDR" == 127.* || "$MASTER_ADDR" == ::1 ]]; then
        echo 'Multi-node MASTER_ADDR must be reachable by all nodes, not loopback.' >&2
        exit 1
    fi
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-29537}" NODE_RANK="${NODE_RANK:-0}"
for number in "$NNODES" "$NPROC_PER_NODE" "$NODE_RANK" "$MASTER_PORT"; do
    [[ "$number" =~ ^[0-9]+$ ]] || { echo "Invalid distributed integer: $number" >&2; exit 1; }
done
(( NNODES >= 1 && NPROC_PER_NODE >= 1 && NODE_RANK < NNODES && MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || exit 1
PYTHON="${PYTHON:-/opt/conda/bin/python}"
SMOKE="${SMOKE:-1}"
export SMOKE
export CONFIG="${CONFIG:-configs/multidataset_lap_robot6_14a_norm.yaml}"
if [[ "$SMOKE" == 1 ]]; then
    DEFAULT_RUN_NAME=robot6_8gpu_bs5_smoke100
    DEFAULT_OUTPUT_DIR=outputs/robot6_14a_norm_smoke
else
    DEFAULT_RUN_NAME=robot6_8gpu_bs5
    DEFAULT_OUTPUT_DIR=outputs/robot6_14a_norm_train
fi
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
export RUN_NAME
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
NODE_OUTPUT_DIR="$OUTPUT_DIR"
if (( NNODES > 1 )); then NODE_OUTPUT_DIR="$OUTPUT_DIR/node${NODE_RANK}"; fi
export NODE_OUTPUT_DIR
command=("$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --nnodes="$NNODES" --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    train/train.py --config "$NODE_OUTPUT_DIR/resolved_config.yaml"
    --deepspeed "${DEEPSPEED_CONFIG:-configs/zero2.json}"
    --run_name "$RUN_NAME" --report_to "${REPORT_TO:-tensorboard}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if [[ -e "$NODE_OUTPUT_DIR/train.log" ]]; then
    echo "Existing run found in $OUTPUT_DIR; select a fresh OUTPUT_DIR." >&2
    exit 1
fi
if compgen -G "checkpoints/resolved_config/$RUN_NAME/checkpoint_step_*" > /dev/null; then
    echo "Existing checkpoints for $RUN_NAME; select a fresh RUN_NAME." >&2
    exit 1
fi
mkdir -p "$NODE_OUTPUT_DIR"
export OUTPUT_DIR
"$PYTHON" - <<'CHECK'
import os, json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.motus import Motus
from data.lerobot.lerobot_sim_dataset import LeRobotSimDataset
from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
assert torch.cuda.device_count() == int(os.environ['NPROC_PER_NODE']), 'Visible GPU count differs from NPROC_PER_NODE'
assert FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE, 'Install ABI-matched FlashAttention first'
c = OmegaConf.load(os.environ['CONFIG'])
paths = [c.model.wan.config_path, c.model.wan.checkpoint_path, c.model.wan.vae_path,
         c.model.vlm.checkpoint_path, c.finetune.checkpoint_path]
for ds in c.dataset.datasets:
    paths.append(ds.dataset_dir)
    if ds.type == 'lerobot_sim': paths.append(ds.params.stats_path)
for value in paths:
    if value and not Path(value).exists(): raise FileNotFoundError(value)
print(f"Node {os.environ['NODE_RANK']} preflight passed; global batch="
      f"{int(os.environ['NNODES']) * int(os.environ['NPROC_PER_NODE']) * c.training.batch_size * c.training.gradient_accumulation_steps}", flush=True)
CHECK
"$PYTHON" - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
c=OmegaConf.load(os.environ['CONFIG'])
if os.environ.get('SMOKE','1')=='1':
 c.training.max_steps=100
 c.system.val_interval=100
 c.system.save_interval=100
 c.system.save_final_checkpoint=True
 c.system.audit_batches=True
 c.system.audit_dir=os.environ['OUTPUT_DIR']
c.logging.run_name=os.environ['RUN_NAME']
OmegaConf.save(c,Path(os.environ['NODE_OUTPUT_DIR'])/'resolved_config.yaml')
PY
set +e
"${command[@]}" > "$NODE_OUTPUT_DIR/train.log" 2>&1
result=$?
printf '%s\n' "$result" > "$NODE_OUTPUT_DIR/exit_code"
exit "$result"
