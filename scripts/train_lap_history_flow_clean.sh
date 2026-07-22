#!/bin/bash
#SBATCH --job-name=motus_lap_history_flow_clean
#SBATCH --partition=acd_u
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --chdir=/data/user/wsong890/user68/cjy/Motus
# Train RoboTwin LAP with current-frame/history-qpos flow sources on the clean config.
cd /data/user/wsong890/user68/cjy/Motus
source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/wsong890/envs/motus
python -c "import peft" >/dev/null 2>&1 || { echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft"; exit 1; }

TASK="robotwin_clean"
CONFIG_FILE="configs/robotwin_lap_history_flow_clean.yaml"

export OUTPUT_DIR="outputs/motus-${TASK}"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29501 \
    train/train.py \
    --deepspeed configs/zero2.json \
    --config $CONFIG_FILE \
    --run_name ${TASK}_lap_history_flow_clean \
    --report_to tensorboard \
    > $OUTPUT_DIR/train_lap_history_flow_action.log 2>&1
