#!/bin/bash
# Define your env settings here 
# e.g., nccl, network, proxy, etc.
# export CUDA_VISIBLE_DEVICES=6,7
source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/wsong890/envs/motus
python -c "import peft" >/dev/null 2>&1 || { echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft"; exit 1; }
TASK="robotwin"  # Define your task name here
CONFIG_FILE="configs/robotwin_unfrozen.yaml"  # Define your dataset config path here

export OUTPUT_DIR="outputs/motus-${TASK}" # Define your output directory here

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Single-node training with torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train/train.py \
    --deepspeed configs/zero2.json \
    --config $CONFIG_FILE \
    --run_name ${TASK}_vlm \
    --report_to tensorboard \
    > $OUTPUT_DIR/train_debug.log 2>&1
