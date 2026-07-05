#!/bin/bash
#SBATCH --job-name=motus_clean
#SBATCH --partition=acd_u
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --chdir=/data/user/wsong890/user68/cjy/Motus
# Define your env settings here
# e.g., nccl, network, proxy, etc.
source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/wsong890/envs/motus
TASK="robotwin_clean"  # Define your task name here
CONFIG_FILE="configs/robotwin_clean.yaml"  # Define your dataset config path here

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
    --deepspeed configs/zero1.json \
    --config $CONFIG_FILE \
    --run_name $TASK \
    --report_to tensorboard \
    > $OUTPUT_DIR/train_clean.log 2>&1
