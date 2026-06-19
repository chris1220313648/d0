#!/bin/bash

set -eo pipefail

log_fn() {
    mkdir -p logs/fsdp
    LOG_FILE="logs/fsdp/train_node_${NODE_RANK}.log"
}

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-"auto"}
NNODES=${NNODES:-1}
DIST_BACKEND=${DIST_BACKEND:-"deepspeed"}
WORLD_SIZE=${NNODES}*${NPROC_PER_NODE}

echo "==========TRAINING ENV =========="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "NNODES: $NNODES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "DIST_BACKEND: $DIST_BACKEND"
echo "================================"

# 需要提前下载模型和数据集，并写在ENV中
DATASET_PATH=${DATASET_PATH:-"data/Tiny_TCM"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/userdata/llms/Qwen/Qwen3-4B"}

LOG_FILE=$(log_fn)

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    main.py \
    --distributed_backend $DIST_BACKEND \
    --dataset_path $DATASET_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --preprocessing_num_workers 2 \
    --preprocessing_batch_size 80 \
    --max_samples 400 \
    --num_epochs 1 \
    --cutoff_len 128 \
    --save_strategy no \
    --per_device_batch_size 4 \
    --output_dir ./outputs/${DIST_BACKEND} 2>&1 | tee -a $LOG_FILE
