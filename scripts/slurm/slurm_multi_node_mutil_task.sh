#!/bin/bash
# SLURM script for multi-node distributed training

#SBATCH --job-name=motus_multi_node_mutil_task
#SBATCH --output=./logs/slurm_multi_%j.out
#SBATCH --error=./logs/slurm_multi_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=40
#SBATCH --mem=800GB
#SBATCH --partition=acd_u  # change here
#SBATCH --exclusive

echo "Starting multi-node job on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"

# Setup environment
PROJECT_ROOT="/data/user/wsong890/user68/cjy/Motus"
cd $PROJECT_ROOT

# Load modules and activate conda environment
module load cuda/12.8 || echo "Warning: Could not load CUDA module"
source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/wsong890/envs/motus
python -c "import peft" >/dev/null 2>&1 || { echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft"; exit 1; }

# Set environment variables
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export OMP_NUM_THREADS=8
# export CUDA_HOME=$CONDA_PREFIX

# Get node information
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
master_addr=$(echo "$nodes" | head -n 1)
export MASTER_ADDR=$master_addr

echo "NODELIST: $nodes"
echo "MASTER_ADDR: $master_addr"
echo "Current node index: $SLURM_NODEID"

# NCCL settings for multi-node
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_13:1,mlx5_16:1,mlx5_17:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=vlan0.2135
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=23
export NCCL_DEBUG=INFO

# Increase timeout for checkpoint saving (default is 600s/10min, set to 30min)
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# export NCCL_SOCKET_IFNAME=vlan0.2135
# export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=138
# export NCCL_NSOCKS_PERTHREAD=8
# export NCCL_SOCKET_NTHREADS=4
# export NCCL_NVLS_ENABLE=0
# export NCCL_NVLS_PLUGIN=1
# export NCCL_NVLS_LANES=2
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_IB_SPLIT_DATA_ON_QPS=1
# export NCCL_MIN_CTAS=32
# export NCCL_MAX_CTAS=128
# export NCCL_IB_RETRY_CNT=7
# export NCCL_MIN_NCHANNELS=64
# export NCCL_MAX_NCHANNELS=256
# export NCCL_NCHANNELS_PER_NET_PEER=32
# export NCCL_BUFFSIZE=33554432
# export NCCL_LL_BUFFSIZE=33554432
# export NCCL_P2P_NET_CHUNKSIZE=2097152
# export NCCL_P2P_LEVEL=nvl
# export NCCL_ALGO=nvlstree,ring
# export NCCL_LL_MAX_NCHANNELS=4
# export NCCL_CROSS_NIC=1
# export NCCL_IGNORE_CPU_AFFINITY=1
# export NCCL_SHM_DISABLE=0
# export NCCL_COLLNET_ENABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=3600
# export NCCL_IB_TIMEOUT=3600

# Create logs directory
mkdir -p logs

# Multi-dataset training configuration
TASK=${TASK:-"multidataset"}
CONFIG_FILE=${CONFIG_FILE:-"configs/multidataset_lap.yaml"}
RUN_NAME=${RUN_NAME:-"${TASK}_multidataset_lap_multi_node"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"configs/zero2.json"}
REPORT_TO=${REPORT_TO:-"tensorboard"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/motus-${TASK}"}
# Use job-id-derived port by default to reduce collision risk between concurrent jobs.
if [ -z "${MASTER_PORT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
    else
        MASTER_PORT=29500
    fi
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Multi-Node Training Configuration"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "Master addr: $master_addr"
echo "Master port: $MASTER_PORT"
echo "Task: $TASK"
echo "Config: $CONFIG_FILE"
echo "Run name: $RUN_NAME"
echo "DeepSpeed: $DEEPSPEED_CONFIG"
echo "Report to: $REPORT_TO"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

# Export configuration variables for worker script
export CONFIG_FILE
export RUN_NAME
export MASTER_PORT
export DEEPSPEED_CONFIG

# Multi-node distributed training - launch exactly one worker per allocated node.
# Explicitly disable packing to avoid multiple node-rank=0 workers on the same host.
srun \
    --nodes=${SLURM_JOB_NUM_NODES} \
    --ntasks=${SLURM_JOB_NUM_NODES} \
    --ntasks-per-node=1 \
    --distribution=block:block,NoPack \
    --exact \
    bash scripts/slurm/multi_node_worker.sh
# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --nproc_per_node=$SLURM_GPUS_ON_NODE \
#     --node_rank=$SLURM_NODEID \
#     --master_addr=$master_addr \
#     --master_port=$MASTER_PORT \
#     train/train.py \
#     --deepspeed ${DEEPSPEED_CONFIG:-configs/zero1.json} \
#     --config $CONFIG_FILE \
#     $(if [ -n "$RUN_NAME" ]; then echo "--run_name $RUN_NAME"; fi) \
#     --report_to ${REPORT_TO:-tensorboard}
#     > $OUTPUT_DIR/train_lap_multi_node.log 2>&1


echo "Training completed at $(date)"
