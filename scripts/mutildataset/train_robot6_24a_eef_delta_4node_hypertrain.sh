#!/usr/bin/env bash
set -euo pipefail
# HyperTrain/Kubernetes launcher; follows train_lap_multidataset_v4_14a_hypertrain.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

TASK="${TASK:-multidataset_lap_robot6_24a_norm}"
CONFIG_FILE="${CONFIG_FILE:-${CONFIG:-configs/multidataset_lap_robot6_24a_norm.yaml}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_lap_hypertrain}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${RUN_NAME}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

export HF_HOME="${HF_HOME:-/root/nasbak/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
: "${MASTER_ADDR:?HyperTrain must provide MASTER_ADDR reachable by all nodes}"
: "${NODE_RANK:?HyperTrain must provide distinct NODE_RANK values 0..NNODES-1}"
MASTER_PORT="${MASTER_PORT:-29514}"
DIST_BACKEND="${DIST_BACKEND:-deepspeed}"
for name in NNODES NODE_RANK MASTER_PORT; do
    [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "Invalid distributed integer: $name=${!name}" >&2; exit 1; }
done
case "$NPROC_PER_NODE" in
    auto|gpu|cpu) ;; # Native torchrun values; resolve inside the activated motus environment.
    *) [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] && (( NPROC_PER_NODE >= 1 )) || {
        echo "Invalid NPROC_PER_NODE=$NPROC_PER_NODE (expected positive integer, auto, gpu or cpu)" >&2; exit 1;
    } ;;
esac
(( NNODES >= 1 && NODE_RANK < NNODES && MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || exit 1
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == localhost || "$MASTER_ADDR" == 127.* || "$MASTER_ADDR" == ::1 ]]; then
    echo '[ERROR] Multi-node MASTER_ADDR cannot be loopback.' >&2
    exit 1
fi
if [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
    WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
else
    WORLD_SIZE="$NNODES nodes x $NPROC_PER_NODE (resolved by torchrun)"
fi
[[ -f "$CONFIG_FILE" ]] || { echo "[ERROR] Config file not found: $CONFIG_FILE" >&2; exit 1; }
[[ -f "$DEEPSPEED_CONFIG" ]] || { echo "[ERROR] DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2; exit 1; }
CONFIG_TO_RUN="$CONFIG_FILE"
LOG_FILE="$OUTPUT_DIR/train_lap_hypertrain_node${NODE_RANK}.log"
command=(torchrun --nproc_per_node="$NPROC_PER_NODE" --nnodes="$NNODES"
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT"
    train/train.py --deepspeed "$DEEPSPEED_CONFIG" --config "$CONFIG_TO_RUN"
    --run_name "$RUN_NAME" --report_to "$REPORT_TO" --log_level "$LOG_LEVEL")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Environment: conda activate motus; config=$CONFIG_FILE; world_size=$WORLD_SIZE; log=$LOG_FILE"
    printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f /share/anaconda3/etc/profile.d/conda.sh ]]; then
    source /share/anaconda3/etc/profile.d/conda.sh
else
    echo '[ERROR] Could not find conda.sh' >&2
    exit 1
fi
conda activate motus
python -c 'import peft' >/dev/null 2>&1 || {
    echo '[ERROR] peft is required. Please install it in the motus environment.' >&2; exit 1;
}
mkdir -p "$OUTPUT_DIR"
[[ ! -e "$LOG_FILE" ]] || { echo "[ERROR] Existing log: $LOG_FILE; select a fresh OUTPUT_DIR." >&2; exit 1; }
exec > >(tee "$LOG_FILE") 2>&1
printf '%s\n' '==========================================' 'HyperTrain seven-source 24D training' \
    "Time: $(date)" "Host: ${HOSTNAME:-$(hostname)}" "Project root: $PROJECT_ROOT" \
    "NNODES: $NNODES" "NPROC_PER_NODE: $NPROC_PER_NODE" "WORLD_SIZE: $WORLD_SIZE" \
    "NODE_RANK: $NODE_RANK" "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" \
    "MASTER_ADDR: $MASTER_ADDR" "MASTER_PORT: $MASTER_PORT" "DIST_BACKEND: $DIST_BACKEND" \
    "Config: $CONFIG_TO_RUN" "DeepSpeed: $DEEPSPEED_CONFIG" "Run name: $RUN_NAME" \
    "Log file: $LOG_FILE" "HF_HOME: $HF_HOME" "Python: $(command -v python)"
set +e
"${command[@]}"
result=$?
printf '%s\n' "$result" > "$OUTPUT_DIR/exit_code_node${NODE_RANK}"
exit "$result"
