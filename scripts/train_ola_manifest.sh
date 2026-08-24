#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/nas/code/d0}"
MANIFEST="${1:?Usage: train_ola_manifest.sh MANIFEST [RUN_NAME] [--episode-fraction FRACTION]}"
shift
RUN_NAME_ARG=""
EPISODE_FRACTION=""
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    RUN_NAME_ARG="$1"
    shift
fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --episode-fraction)
            [ "$#" -ge 2 ] || { echo "[ERROR] --episode-fraction needs a value" >&2; exit 2; }
            EPISODE_FRACTION="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
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

GENERATOR_ARGS=(--manifest "$MANIFEST" --check-artifacts)
if [ -n "$EPISODE_FRACTION" ]; then
    GENERATOR_ARGS+=(--episode-fraction "$EPISODE_FRACTION")
fi
GENERATED_JSON="$(python scripts/build_ola_manifest_config.py "${GENERATOR_ARGS[@]}")"
readarray -t GENERATED < <(python -c 'import json,sys; x=json.loads(sys.argv[1]); print(x["name"]); print(x["config"])' "$GENERATED_JSON")
DATASET_NAME="${GENERATED[0]}"
CONFIG_FILE="${GENERATED[1]}"
RUN_NAME="${RUN_NAME_ARG:-${RUN_NAME:-${DATASET_NAME}_hisinit_future_noise}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
REPORT_TO="${REPORT_TO:-wandb}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET_NAME}/${RUN_NAME}}"
CHECKPOINT_DIR="checkpoints/${DATASET_NAME}/${RUN_NAME}"
export OUTPUT_DIR

if [ -e "$CHECKPOINT_DIR" ] && [ "${ALLOW_EXISTING:-0}" != "1" ]; then
    echo "[ERROR] Checkpoint target already exists: $CHECKPOINT_DIR" >&2
    echo "Set ALLOW_EXISTING=1 only when overwriting/resuming is intentional." >&2
    exit 1
fi
python -c "import peft" >/dev/null 2>&1 || {
    echo "[ERROR] peft is required for VLM LoRA. Please run: pip install peft" >&2
    exit 1
}
mkdir -p "$OUTPUT_DIR"
echo "[INFO] manifest=$MANIFEST config=$CONFIG_FILE run=$RUN_NAME episode_fraction=${EPISODE_FRACTION:-full} subset_seed=42"

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config "$CONFIG_FILE" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO" \
    > "$OUTPUT_DIR/train_lap.log" 2>&1
