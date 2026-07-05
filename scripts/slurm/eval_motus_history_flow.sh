#!/bin/bash
#SBATCH --job-name=motus_hist_eval
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/data/user/wsong890/user68/cjy/Motus/logs/motus_hist_eval_%j.out
#SBATCH --error=/data/user/wsong890/user68/cjy/Motus/logs/motus_hist_eval_%j.err
#SBATCH --time=04:00:00

set -o pipefail

PROJECT_ROOT="/data/user/wsong890/user68/cjy/Motus"
ROBOTWIN_ROOT="${PROJECT_ROOT}/RoboTwin"
POLICY_DIR="${ROBOTWIN_ROOT}/policy/Motus"
PATHS_CONFIG="${POLICY_DIR}/paths_config.yml"

TASK_NAME="${TASK_NAME:-$(grep -v '^[[:space:]]*$' "${POLICY_DIR}/tasks_all.txt" | head -n 1)}"
TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
SEED="${SEED:-42}"
TEST_NUM="${TEST_NUM:-100}"
INFERENCE_MODE="${INFERENCE_MODE:-history_flow}"
NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-5}"
HISTORY_ACTION_NOISE_STD="${HISTORY_ACTION_NOISE_STD:-0.02}"
RUN_TAG="${RUN_TAG:-baseline}"

read_yaml_value() {
    local key=$1
    grep "^${key}:" "$PATHS_CONFIG" \
        | sed 's/#.*//' \
        | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' \
        | tr -d '"' \
        | xargs
}

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$(read_yaml_value checkpoint_path)}"
WAN_PATH="${WAN_PATH:-$(read_yaml_value wan_path)}"
VLM_PATH="${VLM_PATH:-$(read_yaml_value vlm_path)}"
CONDA_ENV="${CONDA_ENV:-$(read_yaml_value conda_env)}"

RUN_ID="${SLURM_JOB_ID:-local}_${RUN_TAG}_n${TEST_NUM}_s${NUM_INFERENCE_TIMESTEPS}_noise${HISTORY_ACTION_NOISE_STD}"
RUN_LOG_DIR="${POLICY_DIR}/slurm_eval_logs/${RUN_ID}"
TASK_LOG="${RUN_LOG_DIR}/${TASK_NAME}.log"
RESULT_RECORD="${RUN_LOG_DIR}/result_summary.txt"

mkdir -p "$RUN_LOG_DIR"

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Task: $TASK_NAME"
echo "Task config: $TASK_CONFIG"
echo "Seed: $SEED"
echo "Test episodes: $TEST_NUM"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Inference mode: $INFERENCE_MODE"
echo "Inference steps: $NUM_INFERENCE_TIMESTEPS"
echo "History action noise std: $HISTORY_ACTION_NOISE_STD"
echo "Run log directory: $RUN_LOG_DIR"

for required_dir in "$ROBOTWIN_ROOT" "$CHECKPOINT_PATH" "$WAN_PATH" "$VLM_PATH" "$CONDA_ENV"; do
    if [ ! -d "$required_dir" ]; then
        echo "ERROR: required directory not found: $required_dir" | tee -a "$RESULT_RECORD"
        exit 1
    fi
done

if [[ "$CHECKPOINT_PATH" != *"_action/"* ]]; then
    echo "ERROR: expected the action checkpoint, got: $CHECKPOINT_PATH" | tee -a "$RESULT_RECORD"
    exit 1
fi

CHECKPOINT_CONFIG="$(dirname "$CHECKPOINT_PATH")/config.json"
python - "$CHECKPOINT_CONFIG" <<'PY'
import json
import math
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
if not config_path.is_file():
    raise SystemExit(f"checkpoint metadata not found: {config_path}")

with config_path.open("r", encoding="utf-8") as handle:
    config = json.load(handle)

common = config.get("common", {})
flow = config.get("flow_source", {})
chunk_size = int(
    common.get(
        "action_chunk_size",
        int(common.get("num_video_frames", 0))
        * int(common.get("video_action_freq_ratio", 0)),
    )
)
checks = {
    "flow_source.mode": (flow.get("mode"), "history"),
    "flow_source.video_mode": (flow.get("video_mode"), "gaussian"),
    "flow_source.history_length": (int(flow.get("history_length", -1)), 16),
    "action_chunk_size": (chunk_size, 16),
    "action_dim": (int(common.get("action_dim", -1)), 14),
}
errors = [
    f"{name}={actual!r}, expected {expected!r}"
    for name, (actual, expected) in checks.items()
    if actual != expected
]
if not math.isclose(float(flow.get("action_noise_std", -1)), 0.02):
    errors.append(
        f"flow_source.action_noise_std={flow.get('action_noise_std')!r}, expected 0.02"
    )
if errors:
    raise SystemExit("incompatible checkpoint metadata:\n  " + "\n  ".join(errors))

print(
    "Checkpoint metadata OK: action_dim=14, chunk=16, "
    "action_source=history, video_source=gaussian, noise=0.02"
)
PY

if [ $? -ne 0 ]; then
    exit 1
fi

module load cuda/12.8
module load ffmpeg/6.0.1
source /share/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8

cd "$ROBOTWIN_ROOT" || exit 1
START_EPOCH=$(date +%s)

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "policy/Motus/deploy_policy.yml" \
    --overrides \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --ckpt_setting "$CHECKPOINT_PATH" \
    --seed "$SEED" \
    --policy_name "Motus" \
    --instruction_type "unseen" \
    --log_dir "$RUN_LOG_DIR" \
    --wan_path "$WAN_PATH" \
    --vlm_path "$VLM_PATH" \
    --inference_mode "$INFERENCE_MODE" \
    --test_num "$TEST_NUM" \
    --num_inference_timesteps "$NUM_INFERENCE_TIMESTEPS" \
    --history_action_noise_std "$HISTORY_ACTION_NOISE_STD" \
    2>&1 | tee "$TASK_LOG"

eval_status=${PIPESTATUS[0]}
if [ "$eval_status" -ne 0 ]; then
    echo "ERROR: evaluation exited with status $eval_status" | tee -a "$RESULT_RECORD"
    exit "$eval_status"
fi

RESULT_FILE=$(find "${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/Motus/${TASK_CONFIG}" \
    -name _result.txt -type f -newermt "@${START_EPOCH}" -print 2>/dev/null \
    | sort | tail -n 1)

if [ -z "$RESULT_FILE" ] || [ ! -f "$RESULT_FILE" ]; then
    echo "ERROR: evaluation completed but no new _result.txt was found" | tee -a "$RESULT_RECORD"
    exit 2
fi

SUCCESS_RATE=$(tail -n 1 "$RESULT_FILE" | tr -d '[:space:]')
if ! [[ "$SUCCESS_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: invalid success rate in $RESULT_FILE: $SUCCESS_RATE" | tee -a "$RESULT_RECORD"
    exit 3
fi

SUCCESS_COUNT=$(awk -v rate="$SUCCESS_RATE" -v total="$TEST_NUM" \
    'BEGIN {printf "%d", rate * total + 0.5}')

{
    echo "Job ID: ${SLURM_JOB_ID:-none}"
    echo "Task: $TASK_NAME"
    echo "Episodes: $TEST_NUM"
    echo "Success: ${SUCCESS_COUNT}/${TEST_NUM}"
    echo "Success rate: $SUCCESS_RATE"
    echo "Result file: $RESULT_FILE"
    echo "Task log: $TASK_LOG"
    echo "Completed at: $(date)"
} | tee "$RESULT_RECORD"

echo "Job ended at $(date)"
