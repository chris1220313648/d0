#!/bin/bash
#SBATCH --job-name=motus_lap_future_noise_eval
#SBATCH --partition=acd_u
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --chdir=/data/user/wsong890/user68/cjy/Motus

set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-/data/user/wsong890/user68/cjy/Motus}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${MOTUS_ROOT}/RoboTwin}"
POLICY_DIR="${POLICY_DIR:-${ROBOTWIN_ROOT}/policy/Motus}"
CONDA_ENV="${CONDA_ENV:-/data/user/wsong890/user68/conda_env/robotwin_motus}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${MOTUS_ROOT}/checkpoints/robotwin_lap_future_noise/robotwin_lap_future_video_noise/checkpoint_step_200000/pytorch_model}"
WAN_PATH="${WAN_PATH:-${MOTUS_ROOT}/pretrained_models/Wan2.2-TI2V-5B}"
VLM_PATH="${VLM_PATH:-${MOTUS_ROOT}/pretrained_models/Qwen3-VL-2B-Instruct}"

TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
TASKS_FILE="${TASKS_FILE:-tasks_all.txt}"
TASK_NAME="${TASK_NAME:-}"
SEED="${SEED:-42}"
TEST_NUM="${TEST_NUM:-100}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
POLICY_NAME="${POLICY_NAME:-Motus}"

INFERENCE_MODE="${INFERENCE_MODE:-history_flow}"
NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-10}"
HISTORY_ACTION_NOISE_STD="${HISTORY_ACTION_NOISE_STD:-0.02}"

if [ -n "${GPU_IDS:-}" ]; then
    IFS=',' read -ra GPU_ID_LIST <<< "$GPU_IDS"
else
    GPU_ID_LIST=()
fi

echo "Starting Motus LAP future-noise inference on RoboTwin at $(date)"

source /share/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if type module >/dev/null 2>&1; then
    module load cuda/12.8
    module load ffmpeg/6.0.1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$POLICY_DIR" ]; then
    echo "Error: Motus policy dir not found: $POLICY_DIR"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    echo "Set CHECKPOINT_PATH=/path/to/checkpoint/pytorch_model to override."
    exit 1
fi

if [ ! -f "$(dirname "$CHECKPOINT_PATH")/config.json" ]; then
    echo "Error: Checkpoint metadata not found: $(dirname "$CHECKPOINT_PATH")/config.json"
    echo "history_flow inference needs the config.json saved beside pytorch_model."
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT"

export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

LOG_DIR="${LOG_DIR:-${POLICY_DIR}/logs_future_noise_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

if [ -n "$TASK_NAME" ]; then
    tasks=("$TASK_NAME")
else
    TASKS_PATH="${POLICY_DIR}/${TASKS_FILE}"
    if [ ! -f "$TASKS_PATH" ]; then
        echo "Error: Tasks file not found: $TASKS_PATH"
        exit 1
    fi
    mapfile -t tasks < <(sed 's/\r$//' "$TASKS_PATH" | grep -v '^[[:space:]]*$')
fi

if [ "${#tasks[@]}" -eq 0 ]; then
    echo "Error: No tasks to evaluate."
    exit 1
fi

unique_task_count=$(printf '%s\n' "${tasks[@]}" | sort -u | wc -l)
if [ "$unique_task_count" -ne "${#tasks[@]}" ]; then
    echo "Error: Expected unique tasks, found ${#tasks[@]} entries but only $unique_task_count unique tasks"
    exit 1
fi

if [ "${#GPU_ID_LIST[@]}" -eq 0 ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPU_ID_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    else
        GPU_ID_LIST=(0)
    fi
fi

echo ""
echo "=== Inference Configuration ==="
echo "Motus Root: $MOTUS_ROOT"
echo "RoboTwin Root: $ROBOTWIN_ROOT"
echo "Policy Dir: $POLICY_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "WAN Path: $WAN_PATH"
echo "VLM Path: $VLM_PATH"
echo "Inference Mode: $INFERENCE_MODE"
echo "Inference Steps: $NUM_INFERENCE_TIMESTEPS"
echo "History Action Noise Std: $HISTORY_ACTION_NOISE_STD"
echo "Test Episodes: $TEST_NUM"
echo "Instruction Type: $INSTRUCTION_TYPE"
echo "Task Config: $TASK_CONFIG"
echo "Tasks: ${#tasks[@]}"
echo "GPUs: ${GPU_ID_LIST[*]}"
echo "Seed: $SEED"
echo "Log Dir: $LOG_DIR"
echo "==============================="

declare -A gpu_pid
for gpu_id in "${GPU_ID_LIST[@]}"; do
    gpu_pid[$gpu_id]=""
done

is_running() {
    [ -n "$1" ] && kill -0 "$1" 2>/dev/null
}

get_free_gpu() {
    while true; do
        for gpu_id in "${GPU_ID_LIST[@]}"; do
            if ! is_running "${gpu_pid[$gpu_id]}"; then
                echo "$gpu_id"
                return 0
            fi
        done
        sleep 2
    done
}

show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local bar_length=50
    local filled=$((percent * bar_length / 100))

    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%$((bar_length - filled))s" | tr ' ' ' '
    printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

pids=()
completed=0
total=${#tasks[@]}

echo ""
echo "Launching inference tasks..."

for task in "${tasks[@]}"; do
    gpu_id=$(get_free_gpu)
    log_file="${LOG_DIR}/${task}.log"

    echo "Task: $task | GPU: $gpu_id"

    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"

        PYTHONWARNINGS=ignore::UserWarning \
        python script/eval_policy.py \
            --config "policy/${POLICY_NAME}/deploy_policy.yml" \
            --overrides \
            --task_name "$task" \
            --task_config "$TASK_CONFIG" \
            --ckpt_setting "$CHECKPOINT_PATH" \
            --seed "$SEED" \
            --policy_name "$POLICY_NAME" \
            --instruction_type "$INSTRUCTION_TYPE" \
            --log_dir "$LOG_DIR" \
            --wan_path "$WAN_PATH" \
            --vlm_path "$VLM_PATH" \
            --inference_mode "$INFERENCE_MODE" \
            --test_num "$TEST_NUM" \
            --num_inference_timesteps "$NUM_INFERENCE_TIMESTEPS" \
            --history_action_noise_std "$HISTORY_ACTION_NOISE_STD" \
            > "$log_file" 2>&1

        exit_code=$?
        if [ "$exit_code" -eq 0 ]; then
            echo "Task $task completed successfully" >> "$log_file"
        else
            echo "Task $task failed with exit code $exit_code" >> "$log_file"
        fi
        exit "$exit_code"
    ) &

    pid=$!
    gpu_pid[$gpu_id]=$pid
    pids+=("$pid")
    sleep 1
done

echo ""
echo "Waiting for completion..."

failed_waits=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed_waits=$((failed_waits + 1))
    fi
    completed=$((completed + 1))
    show_progress "$completed" "$total"
done

echo ""
echo "All inference tasks finished."

summary="${LOG_DIR}/evaluation_summary.txt"
failed_tasks_file="${LOG_DIR}/failed_tasks.txt"
: > "$failed_tasks_file"

cat > "$summary" << EOF
Motus LAP Future-Noise Inference Summary
========================================
Date: $(date)
Host: $(hostname)
RoboTwin: $ROBOTWIN_ROOT
Checkpoint: $CHECKPOINT_PATH
WAN Path: $WAN_PATH
VLM Path: $VLM_PATH
Inference Mode: $INFERENCE_MODE
Inference Steps: $NUM_INFERENCE_TIMESTEPS
History Action Noise Std: $HISTORY_ACTION_NOISE_STD
Test Episodes Per Task: $TEST_NUM
Instruction Type: $INSTRUCTION_TYPE
Policy: $POLICY_NAME
Task Config: $TASK_CONFIG
Seed: $SEED
Total Tasks: $total
GPUs: ${GPU_ID_LIST[*]}

Task Results:
-------------
EOF

success=0
failed=0
scored=0
total_score=0

for task in "${tasks[@]}"; do
    log_file="${LOG_DIR}/${task}.log"

    if [ ! -f "$log_file" ]; then
        echo "  $task: LOG NOT FOUND" >> "$summary"
        failed=$((failed + 1))
        echo "$task" >> "$failed_tasks_file"
    elif grep -q "Task $task completed successfully" "$log_file" 2>/dev/null; then
        clean_rate_line=$(
            grep -i "Success rate:" "$log_file" \
                | tail -n 1 \
                | sed $'s/\033\\[[0-9;]*m//g'
        )
        score=$(printf '%s\n' "$clean_rate_line" \
            | sed -n 's/.*=>[[:space:]]*\([0-9.][0-9.]*\)%.*/\1/p')
        if [ -n "$score" ]; then
            echo "  $task: ${score}%" >> "$summary"
            total_score=$(awk -v total="$total_score" -v value="$score" \
                'BEGIN {printf "%.6f", total + value}')
            scored=$((scored + 1))
        else
            echo "  $task: COMPLETED (score unavailable)" >> "$summary"
        fi
        success=$((success + 1))
    else
        echo "  $task: FAILED" >> "$summary"
        failed=$((failed + 1))
        echo "$task" >> "$failed_tasks_file"
    fi
done

if [ "$scored" -gt 0 ]; then
    average_score=$(awk -v total="$total_score" -v count="$scored" \
        'BEGIN {printf "%.2f", total / count}')
else
    average_score="N/A"
fi

cat >> "$summary" << EOF

Summary Statistics:
-------------------
Completed: $success
Failed: $failed
Total: $total
Tasks With Scores: $scored
Average Task Success Rate: ${average_score}%
Completion Rate: $(awk "BEGIN {printf \"%.1f\", $success * 100.0 / $total}")%
Wait Failures: $failed_waits

Logs: $LOG_DIR
Failed Tasks: $failed_tasks_file
EOF

echo ""
echo "=== Summary ==="
echo "Completed: $success"
echo "Failed: $failed"
echo "Average task success rate: ${average_score}%"
echo "Completion rate: $(awk "BEGIN {printf \"%.1f\", $success * 100.0 / $total}")%"
echo "Summary: $summary"

if [ "$failed" -eq 0 ] && [ "$failed_waits" -eq 0 ]; then
    exit 0
fi

exit 1
