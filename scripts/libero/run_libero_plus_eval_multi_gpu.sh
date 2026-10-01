#!/usr/bin/env bash
# Run one task shard and one Motus server per GPU.

set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECKPOINT=""
SUITE="libero_spatial"
GPU_IDS="0,1,2,3"
BASE_PORT=9883
MAX_TASKS=-1
NUM_TRIALS=1
VIDEO_MODE="none"
MUJOCO_GL="egl"
OUTPUT_ROOT="${MOTUS_ROOT}/outputs/libero_plus_parallel_multigpu"
RUN_NAME=""
RESUME=0
DRY_RUN=0
SERVER_WAIT_SECONDS=600
MOTUS_PYTHON="${MOTUS_PYTHON:-/opt/conda/envs/motus/bin/python}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/opt/conda/envs/liberoplus/bin/python}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/nas/code/LIBERO-plus}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/.libero_plus}"

usage() {
    echo "Usage: $0 --checkpoint PATH [--task SUITE] [--gpu-ids 0,1,2,3]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
        --task|-t) SUITE="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; shift 2 ;;
        --max-tasks) MAX_TASKS="$2"; shift 2 ;;
        --num-trials|-n) NUM_TRIALS="$2"; shift 2 ;;
        --video-mode) VIDEO_MODE="$2"; shift 2 ;;
        --mujoco-gl) MUJOCO_GL="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "${CHECKPOINT}" ]] || { usage; exit 2; }
[[ -d "${CHECKPOINT}" || -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
case "${SUITE}" in
    libero_spatial) SUITE_TASKS=2402 ;;
    libero_object) SUITE_TASKS=2518 ;;
    libero_goal) SUITE_TASKS=2591 ;;
    libero_10) SUITE_TASKS=2519 ;;
    *) echo "Unsupported suite: ${SUITE}" >&2; exit 2 ;;
esac
case "${VIDEO_MODE}" in all|failures|none) ;; *) echo "Invalid video mode" >&2; exit 2 ;; esac
case "${MUJOCO_GL}" in egl|osmesa) ;; *) echo "Invalid MuJoCo backend" >&2; exit 2 ;; esac

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
WORKERS="${#GPUS[@]}"
(( WORKERS > 0 )) || { echo "No GPUs provided" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: ${gpu}" >&2; exit 2; }
done

TASK_COUNT="${SUITE_TASKS}"
if (( MAX_TASKS > 0 && MAX_TASKS < TASK_COUNT )); then TASK_COUNT="${MAX_TASKS}"; fi
if (( WORKERS > TASK_COUNT )); then echo "More GPUs than selected tasks" >&2; exit 2; fi

CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(basename "$(dirname "${CHECKPOINT}")")_${SUITE}_n${TASK_COUNT}_${WORKERS}gpu"
fi
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RESULTS_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"

if (( DRY_RUN == 1 )); then
    echo "mode=multi-gpu suite=${SUITE} task_count=${TASK_COUNT} workers=${WORKERS}"
    for ((worker=0; worker<WORKERS; worker++)); do
        start_task=$((worker * TASK_COUNT / WORKERS))
        end_task=$(((worker + 1) * TASK_COUNT / WORKERS))
        port=$((BASE_PORT + worker))
        echo "worker=${worker} gpu=${GPUS[$worker]} port=${port} tasks=[${start_task},${end_task})"
    done
    exit 0
fi
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

server_pids=()
client_pids=()
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    for pid in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
    done
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

for ((worker=0; worker<WORKERS; worker++)); do
    gpu="${GPUS[$worker]}"
    port=$((BASE_PORT + worker))
    echo "Starting server ${worker}: GPU=${gpu} port=${port}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        "${MOTUS_ROOT}/scripts/serve_libero_plus.sh" "${CHECKPOINT}" "${port}" \
        >"${LOG_DIR}/server_gpu${gpu}_port${port}.log" 2>&1 &
    server_pids+=("$!")
done

for ((worker=0; worker<WORKERS; worker++)); do
    port=$((BASE_PORT + worker))
    pid="${server_pids[$worker]}"
    ready=0
    for ((i=0; i<SERVER_WAIT_SECONDS; i++)); do
        kill -0 "${pid}" 2>/dev/null || break
        if bash -c ">/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then ready=1; break; fi
        sleep 1
    done
    (( ready == 1 )) || { echo "Server ${worker} failed to become ready" >&2; exit 1; }
done

resume_args=()
if (( RESUME == 1 )); then resume_args+=(--resume); fi

for ((worker=0; worker<WORKERS; worker++)); do
    gpu="${GPUS[$worker]}"
    port=$((BASE_PORT + worker))
    start_task=$((worker * TASK_COUNT / WORKERS))
    end_task=$(((worker + 1) * TASK_COUNT / WORKERS))
    shard_name="$(printf '%06d_%06d' "${start_task}" "${end_task}")"
    echo "GPU=${gpu} tasks=[${start_task},${end_task}) port=${port}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export LIBERO_HOME="${LIBERO_PLUS_ROOT}"
        export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
        export PYTHONPATH="${LIBERO_PLUS_ROOT}:${MOTUS_ROOT}:${PYTHONPATH:-}"
        export MUJOCO_GL="${MUJOCO_GL}"
        export PYOPENGL_PLATFORM="${MUJOCO_GL}"
        export TOKENIZERS_PARALLELISM=false
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1
        exec "${LIBERO_PLUS_PYTHON}" "${MOTUS_ROOT}/examples/libero_plus/eval_libero.py" \
            --host 127.0.0.1 --port "${port}" \
            --task-suite-name "${SUITE}" \
            --num-trials-per-task "${NUM_TRIALS}" \
            --max-tasks "${TASK_COUNT}" \
            --start-task "${start_task}" --end-task "${end_task}" --shard-output \
            --video-mode "${VIDEO_MODE}" \
            --output-dir "${RESULTS_DIR}" \
            "${resume_args[@]}"
    ) >"${LOG_DIR}/${SUITE}_${shard_name}.log" 2>&1 &
    client_pids+=("$!")
done

status=0
for pid in "${client_pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
done
if (( status != 0 )); then
    echo "At least one evaluation shard failed; inspect ${LOG_DIR}" >&2
    exit 1
fi

"${MOTUS_PYTHON}" "${MOTUS_ROOT}/examples/libero_plus/aggregate_shards.py" \
    --suite-dir "${RESULTS_DIR}/${SUITE}" \
    --expected-task-count "${TASK_COUNT}"
echo "Multi-GPU evaluation complete: ${RUN_DIR}"
