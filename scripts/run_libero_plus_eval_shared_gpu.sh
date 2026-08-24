#!/usr/bin/env bash
# Run task-sharded LIBERO-plus environments against one shared Motus server.

set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECKPOINT=""
SUITE="libero_spatial"
WORKERS=4
PORT=9883
GPU_ID=0
SIM_GPU_IDS=""
MAX_TASKS=-1
NUM_TRIALS=1
VIDEO_MODE="none"
MUJOCO_GL="egl"
OUTPUT_ROOT="${MOTUS_ROOT}/outputs/libero_plus_parallel_shared"
RUN_NAME=""
RESUME=0
DRY_RUN=0
SERVER_WAIT_SECONDS=600
MOTUS_PYTHON="${MOTUS_PYTHON:-/opt/conda/envs/motus/bin/python}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/opt/conda/envs/liberoplus/bin/python}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/nas/code/LIBERO-plus}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/.libero_plus}"

usage() {
    echo "Usage: $0 --checkpoint PATH [--task SUITE] [--workers 4] [--gpu-id 0] [--sim-gpu-ids 0,1,2,3]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
        --task|-t) SUITE="$2"; shift 2 ;;
        --workers|-w) WORKERS="$2"; shift 2 ;;
        --port|-p) PORT="$2"; shift 2 ;;
        --gpu-id|-g) GPU_ID="$2"; shift 2 ;;
        --sim-gpu-ids) SIM_GPU_IDS="$2"; shift 2 ;;
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
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be positive" >&2; exit 2; }
case "${SUITE}" in
    libero_spatial) SUITE_TASKS=2402 ;;
    libero_object) SUITE_TASKS=2518 ;;
    libero_goal) SUITE_TASKS=2591 ;;
    libero_10) SUITE_TASKS=2519 ;;
    *) echo "Unsupported suite: ${SUITE}" >&2; exit 2 ;;
esac
case "${VIDEO_MODE}" in all|failures|none) ;; *) echo "Invalid video mode" >&2; exit 2 ;; esac
case "${MUJOCO_GL}" in egl|osmesa) ;; *) echo "Invalid MuJoCo backend" >&2; exit 2 ;; esac

TASK_COUNT="${SUITE_TASKS}"
if (( MAX_TASKS > 0 && MAX_TASKS < TASK_COUNT )); then
    TASK_COUNT="${MAX_TASKS}"
fi
if (( WORKERS > TASK_COUNT )); then
    WORKERS="${TASK_COUNT}"
fi
SIM_GPUS=()
if [[ -n "${SIM_GPU_IDS}" ]]; then
    IFS=',' read -ra SIM_GPUS <<< "${SIM_GPU_IDS}"
    if (( ${#SIM_GPUS[@]} < WORKERS )); then
        echo "--sim-gpu-ids must provide at least one GPU per worker" >&2
        exit 2
    fi
fi

CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(basename "$(dirname "${CHECKPOINT}")")_${SUITE}_n${TASK_COUNT}_w${WORKERS}"
fi
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RESULTS_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"

if (( DRY_RUN == 1 )); then
    echo "mode=shared-gpu suite=${SUITE} task_count=${TASK_COUNT} workers=${WORKERS}"
    for ((worker=0; worker<WORKERS; worker++)); do
        start_task=$((worker * TASK_COUNT / WORKERS))
        end_task=$(((worker + 1) * TASK_COUNT / WORKERS))
        sim_gpu="${GPU_ID}"
        if (( ${#SIM_GPUS[@]} > 0 )); then sim_gpu="${SIM_GPUS[worker]}"; fi
        echo "worker=${worker} sim_gpu=${sim_gpu} policy_gpu=${GPU_ID} port=${PORT} tasks=[${start_task},${end_task})"
    done
    exit 0
fi
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

server_pid=""
client_pids=()
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    for pid in "${client_pids[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "Starting one shared Motus server on GPU ${GPU_ID}, port ${PORT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${MOTUS_ROOT}/scripts/serve_libero_plus.sh" "${CHECKPOINT}" "${PORT}" \
    >"${LOG_DIR}/server.log" 2>&1 &
server_pid=$!

ready=0
for ((i=0; i<SERVER_WAIT_SECONDS; i++)); do
    kill -0 "${server_pid}" 2>/dev/null || break
    if bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then ready=1; break; fi
    sleep 1
done
if (( ready == 0 )); then
    echo "Policy server did not become ready" >&2
    tail -100 "${LOG_DIR}/server.log" >&2 || true
    exit 1
fi

resume_args=()
if (( RESUME == 1 )); then resume_args+=(--resume); fi

for ((worker=0; worker<WORKERS; worker++)); do
    start_task=$((worker * TASK_COUNT / WORKERS))
    end_task=$(((worker + 1) * TASK_COUNT / WORKERS))
    shard_name="$(printf '%06d_%06d' "${start_task}" "${end_task}")"
    sim_gpu="${GPU_ID}"
    if (( ${#SIM_GPUS[@]} > 0 )); then sim_gpu="${SIM_GPUS[worker]}"; fi
    echo "worker=${worker} sim_gpu=${sim_gpu} tasks=[${start_task},${end_task}) port=${PORT}"
    (
        export CUDA_VISIBLE_DEVICES="${sim_gpu}"
        export LIBERO_HOME="${LIBERO_PLUS_ROOT}"
        export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
        export PYTHONPATH="${LIBERO_PLUS_ROOT}:${MOTUS_ROOT}:${PYTHONPATH:-}"
        export MUJOCO_GL="${MUJOCO_GL}"
        export PYOPENGL_PLATFORM="${MUJOCO_GL}"
        export TOKENIZERS_PARALLELISM=false
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1
        exec "${LIBERO_PLUS_PYTHON}" "${MOTUS_ROOT}/examples/libero_plus/eval_libero.py" \
            --host 127.0.0.1 --port "${PORT}" \
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
echo "Shared-GPU evaluation complete: ${RUN_DIR}"
