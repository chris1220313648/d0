#!/bin/bash
# One-click runner for RobotWin epos generation from raw dataset.

set -u

echo "Starting RobotWin epos generation at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/robotwin_generate_epos_from_raw.py"

RAW_ROOT="/data/user/wsong890/user68/cjy/Motus/data/robotwin_raw_dataset"
TARGET_ROOT="/data/user/wsong890/user68/cjy/Motus/data/robotwin_dataset"
SUBSETS="clean,randomized"
TASKS=""
OUTPUT_DIR_NAME="epos"
OVERWRITE=false
ALIGN_WITH_QPOS=true
LENGTH_MODE="trim"
EPISODE_LIMIT=0
QUAT_ORDER="wxyz"
WITHOUT_GRIPPER=false
USER_VERBOSE=false
PYTHON_CMD=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --raw-root PATH          Raw root (default: ${RAW_ROOT})
  --target-root PATH       Target root (default: ${TARGET_ROOT})
  --subsets STR            Comma-separated subsets (default: ${SUBSETS})
  --tasks STR              Comma-separated tasks (default: all tasks)
  --output-dir-name NAME   Output dir name under each task (default: ${OUTPUT_DIR_NAME})
  --overwrite              Overwrite existing epos/*.pt
  --no-align-with-qpos     Do not align epos length to qpos length
  --length-mode MODE       trim|strict (default: ${LENGTH_MODE})
  --episode-limit N        Episodes per task for debug (default: ${EPISODE_LIMIT})
  --quat-order ORDER       wxyz|xyzw (default: ${QUAT_ORDER})
  --without-gripper        Output without gripper columns (T x 12)
  --python PATH            Python executable to use
  --verbose                Enable verbose logs
  -h, --help               Show this help

Examples:
  # Full run (both clean + randomized)
  $(basename "$0")

  # Single task quick check
  $(basename "$0") --subsets clean --tasks adjust_bottle --episode-limit 1 --overwrite --verbose

  # If your quaternion order is xyzw
  $(basename "$0") --quat-order xyzw
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-root)
            RAW_ROOT="$2"; shift 2 ;;
        --target-root)
            TARGET_ROOT="$2"; shift 2 ;;
        --subsets)
            SUBSETS="$2"; shift 2 ;;
        --tasks)
            TASKS="$2"; shift 2 ;;
        --output-dir-name)
            OUTPUT_DIR_NAME="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=true; shift ;;
        --no-align-with-qpos)
            ALIGN_WITH_QPOS=false; shift ;;
        --length-mode)
            LENGTH_MODE="$2"; shift 2 ;;
        --episode-limit)
            EPISODE_LIMIT="$2"; shift 2 ;;
        --quat-order)
            QUAT_ORDER="$2"; shift 2 ;;
        --without-gripper)
            WITHOUT_GRIPPER=true; shift ;;
        --python)
            PYTHON_CMD="$2"; shift 2 ;;
        --verbose)
            USER_VERBOSE=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Error: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "Error: Python script not found: ${PY_SCRIPT}"
    exit 1
fi

if [[ -z "${PYTHON_CMD}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    else
        echo "Error: No Python interpreter found"
        exit 1
    fi
fi

if [[ ! -d "${RAW_ROOT}" ]]; then
    echo "Error: raw root not found: ${RAW_ROOT}"
    exit 1
fi

if [[ ! -d "${TARGET_ROOT}" ]]; then
    echo "Error: target root not found: ${TARGET_ROOT}"
    exit 1
fi

if [[ "${LENGTH_MODE}" != "trim" && "${LENGTH_MODE}" != "strict" ]]; then
    echo "Error: --length-mode must be trim or strict, got: ${LENGTH_MODE}"
    exit 1
fi

if [[ "${QUAT_ORDER}" != "wxyz" && "${QUAT_ORDER}" != "xyzw" ]]; then
    echo "Error: --quat-order must be wxyz or xyzw, got: ${QUAT_ORDER}"
    exit 1
fi

echo "Using Python: $(which "${PYTHON_CMD}" 2>/dev/null || echo "${PYTHON_CMD}")"
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Conda environment: ${CONDA_DEFAULT_ENV}"
fi

if ! "${PYTHON_CMD}" -c "import torch, h5py, numpy" >/dev/null 2>&1; then
    echo "Error: Missing required Python packages (need: torch, h5py, numpy)"
    exit 1
fi

echo "Configuration:"
echo "  raw_root: ${RAW_ROOT}"
echo "  target_root: ${TARGET_ROOT}"
echo "  subsets: ${SUBSETS}"
echo "  tasks: ${TASKS:-<all>}"
echo "  output_dir_name: ${OUTPUT_DIR_NAME}"
echo "  overwrite: ${OVERWRITE}"
echo "  align_with_qpos: ${ALIGN_WITH_QPOS}"
echo "  length_mode: ${LENGTH_MODE}"
echo "  episode_limit: ${EPISODE_LIMIT}"
echo "  quat_order: ${QUAT_ORDER}"
echo "  without_gripper: ${WITHOUT_GRIPPER}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/epos_backfill_${TIMESTAMP}.log"
echo "Log file: ${LOG_FILE}"

CMD=(
    "${PYTHON_CMD}" "${PY_SCRIPT}"
    --raw-root "${RAW_ROOT}"
    --target-root "${TARGET_ROOT}"
    --subsets "${SUBSETS}"
    --output-dir-name "${OUTPUT_DIR_NAME}"
    --length-mode "${LENGTH_MODE}"
    --episode-limit "${EPISODE_LIMIT}"
    --quat-order "${QUAT_ORDER}"
)

if [[ -n "${TASKS}" ]]; then
    CMD+=(--tasks "${TASKS}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
    CMD+=(--overwrite)
fi

if [[ "${ALIGN_WITH_QPOS}" == "false" ]]; then
    CMD+=(--no-align-with-qpos)
fi

if [[ "${WITHOUT_GRIPPER}" == "true" ]]; then
    CMD+=(--without-gripper)
fi

if [[ "${USER_VERBOSE}" == "true" ]]; then
    CMD+=(--verbose)
fi

echo "Executing:"
printf '  %q ' "${CMD[@]}"
echo

cd "${SCRIPT_DIR}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}

if [[ ${STATUS} -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "EPOS BACKFILL SUMMARY"
    echo "=========================================="
    echo "End time: $(date +%H:%M:%S)"
    echo "Target: ${TARGET_ROOT}"
    echo "Log file: ${LOG_FILE}"
    EPOS_COUNT=$(find "${TARGET_ROOT}" -path "*/${OUTPUT_DIR_NAME}/*.pt" | wc -l)
    echo "Total ${OUTPUT_DIR_NAME} files: ${EPOS_COUNT}"
    echo "EPOS backfill completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "EPOS BACKFILL FAILED"
    echo "=========================================="
    echo "Exit code: ${STATUS}"
    echo "Check log file for details: ${LOG_FILE}"
    echo "=========================================="
    exit ${STATUS}
fi

echo "Script completed at $(date)"

