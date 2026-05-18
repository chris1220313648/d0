#!/bin/bash
# One-click runner for Bridge epos generation from RLDS action.

set -u

echo "Starting Bridge epos generation at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/bridge_generate_epos_from_raw.py"

PYTHON_CMD="/data/user/wsong890/envs/motus/bin/python"
CONFIG_PATH="${SCRIPT_DIR}/config_bridge_convert.yml"
SOURCE_ROOT=""
OUTPUT_ROOT=""
SPLITS="train test"
TASK_GROUPING=""
EPOS_DIR_NAME="epos"
REQUIRE_EXISTING_QPOS=true
ALIGN_WITH_QPOS=true
LENGTH_MODE="trim"
OVERWRITE=true
MAX_EPISODES_PER_SPLIT=0
USER_VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python PATH                   Python executable (default: ${PYTHON_CMD})
  --config PATH                   Config yaml path (default: ${CONFIG_PATH})
  --source-root PATH              Override source_root from config
  --output-root PATH              Override output_root from config
  --splits "train test"           Splits to process (default: "${SPLITS}")
  --task-grouping MODE            normalized_instruction|exact_instruction
  --epos-dir-name NAME            Epos dir name (default: ${EPOS_DIR_NAME})
  --no-require-existing-qpos      Allow writing when qpos file is missing
  --no-align-with-qpos            Disable qpos length alignment
  --length-mode MODE              trim|strict (default: ${LENGTH_MODE})
  --no-overwrite                  Skip existing epos files
  --max-episodes-per-split N      0 means full split (default: ${MAX_EPISODES_PER_SPLIT})
  --verbose                       Enable verbose logs
  -h, --help                      Show help

Examples:
  $(basename "$0")
  $(basename "$0") --splits "train" --max-episodes-per-split 50 --no-overwrite
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --source-root) SOURCE_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --task-grouping) TASK_GROUPING="$2"; shift 2 ;;
        --epos-dir-name) EPOS_DIR_NAME="$2"; shift 2 ;;
        --no-require-existing-qpos) REQUIRE_EXISTING_QPOS=false; shift ;;
        --no-align-with-qpos) ALIGN_WITH_QPOS=false; shift ;;
        --length-mode) LENGTH_MODE="$2"; shift 2 ;;
        --no-overwrite) OVERWRITE=false; shift ;;
        --max-episodes-per-split) MAX_EPISODES_PER_SPLIT="$2"; shift 2 ;;
        --verbose) USER_VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
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
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: Config file not found: ${CONFIG_PATH}"
    exit 1
fi
if [[ "${LENGTH_MODE}" != "trim" && "${LENGTH_MODE}" != "strict" ]]; then
    echo "Error: --length-mode must be trim or strict"
    exit 1
fi
if [[ -n "${TASK_GROUPING}" && "${TASK_GROUPING}" != "normalized_instruction" && "${TASK_GROUPING}" != "exact_instruction" ]]; then
    echo "Error: --task-grouping must be normalized_instruction or exact_instruction"
    exit 1
fi

if ! "${PYTHON_CMD}" -c "import torch, numpy, yaml, tensorflow, tensorflow_datasets" >/dev/null 2>&1; then
    echo "Error: Missing required packages in ${PYTHON_CMD} (need torch, numpy, pyyaml, tensorflow, tensorflow_datasets)"
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/bridge_epos_from_raw_${TIMESTAMP}.log"

echo "Configuration:"
echo "  python: ${PYTHON_CMD}"
echo "  config: ${CONFIG_PATH}"
echo "  source_root: ${SOURCE_ROOT:-<from config>}"
echo "  output_root: ${OUTPUT_ROOT:-<from config>}"
echo "  splits: ${SPLITS}"
echo "  task_grouping: ${TASK_GROUPING:-<from config>}"
echo "  epos_dir_name: ${EPOS_DIR_NAME}"
echo "  require_existing_qpos: ${REQUIRE_EXISTING_QPOS}"
echo "  align_with_qpos: ${ALIGN_WITH_QPOS}"
echo "  length_mode: ${LENGTH_MODE}"
echo "  overwrite: ${OVERWRITE}"
echo "  max_episodes_per_split: ${MAX_EPISODES_PER_SPLIT}"
echo "  log_file: ${LOG_FILE}"

CMD=(
    "${PYTHON_CMD}" "${PY_SCRIPT}"
    --config "${CONFIG_PATH}"
    --splits ${SPLITS}
    --epos_dir_name "${EPOS_DIR_NAME}"
    --length_mode "${LENGTH_MODE}"
    --max_episodes_per_split "${MAX_EPISODES_PER_SPLIT}"
)

if [[ -n "${SOURCE_ROOT}" ]]; then
    CMD+=(--source_root "${SOURCE_ROOT}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
    CMD+=(--output_root "${OUTPUT_ROOT}")
fi
if [[ -n "${TASK_GROUPING}" ]]; then
    CMD+=(--task_grouping "${TASK_GROUPING}")
fi
if [[ "${REQUIRE_EXISTING_QPOS}" == "false" ]]; then
    CMD+=(--no_require_existing_qpos)
fi
if [[ "${ALIGN_WITH_QPOS}" == "false" ]]; then
    CMD+=(--no_align_with_qpos)
fi
if [[ "${OVERWRITE}" == "false" ]]; then
    CMD+=(--no_overwrite)
fi
if [[ "${USER_VERBOSE}" == "true" ]]; then
    CMD+=(--verbose)
fi

echo "Executing:"
printf '  %q ' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}

if [[ ${STATUS} -eq 0 ]]; then
    echo "Run success."
else
    echo "Run failed with code ${STATUS}. Check log: ${LOG_FILE}"
fi

exit ${STATUS}
