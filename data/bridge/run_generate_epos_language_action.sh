#!/bin/bash
# One-click runner for Bridge epos + language_action generation from RLDS action.

set -u

echo "Starting Bridge epos/language_action generation at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/bridge_generate_epos_language_action.py"

PYTHON_CMD="/data/user/wsong890/envs/motus/bin/python"
CONFIG_PATH="${SCRIPT_DIR}/config_bridge_convert.yml"
SOURCE_ROOT=""
OUTPUT_ROOT=""
SPLITS="train test"
WINDOW_SIZE=16
EPOS_DIR_NAME="epos"
LANGUAGE_ACTION_DIR_NAME="language_action"
OVERWRITE_ALL=false
OVERWRITE_EPOS=false
OVERWRITE_LANGUAGE_ACTION=false
REQUIRE_EXISTING_QPOS=true
ALIGN_WITH_QPOS=true
LENGTH_MODE="trim"
MAX_EPISODES_PER_SPLIT=0
USER_VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python PATH                      Python executable (default: ${PYTHON_CMD})
  --config PATH                      Config yaml path (default: ${CONFIG_PATH})
  --source-root PATH                 Override source_root from config
  --output-root PATH                 Override output_root from config
  --splits "train test"              Splits to process (default: "${SPLITS}")
  --window-size N                    Language window size (default: ${WINDOW_SIZE})
  --epos-dir-name NAME               Epos dir name (default: ${EPOS_DIR_NAME})
  --language-action-dir-name NAME    Language-action dir name (default: ${LANGUAGE_ACTION_DIR_NAME})
  --overwrite-all                    Overwrite both epos and language_action
  --overwrite-epos                   Overwrite only epos
  --overwrite-language-action        Overwrite only language_action
  --no-require-existing-qpos         Allow writing even when qpos file is missing
  --no-align-with-qpos               Disable qpos length alignment
  --length-mode MODE                 trim|strict (default: ${LENGTH_MODE})
  --max-episodes-per-split N         0 means full split (default: ${MAX_EPISODES_PER_SPLIT})
  --verbose                          Enable verbose logs
  -h, --help                         Show help

Examples:
  $(basename "$0") --overwrite-all
  $(basename "$0") --splits "train" --max-episodes-per-split 50 --overwrite-language-action
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --source-root) SOURCE_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --window-size) WINDOW_SIZE="$2"; shift 2 ;;
        --epos-dir-name) EPOS_DIR_NAME="$2"; shift 2 ;;
        --language-action-dir-name) LANGUAGE_ACTION_DIR_NAME="$2"; shift 2 ;;
        --overwrite-all) OVERWRITE_ALL=true; shift ;;
        --overwrite-epos) OVERWRITE_EPOS=true; shift ;;
        --overwrite-language-action) OVERWRITE_LANGUAGE_ACTION=true; shift ;;
        --no-require-existing-qpos) REQUIRE_EXISTING_QPOS=false; shift ;;
        --no-align-with-qpos) ALIGN_WITH_QPOS=false; shift ;;
        --length-mode) LENGTH_MODE="$2"; shift 2 ;;
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

if ! "${PYTHON_CMD}" -c "import torch, numpy, yaml, tensorflow, tensorflow_datasets" >/dev/null 2>&1; then
    echo "Error: Missing required packages in ${PYTHON_CMD} (need torch, numpy, pyyaml, tensorflow, tensorflow_datasets)"
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/bridge_epos_lang_action_${TIMESTAMP}.log"

echo "Configuration:"
echo "  python: ${PYTHON_CMD}"
echo "  config: ${CONFIG_PATH}"
echo "  source_root: ${SOURCE_ROOT:-<from config>}"
echo "  output_root: ${OUTPUT_ROOT:-<from config>}"
echo "  splits: ${SPLITS}"
echo "  window_size: ${WINDOW_SIZE}"
echo "  epos_dir_name: ${EPOS_DIR_NAME}"
echo "  language_action_dir_name: ${LANGUAGE_ACTION_DIR_NAME}"
echo "  overwrite_all: ${OVERWRITE_ALL}"
echo "  overwrite_epos: ${OVERWRITE_EPOS}"
echo "  overwrite_language_action: ${OVERWRITE_LANGUAGE_ACTION}"
echo "  require_existing_qpos: ${REQUIRE_EXISTING_QPOS}"
echo "  align_with_qpos: ${ALIGN_WITH_QPOS}"
echo "  length_mode: ${LENGTH_MODE}"
echo "  max_episodes_per_split: ${MAX_EPISODES_PER_SPLIT}"
echo "  log_file: ${LOG_FILE}"

CMD=(
    "${PYTHON_CMD}" "${PY_SCRIPT}"
    --config "${CONFIG_PATH}"
    --splits ${SPLITS}
    --window_size "${WINDOW_SIZE}"
    --epos_dir_name "${EPOS_DIR_NAME}"
    --language_action_dir_name "${LANGUAGE_ACTION_DIR_NAME}"
    --length_mode "${LENGTH_MODE}"
    --max_episodes_per_split "${MAX_EPISODES_PER_SPLIT}"
)

if [[ -n "${SOURCE_ROOT}" ]]; then
    CMD+=(--source_root "${SOURCE_ROOT}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
    CMD+=(--output_root "${OUTPUT_ROOT}")
fi
if [[ "${OVERWRITE_ALL}" == "true" ]]; then
    CMD+=(--overwrite_all)
fi
if [[ "${OVERWRITE_EPOS}" == "true" ]]; then
    CMD+=(--overwrite_epos)
fi
if [[ "${OVERWRITE_LANGUAGE_ACTION}" == "true" ]]; then
    CMD+=(--overwrite_language_action)
fi
if [[ "${REQUIRE_EXISTING_QPOS}" == "false" ]]; then
    CMD+=(--no_require_existing_qpos)
fi
if [[ "${ALIGN_WITH_QPOS}" == "false" ]]; then
    CMD+=(--no_align_with_qpos)
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
