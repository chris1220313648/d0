#!/bin/bash
# One-click runner for Fractal language_action generation from existing epos files.

set -u

echo "Starting Fractal language_action generation at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/fractal_generate_language_action.py"

PYTHON_CMD="/data/user/wsong890/envs/motus/bin/python"
OUTPUT_ROOT="/data/user/wsong890/user68/cjy/Motus/data/fractal/fractal_dataset"
SPLITS="train"
EPOS_DIR_NAME="epos"
LANGUAGE_ACTION_DIR_NAME="language_action"
WINDOW_SIZE=16
OVERWRITE_LANGUAGE_ACTION=false
MAX_EPISODES_PER_SPLIT=0
LOG_EVERY_N=200
USER_VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python PATH                     Python executable (default: ${PYTHON_CMD})
  --output-root PATH                Dataset output root (default: ${OUTPUT_ROOT})
  --splits "train"                  Splits to process (default: "${SPLITS}")
  --epos-dir-name NAME              Epos dir name (default: ${EPOS_DIR_NAME})
  --language-action-dir-name NAME   Language action dir name (default: ${LANGUAGE_ACTION_DIR_NAME})
  --window-size N                   Sliding window size (default: ${WINDOW_SIZE})
  --overwrite-language-action       Overwrite existing language_action files
  --max-episodes-per-split N        0 means full split (default: ${MAX_EPISODES_PER_SPLIT})
  --log-every-n N                   Log every N episodes (default: ${LOG_EVERY_N})
  --verbose                         Enable verbose logs
  -h, --help                        Show help

Examples:
  $(basename "$0")
  $(basename "$0") --splits "train" --max-episodes-per-split 2
  $(basename "$0") --overwrite-language-action
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --epos-dir-name) EPOS_DIR_NAME="$2"; shift 2 ;;
        --language-action-dir-name) LANGUAGE_ACTION_DIR_NAME="$2"; shift 2 ;;
        --window-size) WINDOW_SIZE="$2"; shift 2 ;;
        --overwrite-language-action) OVERWRITE_LANGUAGE_ACTION=true; shift ;;
        --max-episodes-per-split) MAX_EPISODES_PER_SPLIT="$2"; shift 2 ;;
        --log-every-n) LOG_EVERY_N="$2"; shift 2 ;;
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

if ! "${PYTHON_CMD}" -c "import torch, numpy" >/dev/null 2>&1; then
    echo "Error: Missing required packages in ${PYTHON_CMD} (need torch, numpy)"
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fractal_language_action_${TIMESTAMP}.log"

echo "Configuration:"
echo "  python: ${PYTHON_CMD}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  splits: ${SPLITS}"
echo "  epos_dir_name: ${EPOS_DIR_NAME}"
echo "  language_action_dir_name: ${LANGUAGE_ACTION_DIR_NAME}"
echo "  window_size: ${WINDOW_SIZE}"
echo "  overwrite_language_action: ${OVERWRITE_LANGUAGE_ACTION}"
echo "  max_episodes_per_split: ${MAX_EPISODES_PER_SPLIT}"
echo "  log_every_n: ${LOG_EVERY_N}"
echo "  log_file: ${LOG_FILE}"

CMD=(
    "${PYTHON_CMD}" "${PY_SCRIPT}"
    --output_root "${OUTPUT_ROOT}"
    --splits ${SPLITS}
    --epos_dir_name "${EPOS_DIR_NAME}"
    --language_action_dir_name "${LANGUAGE_ACTION_DIR_NAME}"
    --window_size "${WINDOW_SIZE}"
    --max_episodes_per_split "${MAX_EPISODES_PER_SPLIT}"
    --log_every_n "${LOG_EVERY_N}"
)

if [[ "${OVERWRITE_LANGUAGE_ACTION}" == "true" ]]; then
    CMD+=(--overwrite_language_action)
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
