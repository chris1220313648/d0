#!/bin/bash
# One-click runner for DROID RLDS -> Motus conversion.
export CCUDA_VISIBLE_DEVICES=1
set -u

echo "Starting DROID RLDS conversion at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/droid_rlds_to_motus.py"

PYTHON_CMD="/data/user/wsong890/envs/motus/bin/python"
CONFIG_PATH="${SCRIPT_DIR}/config_droid_convert.yml"
SOURCE_ROOT=""
OUTPUT_ROOT=""
SPLITS="train"
CAMERA_KEYS=("exterior_image_1_left" "exterior_image_2_left" "wrist_image_left")
MAX_EPISODES_PER_SPLIT=0
OVERWRITE=false
SKIP_EXISTING=true
WINDOW_SIZE=""
GENERATE_T5=true
GENERATE_LANGUAGE_ACTION=true
LOG_EVERY_N=""
USER_VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python PATH                  Python executable (default: ${PYTHON_CMD})
  --config PATH                  Config yaml path (default: ${CONFIG_PATH})
  --source-root PATH             Override source_root from config
  --output-root PATH             Override output_root from config
  --splits "train"               Splits to process (default: "${SPLITS}")
  --camera-keys "A B C"          Top/bottom-left/bottom-right camera keys
  --max-episodes-per-split N     0 means full available split (default: ${MAX_EPISODES_PER_SPLIT})
  --overwrite                    Force overwrite existing outputs
  --no-skip-existing             Regenerate even if all expected outputs exist
  --window-size N                Language-action sliding window size
  --no-generate-t5               Disable UMT5 embedding generation
  --no-generate-language-action  Disable language_action generation
  --log-every-n N                Override log frequency
  --verbose                      Enable verbose logs
  -h, --help                     Show help

Examples:
  $(basename "$0")
  $(basename "$0") --max-episodes-per-split 2 --no-generate-t5 --verbose
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --source-root) SOURCE_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --camera-keys) read -r -a CAMERA_KEYS <<< "$2"; shift 2 ;;
        --max-episodes-per-split) MAX_EPISODES_PER_SPLIT="$2"; shift 2 ;;
        --overwrite) OVERWRITE=true; shift ;;
        --no-skip-existing) SKIP_EXISTING=false; shift ;;
        --window-size) WINDOW_SIZE="$2"; shift 2 ;;
        --no-generate-t5) GENERATE_T5=false; shift ;;
        --no-generate-language-action) GENERATE_LANGUAGE_ACTION=false; shift ;;
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
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: Config file not found: ${CONFIG_PATH}"
    exit 1
fi
if [[ ${#CAMERA_KEYS[@]} -ne 3 ]]; then
    echo "Error: --camera-keys must provide exactly 3 keys"
    exit 1
fi

if ! "${PYTHON_CMD}" -c "import torch, numpy, yaml, tensorflow, tensorflow_datasets, cv2" >/dev/null 2>&1; then
    echo "Error: Missing required packages in ${PYTHON_CMD} (need torch, numpy, pyyaml, tensorflow, tensorflow_datasets, opencv-python)"
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/droid_rlds_to_motus_${TIMESTAMP}.log"

echo "Configuration:"
echo "  python: ${PYTHON_CMD}"
echo "  config: ${CONFIG_PATH}"
echo "  source_root: ${SOURCE_ROOT:-<from config>}"
echo "  output_root: ${OUTPUT_ROOT:-<from config>}"
echo "  splits: ${SPLITS}"
echo "  camera_keys: ${CAMERA_KEYS[*]}"
echo "  max_episodes_per_split: ${MAX_EPISODES_PER_SPLIT}"
echo "  overwrite: ${OVERWRITE}"
echo "  skip_existing: ${SKIP_EXISTING}"
echo "  window_size: ${WINDOW_SIZE:-<from config>}"
echo "  generate_t5: ${GENERATE_T5}"
echo "  generate_language_action: ${GENERATE_LANGUAGE_ACTION}"
echo "  log_every_n: ${LOG_EVERY_N:-<from config>}"
echo "  log_file: ${LOG_FILE}"

CMD=(
    "${PYTHON_CMD}" "${PY_SCRIPT}"
    --config "${CONFIG_PATH}"
    --splits ${SPLITS}
    --camera_keys "${CAMERA_KEYS[@]}"
    --max_episodes_per_split "${MAX_EPISODES_PER_SPLIT}"
)

if [[ -n "${SOURCE_ROOT}" ]]; then
    CMD+=(--source_root "${SOURCE_ROOT}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
    CMD+=(--output_root "${OUTPUT_ROOT}")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
    CMD+=(--overwrite)
fi
if [[ "${SKIP_EXISTING}" == "true" ]]; then
    CMD+=(--skip_existing)
fi
if [[ -n "${WINDOW_SIZE}" ]]; then
    CMD+=(--window_size "${WINDOW_SIZE}")
fi
if [[ "${GENERATE_T5}" == "false" ]]; then
    CMD+=(--no_generate_t5)
fi
if [[ "${GENERATE_LANGUAGE_ACTION}" == "false" ]]; then
    CMD+=(--no_generate_language_action)
fi
if [[ -n "${LOG_EVERY_N}" ]]; then
    CMD+=(--log_every_n "${LOG_EVERY_N}")
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
