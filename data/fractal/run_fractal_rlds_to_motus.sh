#!/bin/bash
# One-click runner for Fractal RLDS -> Motus conversion.

set -u

echo "Starting Fractal RLDS conversion at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/fractal_rlds_to_motus.py"

PYTHON_CMD="/data/user/wsong890/envs/motus/bin/python"
CONFIG_PATH="${SCRIPT_DIR}/config_fractal_convert.yml"
SOURCE_ROOT=""
OUTPUT_ROOT=""
SPLITS="train"
MAX_EPISODES_PER_SPLIT=0
OVERWRITE=false
LOG_EVERY_N=""
GENERATE_EPOS=true
EPOS_DIR_NAME=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python PATH                  Python executable (default: ${PYTHON_CMD})
  --config PATH                  Config yaml path (default: ${CONFIG_PATH})
  --source-root PATH             Override source_root from config
  --output-root PATH             Override output_root from config
  --splits "train"               Splits to process (default: "${SPLITS}")
  --max-episodes-per-split N     0 means full split (default: ${MAX_EPISODES_PER_SPLIT})
  --overwrite                    Force overwrite existing outputs
  --log-every-n N                Override log frequency
  --no-generate-epos             Disable epos generation
  --epos-dir-name NAME           Override epos dir name
  -h, --help                     Show help

Examples:
  $(basename "$0")
  $(basename "$0") --splits "train" --max-episodes-per-split 2
  $(basename "$0") --overwrite --max-episodes-per-split 50
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --source-root) SOURCE_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --max-episodes-per-split) MAX_EPISODES_PER_SPLIT="$2"; shift 2 ;;
        --overwrite) OVERWRITE=true; shift ;;
        --log-every-n) LOG_EVERY_N="$2"; shift 2 ;;
        --no-generate-epos) GENERATE_EPOS=false; shift ;;
        --epos-dir-name) EPOS_DIR_NAME="$2"; shift 2 ;;
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

if ! "${PYTHON_CMD}" -c "import torch, numpy, yaml, tensorflow, tensorflow_datasets" >/dev/null 2>&1; then
    echo "Error: Missing required packages in ${PYTHON_CMD} (need torch, numpy, pyyaml, tensorflow, tensorflow_datasets)"
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fractal_rlds_to_motus_${TIMESTAMP}.log"

echo "Configuration:"
echo "  python: ${PYTHON_CMD}"
echo "  config: ${CONFIG_PATH}"
echo "  source_root: ${SOURCE_ROOT:-<from config>}"
echo "  output_root: ${OUTPUT_ROOT:-<from config>}"
echo "  splits: ${SPLITS}"
echo "  max_episodes_per_split: ${MAX_EPISODES_PER_SPLIT}"
echo "  overwrite: ${OVERWRITE}"
echo "  log_every_n: ${LOG_EVERY_N:-<from config>}"
echo "  generate_epos: ${GENERATE_EPOS}"
echo "  epos_dir_name: ${EPOS_DIR_NAME:-<from config>}"
echo "  log_file: ${LOG_FILE}"

CMD=(
    "${PYTHON_CMD}" "${PY_SCRIPT}"
    --config "${CONFIG_PATH}"
    --splits ${SPLITS}
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
if [[ -n "${LOG_EVERY_N}" ]]; then
    CMD+=(--log_every_n "${LOG_EVERY_N}")
fi
if [[ "${GENERATE_EPOS}" == "false" ]]; then
    CMD+=(--no_generate_epos)
fi
if [[ -n "${EPOS_DIR_NAME}" ]]; then
    CMD+=(--epos_dir_name "${EPOS_DIR_NAME}")
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
