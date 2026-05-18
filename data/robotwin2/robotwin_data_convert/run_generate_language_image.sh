#!/bin/bash
# One-click runner for RobotWin language_image generation.
#
# Example:
# bash /data/user/wsong890/user68/cjy/Motus/data/robotwin2/robotwin_data_convert/run_generate_language_image.sh --overwrite

set -u

echo "Starting RobotWin language_image generation at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/robotwin_generate_language_image.py"

# Defaults
PYTHON_CMD="/data/user/wsong890/envs/motus/bin/python"
TARGET_ROOT="/data/user/wsong890/user68/cjy/Motus/data/robotwin_dataset"
SUBSETS="clean,randomized"
INPUT_DIR_NAME="videos"
OUTPUT_DIR_NAME="language_image"
WINDOW_SIZE=16
VLM_CKPT="/data/user/wsong890/user68/cjy/Motus/pretrained_models/Qwen3-VL-2B-Instruct"
GPU_IDS="2,3,4,5,6,7"
DTYPE="auto"
MAX_NEW_TOKENS=96
GEN_BATCH_SIZE=1
VIDEO_NUM_THREADS=2
TEMPERATURE=0.0
TOP_P=0.9
OVERWRITE=true
USER_VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python PATH              Python executable (default: ${PYTHON_CMD})
  --target-root PATH         Dataset root (default: ${TARGET_ROOT})
  --subsets STR              Comma-separated subsets (default: ${SUBSETS})
  --input-dir-name NAME      Input dir under each task (default: ${INPUT_DIR_NAME})
  --output-dir-name NAME     Output dir under each task (default: ${OUTPUT_DIR_NAME})
  --window-size N            Sliding window size (default: ${WINDOW_SIZE})
  --vlm-ckpt PATH            VLM checkpoint path (default: ${VLM_CKPT})
  --gpu-ids IDS              GPU ids, e.g. 3 (single) or 0,1,2,3 (multi) (default: ${GPU_IDS})
  --dtype TYPE               auto|float16|bfloat16|float32 (default: ${DTYPE})
  --max-new-tokens N         Max generated tokens per line (default: ${MAX_NEW_TOKENS})
  --gen-batch-size N         Per-episode generation micro-batch size (default: ${GEN_BATCH_SIZE})
  --video-num-threads N      Decord VideoReader threads (default: ${VIDEO_NUM_THREADS})
  --temperature F            Generation temperature (default: ${TEMPERATURE})
  --top-p F                  Top-p when temperature > 0 (default: ${TOP_P})
  --overwrite                Overwrite existing output txt files
  --verbose                  Enable verbose logging
  -h, --help                 Show this help

Example:
  $(basename "$0") --overwrite --gpu-ids 1
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --target-root) TARGET_ROOT="$2"; shift 2 ;;
        --subsets) SUBSETS="$2"; shift 2 ;;
        --input-dir-name) INPUT_DIR_NAME="$2"; shift 2 ;;
        --output-dir-name) OUTPUT_DIR_NAME="$2"; shift 2 ;;
        --window-size) WINDOW_SIZE="$2"; shift 2 ;;
        --vlm-ckpt) VLM_CKPT="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --gen-batch-size) GEN_BATCH_SIZE="$2"; shift 2 ;;
        --video-num-threads) VIDEO_NUM_THREADS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --overwrite) OVERWRITE=true; shift ;;
        --verbose) USER_VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Error: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Validation
if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "Error: Python script not found: ${PY_SCRIPT}"
    exit 1
fi
if [[ ! -d "${TARGET_ROOT}" ]]; then
    echo "Error: target root not found: ${TARGET_ROOT}"
    exit 1
fi
if [[ ! -d "${VLM_CKPT}" ]]; then
    echo "Warning: VLM checkpoint path not found as local dir: ${VLM_CKPT}"
    echo "If this is a remote model id, you can ignore this warning."
fi
if [[ "${DTYPE}" != "auto" && "${DTYPE}" != "float16" && "${DTYPE}" != "bfloat16" && "${DTYPE}" != "float32" ]]; then
    echo "Error: --dtype must be one of auto|float16|bfloat16|float32"
    exit 1
fi

if ! "${PYTHON_CMD}" -c "import torch, transformers, decord" >/dev/null 2>&1; then
    echo "Error: Required packages missing in Python env (need torch, transformers, decord)"
    echo "Python: ${PYTHON_CMD}"
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/language_image_${TIMESTAMP}.log"
echo "Log file: ${LOG_FILE}"

echo "Configuration:"
echo "  python: ${PYTHON_CMD}"
echo "  target_root: ${TARGET_ROOT}"
echo "  subsets: ${SUBSETS}"
echo "  input_dir_name: ${INPUT_DIR_NAME}"
echo "  output_dir_name: ${OUTPUT_DIR_NAME}"
echo "  window_size: ${WINDOW_SIZE}"
echo "  vlm_ckpt: ${VLM_CKPT}"
echo "  gpu_ids: ${GPU_IDS}"
echo "  dtype: ${DTYPE}"
echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
echo "  gen_batch_size: ${GEN_BATCH_SIZE}"
echo "  video_num_threads: ${VIDEO_NUM_THREADS}"
echo "  temperature: ${TEMPERATURE}"
echo "  top_p: ${TOP_P}"
echo "  overwrite: ${OVERWRITE}"
echo "  verbose: ${USER_VERBOSE}"

CMD=(
    "${PYTHON_CMD}" "${PY_SCRIPT}"
    --target-root "${TARGET_ROOT}"
    --subsets "${SUBSETS}"
    --input-dir-name "${INPUT_DIR_NAME}"
    --output-dir-name "${OUTPUT_DIR_NAME}"
    --window-size "${WINDOW_SIZE}"
    --vlm-checkpoint-path "${VLM_CKPT}"
    --dtype "${DTYPE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --gen-batch-size "${GEN_BATCH_SIZE}"
    --video-num-threads "${VIDEO_NUM_THREADS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
)

if [[ -n "${GPU_IDS}" ]]; then
    CMD+=(--gpu-ids "${GPU_IDS}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
    CMD+=(--overwrite)
fi
if [[ "${USER_VERBOSE}" == "true" ]]; then
    CMD+=(--verbose)
fi

echo ""
echo "Running:"
printf ' %q' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}

echo ""
if [[ ${STATUS} -eq 0 ]]; then
    echo "✅ language_image generation completed successfully."
else
    echo "❌ language_image generation failed with exit code ${STATUS}."
fi
echo "Log: ${LOG_FILE}"

exit ${STATUS}
