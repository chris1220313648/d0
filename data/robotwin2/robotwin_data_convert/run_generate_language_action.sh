#!/bin/bash
# One-click runner for RobotWin language action generation.
# Supports:
# 1) Full generation for selected subsets
# bash /data/user/wsong890/user68/cjy/Motus/data/robotwin2/robotwin_data_convert/run_generate_language_action.sh --overwrite

# 2) Single-case test for one episode sample
# bash /data/user/wsong890/user68/cjy/Motus/data/robotwin2/robotwin_data_convert/run_generate_language_action.sh \
#   --single-test \
#   --test-subset clean \
#   --test-task adjust_bottle \
#   --test-episode-id 0 \
#   --output-dir-name language_action_sample \
#   --overwrite \
#   --preview-lines 3



set -u

echo "Starting RobotWin language_action generation at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/robotwin_generate_language_action.py"

# Defaults (aligned with your current command)
PYTHON_CMD="/data/user/wsong890/envs/motus/bin/python"
TARGET_ROOT="/data/user/wsong890/user68/cjy/Motus/data/robotwin_dataset"
SUBSETS="clean,randomized"
INPUT_DIR_NAME="epos"
INPUT_MODE="absolute_xyzrpy"
QUAT_ORDER="wxyz"
WINDOW_SIZE=16
OUTPUT_DIR_NAME="language_action"
OVERWRITE=false
USER_VERBOSE=false

# Single-test mode
SINGLE_TEST=false
TEST_SUBSET="clean"
TEST_TASK="adjust_bottle"
TEST_EPISODE_ID="0"
PREVIEW_LINES=6

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

General Options:
  --python PATH              Python executable (default: ${PYTHON_CMD})
  --target-root PATH         Dataset root (default: ${TARGET_ROOT})
  --subsets STR              Comma-separated subsets (default: ${SUBSETS})
  --input-dir-name NAME      Input dir under each task (default: ${INPUT_DIR_NAME})
  --input-mode MODE          auto|delta|absolute_xyzrpy|absolute_xyzquat (default: ${INPUT_MODE})
  --quat-order ORDER         wxyz|xyzw (default: ${QUAT_ORDER})
  --window-size N            Sliding window size (default: ${WINDOW_SIZE})
  --output-dir-name NAME     Output dir under each task (default: ${OUTPUT_DIR_NAME})
  --overwrite                Overwrite existing txt
  --verbose                  Enable verbose logging

Single-Test Options:
  --single-test              Run only one episode sample (no full scan)
  --test-subset NAME         Subset for single test (default: ${TEST_SUBSET})
  --test-task NAME           Task name for single test (default: ${TEST_TASK})
  --test-episode-id ID       Episode id for single test (default: ${TEST_EPISODE_ID})
  --preview-lines N          Print first N lines after single test (default: ${PREVIEW_LINES})

Help:
  -h, --help                 Show this help

Examples:
  # Full generation (your target command)
  $(basename "$0") --overwrite

  # Single sample test
  $(basename "$0") --single-test --test-subset clean --test-task adjust_bottle --test-episode-id 0 --overwrite --verbose
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --target-root) TARGET_ROOT="$2"; shift 2 ;;
        --subsets) SUBSETS="$2"; shift 2 ;;
        --input-dir-name) INPUT_DIR_NAME="$2"; shift 2 ;;
        --input-mode) INPUT_MODE="$2"; shift 2 ;;
        --quat-order) QUAT_ORDER="$2"; shift 2 ;;
        --window-size) WINDOW_SIZE="$2"; shift 2 ;;
        --output-dir-name) OUTPUT_DIR_NAME="$2"; shift 2 ;;
        --overwrite) OVERWRITE=true; shift ;;
        --verbose) USER_VERBOSE=true; shift ;;

        --single-test) SINGLE_TEST=true; shift ;;
        --test-subset) TEST_SUBSET="$2"; shift 2 ;;
        --test-task) TEST_TASK="$2"; shift 2 ;;
        --test-episode-id) TEST_EPISODE_ID="$2"; shift 2 ;;
        --preview-lines) PREVIEW_LINES="$2"; shift 2 ;;

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
if [[ "${INPUT_MODE}" != "auto" && "${INPUT_MODE}" != "delta" && "${INPUT_MODE}" != "absolute_xyzrpy" && "${INPUT_MODE}" != "absolute_xyzquat" ]]; then
    echo "Error: --input-mode must be one of auto|delta|absolute_xyzrpy|absolute_xyzquat"
    exit 1
fi
if [[ "${QUAT_ORDER}" != "wxyz" && "${QUAT_ORDER}" != "xyzw" ]]; then
    echo "Error: --quat-order must be wxyz or xyzw"
    exit 1
fi

if ! "${PYTHON_CMD}" -c "import torch, numpy" >/dev/null 2>&1; then
    echo "Error: Required packages missing in Python env (need torch, numpy)"
    echo "Python: ${PYTHON_CMD}"
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/language_action_${TIMESTAMP}.log"
echo "Log file: ${LOG_FILE}"

echo "Configuration:"
echo "  python: ${PYTHON_CMD}"
echo "  target_root: ${TARGET_ROOT}"
echo "  subsets: ${SUBSETS}"
echo "  input_dir_name: ${INPUT_DIR_NAME}"
echo "  input_mode: ${INPUT_MODE}"
echo "  quat_order: ${QUAT_ORDER}"
echo "  window_size: ${WINDOW_SIZE}"
echo "  output_dir_name: ${OUTPUT_DIR_NAME}"
echo "  overwrite: ${OVERWRITE}"
echo "  single_test: ${SINGLE_TEST}"

if [[ "${SINGLE_TEST}" == "true" ]]; then
    echo "Single-test target: ${TEST_SUBSET}/${TEST_TASK}/${TEST_EPISODE_ID}"

    OVERWRITE_ARG="0"
    VERBOSE_ARG="0"
    if [[ "${OVERWRITE}" == "true" ]]; then OVERWRITE_ARG="1"; fi
    if [[ "${USER_VERBOSE}" == "true" ]]; then VERBOSE_ARG="1"; fi

    {
        "${PYTHON_CMD}" - "$PY_SCRIPT" "$TARGET_ROOT" "$TEST_SUBSET" "$TEST_TASK" "$TEST_EPISODE_ID" \
            "$INPUT_DIR_NAME" "$INPUT_MODE" "$QUAT_ORDER" "$WINDOW_SIZE" "$OUTPUT_DIR_NAME" \
            "$OVERWRITE_ARG" "$VERBOSE_ARG" "$PREVIEW_LINES" <<'PY'
import importlib.util
import logging
import sys
from pathlib import Path

script_path = Path(sys.argv[1])
target_root = Path(sys.argv[2])
subset = sys.argv[3]
task = sys.argv[4]
episode_id = sys.argv[5]
input_dir_name = sys.argv[6]
input_mode = sys.argv[7]
quat_order = sys.argv[8]
window_size = int(sys.argv[9])
output_dir_name = sys.argv[10]
overwrite = bool(int(sys.argv[11]))
verbose = bool(int(sys.argv[12]))
preview_lines = int(sys.argv[13])

if verbose:
    logging.getLogger().setLevel(logging.DEBUG)

spec = importlib.util.spec_from_file_location("rw_lang_action", script_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

runner = mod.RobotWinLanguageActionBackfill(
    target_root=target_root,
    subsets=[subset],
    window_size=window_size,
    input_dir_name=input_dir_name,
    input_mode=input_mode,
    quat_order=quat_order,
    output_dir_name=output_dir_name,
    overwrite=overwrite,
)

input_path = target_root / subset / task / input_dir_name / f"{episode_id}.pt"
out_path = target_root / subset / task / output_dir_name / f"{episode_id}.txt"

if not input_path.exists():
    raise FileNotFoundError(f"Input episode not found: {input_path}")
if out_path.exists() and not overwrite:
    print(f"[SKIP] Output exists and overwrite=False: {out_path}")
else:
    ok, lines = runner._process_episode(input_path, out_path)
    print(f"[DONE] ok={ok} lines={lines} output={out_path}")

print(f"[PREVIEW] first {preview_lines} lines from {out_path}:")
with out_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= preview_lines:
            break
        print(line.rstrip("\n"))
PY
    } 2>&1 | tee "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}
else
    CMD=(
        "${PYTHON_CMD}" "${PY_SCRIPT}"
        --target-root "${TARGET_ROOT}"
        --subsets "${SUBSETS}"
        --input-dir-name "${INPUT_DIR_NAME}"
        --input-mode "${INPUT_MODE}"
        --quat-order "${QUAT_ORDER}"
        --window-size "${WINDOW_SIZE}"
        --output-dir-name "${OUTPUT_DIR_NAME}"
    )
    if [[ "${OVERWRITE}" == "true" ]]; then
        CMD+=(--overwrite)
    fi
    if [[ "${USER_VERBOSE}" == "true" ]]; then
        CMD+=(--verbose)
    fi

    echo "Executing full run:"
    printf '  %q ' "${CMD[@]}"
    echo

    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    STATUS=${PIPESTATUS[0]}
fi

if [[ ${STATUS} -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "LANGUAGE ACTION RUN SUCCESS"
    echo "=========================================="
    echo "Log file: ${LOG_FILE}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "LANGUAGE ACTION RUN FAILED"
    echo "=========================================="
    echo "Exit code: ${STATUS}"
    echo "Log file: ${LOG_FILE}"
    echo "=========================================="
    exit ${STATUS}
fi

echo "Completed at $(date)"
