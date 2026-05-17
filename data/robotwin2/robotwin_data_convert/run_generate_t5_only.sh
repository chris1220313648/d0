#!/bin/bash
# One-click script: backfill only umt5_wan embeddings for already-converted data.

set -u

echo "Starting RobotWin T5-only backfill at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config.yml"

CONFIG_FILE="$DEFAULT_CONFIG"
OVERWRITE=true
USER_VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [--config PATH] [--overwrite] [--verbose]

Options:
  --config PATH   Path to config.yml (default: ${DEFAULT_CONFIG})
  --overwrite     Regenerate existing umt5_wan/*.pt files
  --verbose       Enable verbose logs
  -h, --help      Show this help
EOF
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ $# -lt 2 ]]; then
                echo "Error: --config requires a value"
                usage
                exit 1
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --verbose)
            USER_VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML-like fields needed for checks/logging
TARGET_ROOT=$(grep "^target_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
LOG_LEVEL=$(grep "^log_level:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_REPO_PATH=$(grep "^wan_repo_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

LOG_LEVEL=${LOG_LEVEL:-"INFO"}

if [ -z "${TARGET_ROOT}" ]; then
    echo "Error: target_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "${TARGET_ROOT}" ]; then
    echo "Error: target_root directory not found: $TARGET_ROOT"
    exit 1
fi

if [ -z "${WAN_REPO_PATH}" ]; then
    echo "Error: wan_repo_path is not set in $CONFIG_FILE"
    exit 1
fi

echo "Configuration loaded successfully:"
echo "  Target Root: $TARGET_ROOT"
echo "  WAN Repo Path: $WAN_REPO_PATH"
echo "  Log Level: $LOG_LEVEL"
echo "  Overwrite: $OVERWRITE"

echo "Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: No Python interpreter found"
    exit 1
fi

if [ ! -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
    echo "Python executable: $(which $PYTHON_CMD)"
else
    echo "Using system Python: $(which $PYTHON_CMD)"
fi

if ! $PYTHON_CMD -c "import torch" &> /dev/null; then
    echo "Error: PyTorch not found in current Python environment"
    echo "Current environment: ${CONDA_DEFAULT_ENV:-system}"
    echo "Python path: $(which $PYTHON_CMD)"
    exit 1
fi
echo "Python environment check passed"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/t5_backfill_${TIMESTAMP}.log"
echo "Logs will be saved to: $LOG_FILE"

VERBOSE_FLAG=""
if [ "$USER_VERBOSE" = true ] || [ "$LOG_LEVEL" = "DEBUG" ]; then
    VERBOSE_FLAG="--verbose"
fi

OVERWRITE_FLAG=''
if [ "$OVERWRITE" = true ]; then
    OVERWRITE_FLAG="--overwrite"
fi

echo "Executing T5-only backfill script..."
cd "$SCRIPT_DIR"

$PYTHON_CMD robotwin_generate_t5_only.py \
    --config "$CONFIG_FILE" \
    $OVERWRITE_FLAG \
    $VERBOSE_FLAG \
    2>&1 | tee "$LOG_FILE"

STATUS=${PIPESTATUS[0]}

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "T5 BACKFILL SUMMARY"
    echo "=========================================="
    echo "End time: $(date +%H:%M:%S)"
    echo "Target: $TARGET_ROOT"
    echo "Log file: $LOG_FILE"
    if [ -d "$TARGET_ROOT" ]; then
        T5_COUNT=$(find "$TARGET_ROOT" -path "*/umt5_wan/*.pt" | wc -l)
        echo "Total T5 embeddings: $T5_COUNT"
    fi
    echo "T5-only backfill completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "T5 BACKFILL FAILED"
    echo "=========================================="
    echo "Exit code: $STATUS"
    echo "Check log file for details: $LOG_FILE"
    echo "=========================================="
    exit $STATUS
fi

echo "Script completed at $(date)"
