#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/prepare_interndata_lap.py"

PYTHON_CMD="${PYTHON_CMD:-python}"
ROOT="${ROOT:-/root/nas/code/d0/data/robot_data/interndata}"
INCLUDE="${INCLUDE:-physical sim_updated}"
DEVICE="${DEVICE:-cuda}"
WAN_PATH_ARG=()
SKIP_T5=false
SKIP_LANGUAGE_ACTION=false
SMOKE=false
MAX_DATASETS=0
MAX_EPISODES=0
NUM_SHARDS=1
SHARD_INDEX=0
OVERWRITE_LANGUAGE_ACTION=false
OVERWRITE_T5=false
VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python PATH                    Python executable (default: python or PYTHON_CMD)
  --root PATH                      InternData root (default: ${ROOT})
  --include "physical sim_updated" Top-level subtrees to scan
  --wan-path PATH                  Base path containing Wan2.2-TI2V-5B
  --device DEVICE                  T5 device (default: ${DEVICE})
  --skip-t5                       Generate language_action only
  --skip-language-action          Generate T5 only
  --smoke                         Process 1 dataset and 1 episode
  --max-datasets N                Limit discovered datasets
  --max-episodes N                Limit episodes per dataset
  --num-shards N                  Total number of dataset-root shards
  --shard-index N                 This process shard index, 0-based
  --overwrite-language-action     Regenerate existing language_action files
  --overwrite-t5                  Regenerate existing T5 files
  --verbose                       Enable debug logging
  -h, --help                      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --include) INCLUDE="$2"; shift 2 ;;
        --wan-path) WAN_PATH_ARG=(--wan_path "$2"); shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --skip-t5) SKIP_T5=true; shift ;;
        --skip-language-action) SKIP_LANGUAGE_ACTION=true; shift ;;
        --smoke) SMOKE=true; shift ;;
        --max-datasets) MAX_DATASETS="$2"; shift 2 ;;
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        --shard-index) SHARD_INDEX="$2"; shift 2 ;;
        --overwrite-language-action) OVERWRITE_LANGUAGE_ACTION=true; shift ;;
        --overwrite-t5) OVERWRITE_T5=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ "$SMOKE" == "true" ]]; then
    MAX_DATASETS=1
    MAX_EPISODES=1
fi

CMD=(
    "$PYTHON_CMD" "$PY_SCRIPT"
    --root "$ROOT"
    --include $INCLUDE
    --device "$DEVICE"
    --max_datasets "$MAX_DATASETS"
    --max_episodes "$MAX_EPISODES"
    --num_shards "$NUM_SHARDS"
    --shard_index "$SHARD_INDEX"
)

if [[ ${#WAN_PATH_ARG[@]} -gt 0 ]]; then CMD+=("${WAN_PATH_ARG[@]}"); fi
if [[ "$SKIP_T5" == "true" ]]; then CMD+=(--skip_t5); fi
if [[ "$SKIP_LANGUAGE_ACTION" == "true" ]]; then CMD+=(--skip_language_action); fi
if [[ "$OVERWRITE_LANGUAGE_ACTION" == "true" ]]; then CMD+=(--overwrite_language_action); fi
if [[ "$OVERWRITE_T5" == "true" ]]; then CMD+=(--overwrite_t5); fi
if [[ "$VERBOSE" == "true" ]]; then CMD+=(--verbose); fi

cd "$REPO_ROOT"
printf 'Executing:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
