#!/bin/bash
set -euo pipefail

PORT="${1:-9883}"
MOTUS_ROOT="${MOTUS_ROOT:-/root/nas/code/d0}"

for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "Evaluating $suite on policy port $PORT"
    "$MOTUS_ROOT/scripts/eval_libero_plus.sh" "$suite" "$PORT"
done
