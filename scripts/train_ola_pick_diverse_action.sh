#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/nas/code/d0}"
MANIFEST="configs/dataset_manifests/pick_diverse_action.yaml"
RUN_NAME="${1:-pick_diverse_action_hisinit_future_noise}"
PREPARE="${PREPARE:-0}"

cd "$PROJECT_ROOT"

if [ "$PREPARE" = "1" ]; then
    bash scripts/prepare_ola_manifest.sh "$MANIFEST"
fi

exec bash scripts/train_ola_manifest.sh "$MANIFEST" "$RUN_NAME"
