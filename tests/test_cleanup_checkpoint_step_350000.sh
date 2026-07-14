#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLEANUP_SCRIPT="$PROJECT_ROOT/scripts/cleanup_checkpoint_step_350000.sh"

fail() {
    echo "[FAIL] $*" >&2
    exit 1
}

assert_exists() {
    [ -e "$1" ] || fail "expected to exist: $1"
}

assert_missing() {
    [ ! -e "$1" ] || fail "expected to be removed: $1"
}

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

TARGET_DIR="$TMP_ROOT/checkpoint_step_350000"
mkdir -p "$TARGET_DIR/optim_states"
touch "$TARGET_DIR/optim_states/state.bin"
touch "$TARGET_DIR/pytorch.bin"
touch "$TARGET_DIR/pytorch_model_0.bin"
mkdir -p "$TARGET_DIR/pytorch_model"
touch "$TARGET_DIR/pytorch_model/model.bin"
touch "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
touch "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
touch "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_model_states.pt"

bash "$CLEANUP_SCRIPT" --dry-run "$TARGET_DIR" >/tmp/cleanup_checkpoint_dry_run.out

assert_exists "$TARGET_DIR/optim_states"
assert_exists "$TARGET_DIR/pytorch.bin"
assert_exists "$TARGET_DIR/pytorch_model_0.bin"
assert_exists "$TARGET_DIR/pytorch_model/model.bin"
assert_exists "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_exists "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
assert_exists "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_model_states.pt"
grep -q "DRY-RUN" /tmp/cleanup_checkpoint_dry_run.out || fail "dry-run output missing marker"

bash "$CLEANUP_SCRIPT" --yes "$TARGET_DIR" >/tmp/cleanup_checkpoint_delete.out

assert_missing "$TARGET_DIR/optim_states"
assert_missing "$TARGET_DIR/pytorch.bin"
assert_missing "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_missing "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
assert_exists "$TARGET_DIR/pytorch_model_0.bin"
assert_exists "$TARGET_DIR/pytorch_model/model.bin"
assert_exists "$TARGET_DIR/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_model_states.pt"
grep -q "Removed" /tmp/cleanup_checkpoint_delete.out || fail "delete output missing removal message"

BAD_TARGET="$TMP_ROOT/not_a_checkpoint"
mkdir -p "$BAD_TARGET"
if bash "$CLEANUP_SCRIPT" --yes "$BAD_TARGET" >/tmp/cleanup_checkpoint_bad.out 2>&1; then
    fail "expected non-checkpoint target to be rejected"
fi
grep -q "Refusing" /tmp/cleanup_checkpoint_bad.out || fail "bad target output missing refusal"

echo "[PASS] cleanup checkpoint script"
