#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLEANUP_SCRIPT="$PROJECT_ROOT/checkpoints/cleanup_keep_latest_checkpoint_files.sh"

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

make_checkpoint() {
    local checkpoint_dir="$1"
    mkdir -p "$checkpoint_dir/pytorch_model"
    touch "$checkpoint_dir/pytorch_model_0.bin"
    touch "$checkpoint_dir/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    touch "$checkpoint_dir/pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
    touch "$checkpoint_dir/pytorch_model/mp_rank_00_model_states.pt"
    touch "$checkpoint_dir/pytorch_model/extra_model_states.pt"
}

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

EXP_A="$TMP_ROOT/group_a/experiment_a"
make_checkpoint "$EXP_A/checkpoint_step_100000"
make_checkpoint "$EXP_A/checkpoint_step_200000"
make_checkpoint "$EXP_A/checkpoint_step_400000"

EXP_B="$TMP_ROOT/group_b/experiment_b"
make_checkpoint "$EXP_B/checkpoint_step_25000"

bash "$CLEANUP_SCRIPT" --root "$TMP_ROOT" >/tmp/cleanup_keep_latest_dry_run.out

grep -q "DRY-RUN" /tmp/cleanup_keep_latest_dry_run.out || fail "dry-run output missing marker"
grep -q "checkpoint_step_400000" /tmp/cleanup_keep_latest_dry_run.out || fail "dry-run output missing latest checkpoint marker"

assert_exists "$EXP_A/checkpoint_step_100000/pytorch_model_0.bin"
assert_exists "$EXP_A/checkpoint_step_100000/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_exists "$EXP_A/checkpoint_step_200000/pytorch_model_0.bin"
assert_exists "$EXP_A/checkpoint_step_200000/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_exists "$EXP_A/checkpoint_step_400000/pytorch_model_0.bin"
assert_exists "$EXP_A/checkpoint_step_400000/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_exists "$EXP_B/checkpoint_step_25000/pytorch_model_0.bin"

bash "$CLEANUP_SCRIPT" --yes --root "$TMP_ROOT" >/tmp/cleanup_keep_latest_delete.out

grep -q "DELETE" /tmp/cleanup_keep_latest_delete.out || fail "delete output missing marker"
grep -q "Removed" /tmp/cleanup_keep_latest_delete.out || fail "delete output missing removal message"

assert_missing "$EXP_A/checkpoint_step_100000/pytorch_model_0.bin"
assert_missing "$EXP_A/checkpoint_step_100000/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_missing "$EXP_A/checkpoint_step_100000/pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
assert_missing "$EXP_A/checkpoint_step_200000/pytorch_model_0.bin"
assert_missing "$EXP_A/checkpoint_step_200000/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_missing "$EXP_A/checkpoint_step_200000/pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"

assert_exists "$EXP_A/checkpoint_step_100000/pytorch_model/mp_rank_00_model_states.pt"
assert_exists "$EXP_A/checkpoint_step_100000/pytorch_model/extra_model_states.pt"
assert_exists "$EXP_A/checkpoint_step_200000/pytorch_model/mp_rank_00_model_states.pt"
assert_exists "$EXP_A/checkpoint_step_200000/pytorch_model/extra_model_states.pt"

assert_exists "$EXP_A/checkpoint_step_400000/pytorch_model_0.bin"
assert_exists "$EXP_A/checkpoint_step_400000/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
assert_exists "$EXP_A/checkpoint_step_400000/pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
assert_exists "$EXP_A/checkpoint_step_400000/pytorch_model/mp_rank_00_model_states.pt"

assert_exists "$EXP_B/checkpoint_step_25000/pytorch_model_0.bin"
assert_exists "$EXP_B/checkpoint_step_25000/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"

echo "[PASS] cleanup keep latest checkpoint files script"
