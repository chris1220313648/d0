#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <target_remote> <target_branch> [base_ref]"
  echo "Example: $0 origin wenxuan/xxx HEAD"
  exit 1
fi

TARGET_REMOTE="$1"
TARGET_BRANCH="$2"
BASE_REF="${3:-HEAD}"

REMOTE_REF="refs/remotes/${TARGET_REMOTE}/${TARGET_BRANCH}"

echo "== Preflight =="
git rev-parse --abbrev-ref HEAD
git status --short

echo
echo "== Fetch =="
git fetch --all --prune
git fetch "${TARGET_REMOTE}" "${TARGET_BRANCH}:${REMOTE_REF}"
git show-ref --verify "${REMOTE_REF}" >/dev/null
echo "Fetched ${REMOTE_REF}"

BASE_SHA="$(git merge-base "${BASE_REF}" "${REMOTE_REF}")"
echo "Merge base: ${BASE_SHA}"

echo
echo "== Commit Delta =="
git log --oneline --left-right --cherry-pick --no-merges "${BASE_REF}...${REMOTE_REF}"

echo
echo "== Diff Stat =="
git diff --stat "${BASE_SHA}...${REMOTE_REF}"

echo
echo "== File Changes =="
git diff --name-status "${BASE_SHA}...${REMOTE_REF}"

echo
echo "== Risk Heuristics =="
HIGH_RISK_PATTERNS='(train/|inference/|models/|data/|dataset|dataloader|distributed|deepspeed|optimizer|loss|precision|bf16|fp16|amp)'
MEDIUM_RISK_PATTERNS='(config|scripts/|utils/|metrics|eval)'

CHANGED_FILES="$(git diff --name-only "${BASE_SHA}...${REMOTE_REF}" || true)"
HIGH_HITS="$(echo "${CHANGED_FILES}" | rg -n "${HIGH_RISK_PATTERNS}" || true)"
MEDIUM_HITS="$(echo "${CHANGED_FILES}" | rg -n "${MEDIUM_RISK_PATTERNS}" || true)"

if [[ -n "${HIGH_HITS}" ]]; then
  echo "Overall risk: HIGH"
  echo "High-risk files:"
  echo "${HIGH_HITS}"
elif [[ -n "${MEDIUM_HITS}" ]]; then
  echo "Overall risk: MEDIUM"
  echo "Medium-risk files:"
  echo "${MEDIUM_HITS}"
else
  echo "Overall risk: LOW"
fi

echo
echo "== Next Step =="
echo "Review diffs, then merge with:"
echo "git merge --no-ff ${REMOTE_REF}"

