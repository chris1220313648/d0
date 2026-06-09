---
name: branch-merge-review
description: Use when asked to fetch a collaborator branch, compare against current branch, summarize code changes, classify risks, and merge proven improvements safely.
---

# Branch Merge Review

## When to use

Use this skill when the user asks to integrate a collaborator branch and wants:

1. `git fetch` of a target remote branch
2. compare vs current branch
3. concise summary of what changed
4. risk classification before merge
5. safe merge of improvements

## Inputs

- `TARGET_REMOTE`: remote name or URL
- `TARGET_BRANCH`: branch name (for example `wenxuan/xxx`)
- `BASE_BRANCH`: current working branch (default: current `HEAD`)

If branch visibility fails, report exactly which remotes were checked and request the missing remote URL or exact branch name.

## Workflow

1. Preflight safety checks.
2. Fetch branch and verify it exists.
3. Produce compare report (commits + file-level diff + hotspot files).
4. Classify risks by change type.
5. Merge with conflict-aware, non-destructive flow.
6. Report what was merged and what still needs validation.

## Commands

```bash
# 1) Preflight
git rev-parse --abbrev-ref HEAD
git status --short
git remote -v

# 2) Fetch and verify
git fetch --all --prune
git fetch "$TARGET_REMOTE" "$TARGET_BRANCH":"refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH"
git show-ref --verify "refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH"

# 3) Compare
BASE_SHA="$(git merge-base HEAD refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH)"
git log --oneline --left-right --cherry-pick --no-merges \
  "HEAD...refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH"
git diff --stat "$BASE_SHA"...refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH
git diff --name-status "$BASE_SHA"...refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH

# 4) Optional deep inspection of hot files
git diff "$BASE_SHA"...refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH -- <path>

# 5) Safe merge
git merge --no-ff "refs/remotes/$TARGET_REMOTE/$TARGET_BRANCH"
```

## Risk Classification

- `Low`: docs/comments/config-only changes, no runtime path modifications.
- `Medium`: localized logic changes with clear tests or obvious behavior bounds.
- `High`: training/inference core loops, data pipeline transforms, model I/O contracts, distributed/precision behavior, or dependency upgrades.

Always call out:

1. behavior regression risk
2. performance risk
3. reproducibility risk
4. compatibility risk

## Reporting Template

Use this output structure:

1. Branch visibility and fetch status
2. Compare summary
3. Risk matrix (`Low/Medium/High`)
4. Merge result (conflicts, resolutions, final files touched)
5. Required follow-up validation (tests, training smoke, metrics)

## Guardrails

- Do not discard or reset unrelated local changes.
- If working tree is dirty, avoid destructive cleanup; merge cautiously.
- If conflicts occur, resolve only in intended files and document decisions.
- If target branch is missing, stop merge and request exact remote/branch details.

