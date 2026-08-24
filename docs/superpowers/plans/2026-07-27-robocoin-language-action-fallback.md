# RoboCOIN Language-Action Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep RoboCOIN numerical-action samples trainable when their optional language-action file is absent by selecting standard VLM preprocessing for that sample.

**Architecture:** Make missing conventional language-action files return `None`, while metadata-declared missing paths and malformed files remain errors. Centralize the per-sample choice between LAP and standard VLM preprocessing in the eager RoboCOIN loader and reuse it from the lazy subclass so both v2 configurations have identical behavior.

**Tech Stack:** Python 3.10, PyTorch dataset loaders, pytest, LeRobot

## Global Constraints

- Numerical `initial_state` and `action_sequence` outputs must remain unchanged.
- Only absent conventional language-action files fall back; declared paths, unreadable files, empty files, and invalid metadata remain errors.
- Do not alter other dataset families.
- Preserve all pre-existing user changes in the dirty worktree.

---

### Task 1: Add Missing-File and Preprocessing Regression Tests

**Files:**
- Modify: `tests/test_lerobot_lazy_datasets.py`
- Test: `tests/test_lerobot_lazy_datasets.py`

**Interfaces:**
- Consumes: `LeRobotRoboCOINDataset._load_language_action_from_file(...)`
- Produces: expected interface `_preprocess_vlm_inputs(text_instr, first_frame_pil, language_action)` returning processor inputs

- [ ] **Step 1: Add a tiny-data option that omits the language-action metadata pointer and file**

Extend `_make_tiny_lazy_robocoin` with `include_language_action: bool = True`. When false, omit `language_action_path` from `episodes.jsonl` and do not write the corresponding text file, while retaining parquet action data and T5 embeddings.

- [ ] **Step 2: Write the failing missing-file fallback test**

Create a lazy RoboCOIN sample with `include_language_action=False`, assign a minimal non-`None` VLM processor, replace only video decoding and the two external preprocessing functions with deterministic sentinels, then assert:

```python
sample = dataset[0]
assert sample["vlm_inputs"] == {"mode": "default"}
assert sample["action_sequence"].shape == (1, 2)
```

Also assert the LAP sentinel was not selected.

- [ ] **Step 3: Write the failing declared-path integrity test**

Construct a lightweight `LeRobotRoboCOINDataset` instance without running its expensive initializer. Give it episode metadata containing an explicit nonexistent `language_action_path` and assert:

```python
with pytest.raises(FileNotFoundError):
    dataset._load_language_action_from_file(0, 0, 0)
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
/opt/conda/envs/motus/bin/python -m pytest \
  tests/test_lerobot_lazy_datasets.py \
  -k 'missing_language_action or declared_language_action' -vv
```

Expected: the missing conventional-file case fails with the current `FileNotFoundError`; the declared-path test already documents retained behavior.

### Task 2: Implement Per-Sample Default VLM Fallback

**Files:**
- Modify: `data/lerobot/lerobot_robocoin_dataset.py`
- Modify: `data/lerobot/lerobot_lazy_dataset.py`
- Test: `tests/test_lerobot_lazy_datasets.py`

**Interfaces:**
- Produces: `_load_language_action_from_file(...) -> Optional[str]`
- Produces: `_preprocess_vlm_inputs(text_instr: str, first_frame_pil: Any, language_action: Optional[str]) -> Any`
- Consumes: module-level `preprocess_vlm_messages` and `preprocess_vlm_messages_lap`

- [ ] **Step 1: Return `None` only for an absent conventional file**

In eager and lazy `_load_language_action_from_file`, distinguish whether the episode metadata declared a path. Return `None` only when no path was declared and the conventional path does not exist. Continue opening declared paths normally so nonexistent/unreadable paths raise `FileNotFoundError`, and keep empty-file validation unchanged.

- [ ] **Step 2: Centralize preprocessing selection**

Add this behavior to `LeRobotRoboCOINDataset`:

```python
def _preprocess_vlm_inputs(self, text_instr, first_frame_pil, language_action):
    if self.use_language_action and language_action is not None:
        return preprocess_vlm_messages_lap(
            text_instr,
            first_frame_pil,
            self.vlm_processor,
            language_action,
            supervise_answer=True,
        )
    return preprocess_vlm_messages(
        text_instr,
        first_frame_pil,
        self.vlm_processor,
    )
```

Use it from both eager and lazy RoboCOIN `__getitem__` methods.

- [ ] **Step 3: Run focused tests and verify GREEN**

Run:

```bash
/opt/conda/envs/motus/bin/python -m pytest \
  tests/test_lerobot_lazy_datasets.py \
  -k 'robocoin' -vv
```

Expected: all selected RoboCOIN lazy tests pass, including the new fallback and retained-integrity cases.

- [ ] **Step 4: Run related regression tests**

Run:

```bash
/opt/conda/envs/motus/bin/python -m pytest \
  tests/test_lerobot_lazy_datasets.py \
  tests/test_robocoin_lerobot.py \
  tests/test_multidataset_lap_v2.py -q
```

Expected: zero failures.

- [ ] **Step 5: Validate syntax and inspect the scoped diff**

Run:

```bash
/opt/conda/envs/motus/bin/python -m py_compile \
  data/lerobot/lerobot_robocoin_dataset.py \
  data/lerobot/lerobot_lazy_dataset.py
git diff --check -- \
  data/lerobot/lerobot_robocoin_dataset.py \
  data/lerobot/lerobot_lazy_dataset.py \
  tests/test_lerobot_lazy_datasets.py
```

Expected: both commands exit successfully.

