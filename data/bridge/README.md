# Bridge RLDS to Motus Format Converter

This folder provides a converter from Bridge RLDS TFRecord data to Motus-style task folders.

## Output Structure

The converter writes:

```text
bridge_dataset/
├── train/
│   ├── <task_slug>/
│   │   ├── qpos/       # [T, 7] torch tensors (.pt) from observation.state
│   │   ├── videos/     # episode videos (.mp4) from observation.image
│   │   ├── instructions/ # instruction text (.txt), one file per episode
│   │   └── umt5_wan/   # WAN T5 embeddings (.pt)
└── test/
    └── <task_slug>/...
```

Task folder name (`task_slug`) is generated from natural-language instruction text.
By default, we use a normalized instruction key (remove color/spatial/function words)
to reduce over-fragmentation (too many single-episode task folders).

## Dependencies

In addition to project defaults, install:

```bash
pip install tensorflow tensorflow-datasets
```

The converter also needs WAN T5 files at `wan_repo_path`:

- `models_t5_umt5-xxl-enc-bf16.pth`
- `google/umt5-xxl`

## Quick Start

```bash
python data/bridge/bridge_rlds_to_motus.py \
  --config data/bridge/config_bridge_convert.yml
```

## Useful Overrides

```bash
python data/bridge/bridge_rlds_to_motus.py \
  --config data/bridge/config_bridge_convert.yml \
  --splits train \
  --max_episodes_per_split 2 \
  --log_every_n 1
```

```bash
python data/bridge/bridge_rlds_to_motus.py \
  --config data/bridge/config_bridge_convert.yml \
  --overwrite
```

## Generate Epos Only (From Raw RLDS Action)

Use the dedicated epos-only generator:

```bash
python data/bridge/bridge_generate_epos_from_raw.py \
  --config data/bridge/config_bridge_convert.yml \
  --splits train test
```

One-click runner:

```bash
bash data/bridge/run_generate_epos_from_raw.sh
```

Notes:

1. `epos` is built as `[world_vector(3), rotation_delta(3), open_gripper(1)]` with shape `[T, 7]`.
2. By default, existing `epos` files are overwritten.
3. Use `--no_overwrite` to skip existing files.
4. By default, generation requires existing `qpos` and aligns sequence length with `qpos` (`trim` mode).

If you want old behavior (one folder per exact instruction), set in config:

```yaml
task_grouping: "exact_instruction"
```

To skip non-task/noisy instructions (e.g. `Video frame is not showing.`), keep:

```yaml
skip_invalid_instruction: true
invalid_instruction_patterns:
  - "video frame is not showing"
```

## Smoke Test Checklist

Run with `--splits train --max_episodes_per_split 2`, then verify:

1. Generated files include `videos/*.mp4`, `qpos/*.pt`, `umt5_wan/*.pt`.
   Generated files also include `instructions/*.txt`.
2. `qpos` tensors are shaped `[T, 7]` with `T >= 2`.
3. `umt5_wan` is loadable by `torch.load` (list/tensor format).
4. Re-run without `--overwrite`: existing episodes are skipped.

## Failure Handling

- Conversion continues when a single episode fails.
- Failed episodes are recorded in:

```text
<output_root>/failed_episodes.log
```
