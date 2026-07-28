# RoboCOIN Strict Integrity Scanner Design

## Goal

Add a read-only command-line scanner that checks every discovered RoboCOIN
LeRobot task and episode before distributed training. The scanner must identify
episodes that cannot satisfy the current training loader's data, timestamp, and
video requirements, record enough evidence for later manual deletion, and
continue after individual failures.

The scanner does not delete, rename, truncate, rewrite, or repair source data.

## Command and defaults

Add `scripts/scan_robocoin_episode_integrity.py` with these defaults:

- Input root: `/root/nas/code/d0/data/robot_data/robocoin`
- Output directory:
  `outputs/motus-multidataset_lap_v2/robocoin_full_scan`
- Worker count: a conservative CPU default, overridable with `--workers`
- Resume: enabled when an existing per-episode result file is supplied
- Video backend: PyAV from the `motus` environment

The command supports:

- `--root`
- `--output-dir`
- `--workers`
- `--task`
- `--episode`
- `--resume`
- `--overwrite`
- `--progress-interval`

`--task` and `--episode` provide a cheap targeted check for known failures such
as `Agilex_Split_Aloha_food_packaging/episode_000701`.

## Discovery

A task directory is eligible when it contains:

- `meta/info.json`
- `meta/episodes.jsonl`
- the declared parquet data hierarchy
- the declared video hierarchy when video features exist

The scanner reads the LeRobot path templates, chunk size, FPS, and video feature
keys from `meta/info.json`. It does not assume fixed camera names or a fixed
chunk size.

Each non-empty row in `meta/episodes.jsonl` creates one scan item containing the
task-relative path and `episode_index`. Discovery errors are reported as
task-level failures rather than terminating the whole scan.

## Episode validation

Validation is ordered from cheap checks to expensive checks.

### Metadata and parquet

For every episode:

1. Resolve the parquet path from `data_path`, `chunks_size`, and
   `episode_index`.
2. Confirm that the parquet exists and can be read.
3. Read the parquet row count plus `episode_index`, `frame_index`, and
   `timestamp`.
4. Confirm:
   - the parquet is non-empty;
   - all rows belong to the expected episode;
   - frame indices are monotonic and cover the declared episode;
   - timestamps are finite and monotonic;
   - the `episodes.jsonl` declared length matches the parquet row count.

Missing optional `language_action` is not by itself a bad episode because the
current loader falls back to standard VLM preprocessing when optional LAP text
is absent. An explicitly declared `language_action_path` that is missing, or a
present but unreadable/empty file, is reported as an auxiliary-data failure.
The same existence/readability check applies to an explicitly declared
`t5_embedding_path`; the scanner does not load the T5 model.

### Full video decode

For every video feature declared by the task:

1. Resolve the MP4 path from `video_path`.
2. Confirm it exists and can be opened by PyAV.
3. Decode the complete stream, counting frames and collecting decoded
   timestamps.
4. Confirm:
   - at least one frame decodes;
   - timestamps are finite and monotonic;
   - decoded FPS is compatible with task metadata;
   - decoded frame coverage includes every parquet timestamp that the current
     training sampling logic can request.

The coverage comparison uses the same effective tolerance as the current
LeRobot dataset object. A mismatch spanning multiple frames is never hidden by
inflating tolerance.

Because the current sampling logic may select any legal condition frame in an
episode, strict mode validates the union of all condition and predicted-video
timestamps, not one random `__getitem__` sample. This catches truncated tails
such as episode 701 deterministically.

### Training-shape checks

Without loading WAN, T5, or the VLM model, the scanner also validates fields
needed before model preprocessing:

- action column exists and is readable;
- state falls back according to the current eager RoboCOIN loader rules;
- action/state arrays have consistent row counts;
- feature-name metadata is compatible with the observed last dimension;
- language-action text, when present, is non-empty and line-readable;
- canonical55 named mapping can be constructed for the episode's state and
  action schema.

Model tokenization and GPU execution are explicitly outside this scanner's
scope.

## Results and resume

Write results incrementally as JSON Lines so an interrupted multi-hour scan can
resume without losing completed work:

- `episode_results.jsonl`: one terminal result per scanned episode
- `good_episodes.jsonl`: successful episode identities and measurements
- `bad_episodes.jsonl`: failures with deletion-oriented evidence
- `task_failures.jsonl`: failures that prevent episode discovery
- `summary.json`: aggregate counts by status, task, and error code
- `scan.log`: human-readable progress and exception summaries

Every episode result includes:

- task relative path;
- episode index;
- status;
- elapsed time;
- parquet path, row count, and timestamp bounds;
- per-camera path, decoded frame count, FPS, and timestamp bounds;
- stable error codes;
- concise error messages;
- traceback for unexpected exceptions.

Bad episodes can have multiple error codes. Example codes include:

- `missing_parquet`
- `invalid_parquet`
- `episode_length_mismatch`
- `invalid_frame_index`
- `invalid_timestamp`
- `missing_video`
- `video_decode_error`
- `video_frame_count_mismatch`
- `video_too_short`
- `video_fps_mismatch`
- `missing_declared_language_action`
- `invalid_language_action`
- `missing_declared_t5_embedding`
- `schema_mismatch`
- `canonical55_mapping_error`

Resume keys are `(task, episode_index)`. A resumed scan skips only episodes
whose previous result is terminal and parseable. `--overwrite` starts a fresh
report after refusing ambiguous combinations with `--resume`.

## Concurrency and robustness

Use a process pool because PyAV decode and parquet work are independent per
episode. The parent process alone writes reports, preventing interleaved JSONL
and summary corruption.

Workers receive immutable episode descriptors and return serializable result
objects. A worker exception becomes a failed episode result. Keyboard interrupt
stops submitting work, drains already returned results, writes the latest
summary, and exits non-zero.

The default worker count is intentionally conservative for NAS-backed input.
Users can increase it after observing storage throughput. No GPU is required.

## Exit codes

- `0`: scan completed and no bad episodes or task failures were found
- `1`: scan completed and found bad episodes or task failures
- `2`: command/configuration error or scan could not start
- `130`: interrupted by the user after persisting completed results

Finding bad data is therefore visible to automation while still producing the
complete reports.

## Testing

Unit and integration tests use temporary miniature LeRobot task directories.
They cover:

1. a valid episode;
2. a truncated video tail;
3. a missing video;
4. parquet/declared-length mismatch;
5. corrupt video decode;
6. optional missing language action accepted;
7. explicitly declared missing language action rejected;
8. incremental report writing and resume;
9. CLI targeted task/episode selection;
10. multi-error recording without aborting the remaining episodes.

Tests are written before implementation and observed failing for the missing
scanner behavior. After unit tests pass, run a targeted real-data check against
`Agilex_Split_Aloha_food_packaging/episode_000701` and verify that it is
classified as `video_too_short` with the measured parquet/video bounds.

The full 159k-episode scan is not part of the normal test suite because it is a
long-running data audit. The delivered command and expected output location
will be reported separately so it can be launched intentionally.

## Non-goals

- Automatically deleting or repairing episodes
- Rewriting `episodes.jsonl`
- Re-encoding video
- Increasing timestamp tolerance to mask corrupt data
- Loading WAN, T5, or VLM weights
- Proving a distributed training run succeeds
