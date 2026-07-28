# RoboCOIN Full-Load Integrity Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a read-only, resumable scanner that reads every RoboCOIN episode parquet and completely decodes every declared video stream once, while reporting all data-loading failures without stopping at the first bad episode.

**Architecture:** A single CLI module owns typed discovery, per-episode validation, multiprocessing orchestration, and parent-only incremental reporting. Workers validate immutable episode descriptors from cheap metadata/parquet checks through full PyAV decode; they return serializable results, while the parent writes JSONL and summaries so interrupted scans can resume safely.

**Tech Stack:** Python 3.10, PyArrow 24, PyAV 17, PyTorch 2.4, pytest, LeRobot v2 metadata

## Global Constraints

- Input defaults to `/root/nas/code/d0/data/robot_data/robocoin`, whose resolved target is the local RoboCOIN corpus.
- Reports default to `outputs/motus-multidataset_lap_v2/robocoin_full_scan`.
- Scan every row discovered from every valid `meta/episodes.jsonl`; do not use random `dataset[i]` sampling.
- Read each episode parquet and fully decode every video feature declared in `meta/info.json`.
- Match the eager v2 loader's `0.0001` second timestamp tolerance; do not inflate tolerance to hide truncated videos.
- Missing optional conventional `language_action` is valid; an explicitly declared missing file and any present empty/unreadable file are failures.
- Do not load WAN, T5, or VLM weights. An explicitly declared T5 file is checked for existence/readability but not model inference.
- Do not delete, rename, rewrite, truncate, repair, or re-encode source data.
- Continue after task-level and episode-level failures, persist results incrementally, and return non-zero when bad data is found.
- Preserve unrelated changes in the dirty worktree.

---

### Task 1: Add Discovery and Path-Resolution Contracts

**Files:**
- Create: `scripts/scan_robocoin_episode_integrity.py`
- Create: `tests/test_scan_robocoin_episode_integrity.py`

**Interfaces:**
- Produces: `EpisodeSpec`, `Issue`, `EpisodeResult`, and `TaskFailure` frozen dataclasses.
- Produces: `discover_episodes(root: Path, task_filter: str | None, episode_filter: int | None) -> tuple[list[EpisodeSpec], list[TaskFailure]]`.
- Produces: `resolve_episode_path(task_root: Path, template: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> Path`.
- Consumes: LeRobot `meta/info.json` and `meta/episodes.jsonl`.

- [ ] **Step 1: Write miniature-corpus test helpers**

In `tests/test_scan_robocoin_episode_integrity.py`, add `_write_jsonl(...)` and `_make_task(...)`. The fixture must use the real LeRobot path templates and named RoboCOIN fields:

```python
info = {
    "fps": 30,
    "chunks_size": 1000,
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": (
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4"
    ),
    "features": {
        "observation.images.cam_high_rgb": {"dtype": "video"},
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": ["left_arm_joint_1_rad", "right_arm_joint_1_rad"],
        },
        "action": {
            "dtype": "float32",
            "shape": [2],
            "names": ["left_arm_joint_1_rad", "right_arm_joint_1_rad"],
        },
        "timestamp": {"dtype": "float32", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
    },
}
```

The helper creates metadata only in this task; later tasks extend it with parquet and video writers.

- [ ] **Step 2: Write failing discovery tests**

Cover deterministic task ordering, episode ordering, chunk calculation, task filtering, episode filtering, malformed JSON, and a task missing `meta/info.json`. Assert malformed or incomplete tasks become `TaskFailure` entries while valid tasks remain discoverable:

```python
episodes, failures = scanner.discover_episodes(root, None, None)

assert [(item.task, item.episode_index) for item in episodes] == [
    ("task_a", 0),
    ("task_b", 1001),
]
assert Path(episodes[1].parquet_path) == (
    root / "task_b/data/chunk-001/episode_001001.parquet"
)
assert [failure.code for failure in failures] == ["invalid_task_metadata"]
```

- [ ] **Step 3: Run discovery tests and verify RED**

Run:

```bash
python -m pytest tests/test_scan_robocoin_episode_integrity.py \
  -k 'discover or resolve' -vv
```

Expected: collection or import fails because the scanner module and interfaces do not exist.

- [ ] **Step 4: Implement typed discovery and strict template expansion**

Implement frozen dataclasses with JSON-safe primitives. `EpisodeSpec` must carry:

```python
@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class TaskFailure:
    task: str
    code: str
    message: str
    traceback: str | None = None


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    episode_index: int
    status: str
    elapsed_s: float
    parquet: dict[str, object] | None
    videos: tuple[dict[str, object], ...]
    issues: tuple[Issue, ...]
    traceback: str | None = None


@dataclass(frozen=True)
class EpisodeSpec:
    root: str
    task: str
    episode_index: int
    declared_length: int
    fps: float
    tolerance_s: float
    parquet_path: str
    videos: tuple[tuple[str, str], ...]
    action_key: str
    action_names: tuple[str, ...]
    state_key: str
    state_names: tuple[str, ...]
    language_action_path: str | None
    language_action_declared: bool
    t5_embedding_path: str | None
    t5_embedding_declared: bool
```

Expand `{episode_chunk}`, `{episode_index}`, and `{video_key}` using:

```python
episode_chunk = episode_index // chunks_size
relative = template.format(
    episode_chunk=episode_chunk,
    episode_index=episode_index,
    video_key=video_key,
)
```

Reject non-positive `chunks_size`, missing `fps`, missing templates, duplicate `(task, episode_index)` keys, and unsafe paths resolving outside the task root. Choose `action`, falling back to `actions`; choose `observation.state`, falling back to the action key exactly as the eager RoboCOIN loader does.

- [ ] **Step 5: Run discovery tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_scan_robocoin_episode_integrity.py \
  -k 'discover or resolve' -vv
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the discovery unit**

```bash
git add scripts/scan_robocoin_episode_integrity.py \
  tests/test_scan_robocoin_episode_integrity.py
git commit -m "feat: discover RoboCOIN scan episodes"
```

---

### Task 2: Validate Parquet, Schema, Auxiliary Data, and Canonical55 Mapping

**Files:**
- Modify: `scripts/scan_robocoin_episode_integrity.py`
- Modify: `tests/test_scan_robocoin_episode_integrity.py`

**Interfaces:**
- Produces: `validate_parquet(spec: EpisodeSpec) -> tuple[ParquetMeasurement | None, list[float], list[Issue]]`.
- Produces: `validate_auxiliary(spec: EpisodeSpec) -> list[Issue]`.
- Produces internal helper: `validate_canonical55(spec: EpisodeSpec, table: pyarrow.Table) -> list[Issue]`, called before `validate_parquet` returns.
- Consumes: `data.canonical55.map_robocoin_named_vector(source, names)`.

- [ ] **Step 1: Extend the fixture with a real parquet writer**

Write four rows by default:

```python
table = pa.table({
    "observation.state": [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
    "action": [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5], [1.6, 1.7]],
    "timestamp": [0.0, 1 / 30, 2 / 30, 3 / 30],
    "frame_index": [0, 1, 2, 3],
    "episode_index": [episode_index] * 4,
})
pq.write_table(table, parquet_path)
```

Allow tests to override columns, row count, declared length, language-action metadata, and T5 metadata.

- [ ] **Step 2: Write failing parquet and schema tests**

Add parameterized tests for these exact codes:

- missing parquet: `missing_parquet`
- unreadable parquet: `invalid_parquet`
- zero rows: `empty_parquet`
- declared length mismatch: `episode_length_mismatch`
- wrong episode index: `episode_index_mismatch`
- non-contiguous/non-monotonic frame index: `invalid_frame_index`
- NaN or non-monotonic timestamp: `invalid_timestamp`
- missing action: `missing_action`
- null action/state rows or incompatible observed vector widths: `schema_mismatch`
- feature names longer than the observed last dimension: `schema_mismatch`
- canonical mapping exception: `canonical55_mapping_error`

Assert all applicable issues are accumulated rather than only returning the first one:

```python
measurement, timestamps, issues = scanner.validate_parquet(spec)
assert measurement is not None
assert len(timestamps) == measurement.rows
assert {issue.code for issue in issues} >= {
    "episode_length_mismatch",
    "invalid_timestamp",
}
```

- [ ] **Step 3: Write failing auxiliary-data tests**

Cover:

```python
assert scanner.validate_auxiliary(optional_missing_spec) == []
assert [issue.code for issue in scanner.validate_auxiliary(declared_missing_spec)] == [
    "missing_declared_language_action"
]
assert [issue.code for issue in scanner.validate_auxiliary(empty_language_action_spec)] == [
    "invalid_language_action"
]
assert [issue.code for issue in scanner.validate_auxiliary(missing_t5_spec)] == [
    "missing_declared_t5_embedding"
]
```

A readable declared T5 file only needs to be opened in binary mode and have non-zero size; do not call `torch.load`.

- [ ] **Step 4: Run data-validation tests and verify RED**

Run:

```bash
python -m pytest tests/test_scan_robocoin_episode_integrity.py \
  -k 'parquet or schema or canonical or auxiliary' -vv
```

Expected: failures because validation functions do not exist.

- [ ] **Step 5: Implement parquet and schema validation**

Read only required columns when the schema permits, call the canonical helper
before returning, and return finite timestamp values for video coverage checks.
Define:

```python
@dataclass(frozen=True)
class ParquetMeasurement:
    path: str
    rows: int
    timestamp_min: float | None
    timestamp_max: float | None
    frame_index_min: int | None
    frame_index_max: int | None
```

Use NumPy for finite/monotonic checks. Require frame indices to equal
`np.arange(row_count)` and every `episode_index` value to equal the descriptor.
Validate null rows and observed vector widths against `info.json` names. Call
`map_robocoin_named_vector` once for the full state matrix and once for the full
action matrix, then assert both outputs end in dimension 55 and masks contain
at least one valid slot.

- [ ] **Step 6: Implement auxiliary validation**

Treat only a missing conventional, undeclared language-action file as optional. For declared or present text, open UTF-8, strip blank lines, and emit `invalid_language_action` if decoding fails or no non-empty line remains. For a declared T5 path, require a readable non-empty regular file.

- [ ] **Step 7: Run data-validation tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_scan_robocoin_episode_integrity.py \
  -k 'parquet or schema or canonical or auxiliary' -vv
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the non-video validator**

```bash
git add scripts/scan_robocoin_episode_integrity.py \
  tests/test_scan_robocoin_episode_integrity.py
git commit -m "feat: validate RoboCOIN episode records"
```

---

### Task 3: Fully Decode and Validate Every Declared Video

**Files:**
- Modify: `scripts/scan_robocoin_episode_integrity.py`
- Modify: `tests/test_scan_robocoin_episode_integrity.py`

**Interfaces:**
- Produces: `decode_video(path: Path) -> VideoMeasurement`.
- Produces: `validate_videos(spec: EpisodeSpec, parquet: ParquetMeasurement, query_timestamps: Sequence[float]) -> tuple[list[VideoMeasurement], list[Issue]]`.
- Produces: `scan_episode(spec: EpisodeSpec) -> EpisodeResult`.
- Consumes: PyAV `av.open`, stream `time_base`, frame `pts`, and frame `time`.

- [ ] **Step 1: Add a deterministic MP4 fixture**

Create a tiny RGB MP4 using PyAV:

```python
def _write_video(path: Path, frame_count: int, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            image = np.full((16, 16, 3), index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
```

Mark the tests skipped only when `av` itself is unavailable; do not silently skip codec/decode failures in the Motus environment.

- [ ] **Step 2: Write failing full-decode tests**

Cover:

- valid four-frame stream;
- missing stream: `missing_video`;
- corrupt bytes: `video_decode_error`;
- zero decoded frames: `video_decode_error`;
- three decoded frames for four parquet timestamps: `video_frame_count_mismatch` and `video_too_short`;
- declared FPS differing from measured FPS beyond `max(0.01, fps * 0.01)`: `video_fps_mismatch`;
- every declared video key is decoded, not only the eager loader's preferred camera key.

For the valid case assert:

```python
result = scanner.scan_episode(spec)
assert result.status == "good"
assert result.videos[0]["decoded_frames"] == 4
assert result.videos[0]["timestamp_max"] >= 3 / 30 - spec.tolerance_s
```

- [ ] **Step 3: Run video tests and verify RED**

Run:

```bash
/opt/conda/envs/motus/bin/python -m pytest \
  tests/test_scan_robocoin_episode_integrity.py -k 'video or scan_episode' -vv
```

Expected: failures because complete decoding and coverage checks are not implemented.

- [ ] **Step 4: Implement complete PyAV decoding**

Define:

```python
@dataclass(frozen=True)
class VideoMeasurement:
    key: str
    path: str
    decoded_frames: int
    fps: float | None
    timestamp_min: float | None
    timestamp_max: float | None
```

Open the container, select exactly one video stream, iterate `container.decode(stream)` to EOF, and count every decoded frame. Prefer `float(frame.time)`; otherwise use `float(frame.pts * stream.time_base)`. Reject missing/non-finite/non-monotonic decoded timestamps. Measure FPS from `stream.average_rate` when positive, otherwise from timestamp deltas.

- [ ] **Step 5: Implement deterministic coverage validation**

For each parquet timestamp, require at least one decoded timestamp within `spec.tolerance_s` using sorted timestamps and `np.searchsorted`. Emit `video_too_short` when the maximum query timestamp lies beyond decoded coverage, and `video_frame_count_mismatch` whenever decoded frame count differs from parquet row count. Do not suppress either code when both apply.

`scan_episode` must always run auxiliary and parquet checks; it runs video checks when query timestamps were readable. Unexpected exceptions become `Issue(code="unexpected_error", ...)` with traceback, and the returned result status is `"bad"`.

- [ ] **Step 6: Run video tests and verify GREEN**

Run:

```bash
/opt/conda/envs/motus/bin/python -m pytest \
  tests/test_scan_robocoin_episode_integrity.py -k 'video or scan_episode' -vv
```

Expected: all selected tests pass and each fixture video is decoded to EOF.

- [ ] **Step 7: Commit the full video validator**

```bash
git add scripts/scan_robocoin_episode_integrity.py \
  tests/test_scan_robocoin_episode_integrity.py
git commit -m "feat: fully decode RoboCOIN episode videos"
```

---

### Task 4: Add Incremental Reports, Resume, CLI, and Process Isolation

**Files:**
- Modify: `scripts/scan_robocoin_episode_integrity.py`
- Modify: `tests/test_scan_robocoin_episode_integrity.py`

**Interfaces:**
- Produces: `ReportWriter(output_dir: Path, overwrite: bool)`.
- Produces: `load_terminal_keys(path: Path) -> set[tuple[str, int]]`.
- Produces: `run_scan(args: argparse.Namespace) -> int`.
- Produces CLI options: `--root`, `--output-dir`, `--workers`, `--task`, `--episode`, `--resume`, `--overwrite`, and `--progress-interval`.

- [ ] **Step 1: Write failing report and resume tests**

Assert one terminal JSON object is appended and flushed per episode; good and bad subsets are also written; task failures are separate; summary counts match:

```python
summary = json.loads((output_dir / "summary.json").read_text())
assert summary["episodes_discovered"] == 3
assert summary["episodes_scanned"] == 3
assert summary["good"] == 2
assert summary["bad"] == 1
assert summary["errors_by_code"]["missing_video"] == 1
```

Create an `episode_results.jsonl` containing one valid terminal row followed by
one truncated line. Assert resume loads the valid key once, ignores the
truncated tail with a warning, and does not rescan it. Separately assert
`ReportWriter` refuses to append a second terminal result for a key it already
recorded.

- [ ] **Step 2: Write failing CLI and interruption tests**

Use `subprocess.run` for `--help`, invalid worker count, mutually exclusive `--resume/--overwrite`, and targeted `--task/--episode`. Monkeypatch the executor so one worker raises and assert the parent records `unexpected_error` and continues. Raise `KeyboardInterrupt` after one result and assert summary/results are flushed with exit code `130`.

- [ ] **Step 3: Run orchestration tests and verify RED**

Run:

```bash
python -m pytest tests/test_scan_robocoin_episode_integrity.py \
  -k 'report or resume or cli or interrupt or worker' -vv
```

Expected: failures because reporting and orchestration are absent.

- [ ] **Step 4: Implement parent-only incremental reporting**

`ReportWriter.record_episode(result)` writes one line to `episode_results.jsonl` and the matching `good_episodes.jsonl` or `bad_episodes.jsonl`, then flushes each active file. Rewrite `summary.json` atomically through a file in the same output directory followed by `Path.replace`. Include:

```json
{
  "episodes_discovered": 0,
  "episodes_skipped_resume": 0,
  "episodes_scanned": 0,
  "good": 0,
  "bad": 0,
  "task_failures": 0,
  "errors_by_code": {},
  "tasks": {}
}
```

On resume, rebuild `episodes_scanned`, `good`, `bad`,
`errors_by_code`, and per-task counts from unique terminal rows already present,
then increment `episodes_skipped_resume` for descriptors omitted during this
invocation. Thus `episodes_scanned` always means the total unique terminal
results in the report, including results from prior invocations.

Only the parent process may own report file handles.
Configure a `logging.FileHandler(output_dir / "scan.log")` in the parent and a
stderr handler for live progress. Workers return exception text and traceback
inside results instead of opening the log file.

- [ ] **Step 5: Implement bounded process-pool orchestration**

Validate `workers >= 1`, default to `min(4, os.cpu_count() or 1)`, and keep at most `workers * 2` futures in flight so 159k descriptors do not become 159k queued futures. Refill the queue as futures complete. Catch worker exceptions around `future.result()` and convert them to bad episode results. On `KeyboardInterrupt`, stop submitting, cancel pending futures, persist the summary, and return `130`.

- [ ] **Step 6: Implement CLI exit contracts**

Return:

- `0` when the scan completes with no bad episodes or task failures;
- `1` when the scan completes and finds bad data;
- `2` for invalid arguments, inaccessible root, or output initialization failure;
- `130` after a persisted interruption.

Reject `--resume` with `--overwrite`. Refuse to write into a non-empty report directory unless one of those flags is present. Log progress every `--progress-interval` completed episodes with scanned/good/bad counts, elapsed time, and episodes per second.

- [ ] **Step 7: Run orchestration tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_scan_robocoin_episode_integrity.py \
  -k 'report or resume or cli or interrupt or worker' -vv
```

Expected: all selected tests pass.

- [ ] **Step 8: Run the complete scanner test suite**

Run:

```bash
/opt/conda/envs/motus/bin/python -m pytest \
  tests/test_scan_robocoin_episode_integrity.py -q
```

Expected: zero failures.

- [ ] **Step 9: Commit the operational CLI**

```bash
git add scripts/scan_robocoin_episode_integrity.py \
  tests/test_scan_robocoin_episode_integrity.py
git commit -m "feat: add resumable RoboCOIN integrity scan"
```

---

### Task 5: Verify Against Known Real Data and Complete the Full Scan

**Files:**
- Verify: `scripts/scan_robocoin_episode_integrity.py`
- Verify: `tests/test_scan_robocoin_episode_integrity.py`
- Generate: `outputs/motus-multidataset_lap_v2/robocoin_full_scan/episode_results.jsonl`
- Generate: `outputs/motus-multidataset_lap_v2/robocoin_full_scan/good_episodes.jsonl`
- Generate: `outputs/motus-multidataset_lap_v2/robocoin_full_scan/bad_episodes.jsonl`
- Generate: `outputs/motus-multidataset_lap_v2/robocoin_full_scan/task_failures.jsonl`
- Generate: `outputs/motus-multidataset_lap_v2/robocoin_full_scan/summary.json`
- Generate: `outputs/motus-multidataset_lap_v2/robocoin_full_scan/scan.log`

**Interfaces:**
- Consumes CLI delivered by Task 4.
- Produces a terminal result for every discovered `(task, episode_index)` and a reconciled aggregate summary.

- [ ] **Step 1: Run static and focused regression checks**

Run:

```bash
/opt/conda/envs/motus/bin/python -m py_compile \
  scripts/scan_robocoin_episode_integrity.py
python -m pytest \
  tests/test_scan_robocoin_episode_integrity.py \
  tests/test_lerobot_lazy_datasets.py \
  tests/test_canonical55.py -q
git diff --check -- \
  scripts/scan_robocoin_episode_integrity.py \
  tests/test_scan_robocoin_episode_integrity.py
```

Expected: all commands exit successfully.

- [ ] **Step 2: Run a known targeted real-data check**

Run:

```bash
/opt/conda/envs/motus/bin/python \
  scripts/scan_robocoin_episode_integrity.py \
  --root /root/nas/code/d0/data/robot_data/robocoin \
  --output-dir /tmp/robocoin_episode_701_scan \
  --task Agilex_Split_Aloha_food_packaging \
  --episode 701 \
  --workers 1 \
  --overwrite
```

Expected: the command completes without an unhandled traceback. If the previously observed truncation is still present, `bad_episodes.jsonl` contains episode 701 with `video_too_short` and measured parquet/video timestamp bounds. If the source was repaired, record the current measurements and do not force the historical classification.

- [ ] **Step 3: Start the complete read-only scan**

Run with a conservative NAS load:

```bash
/opt/conda/envs/motus/bin/python \
  scripts/scan_robocoin_episode_integrity.py \
  --root /root/nas/code/d0/data/robot_data/robocoin \
  --output-dir outputs/motus-multidataset_lap_v2/robocoin_full_scan \
  --workers 4 \
  --progress-interval 100 \
  --overwrite
```

Expected: all discovered episodes are submitted exactly once, results appear incrementally, and no source files change.

- [ ] **Step 4: Resume after any interruption**

If the process is interrupted, rerun exactly:

```bash
/opt/conda/envs/motus/bin/python \
  scripts/scan_robocoin_episode_integrity.py \
  --root /root/nas/code/d0/data/robot_data/robocoin \
  --output-dir outputs/motus-multidataset_lap_v2/robocoin_full_scan \
  --workers 4 \
  --progress-interval 100 \
  --resume
```

Expected: previously terminal `(task, episode_index)` keys are skipped and no duplicate terminal results are appended.

- [ ] **Step 5: Reconcile terminal counts**

Run:

```bash
/opt/conda/envs/motus/bin/python - <<'PY'
import json
from pathlib import Path

out = Path("outputs/motus-multidataset_lap_v2/robocoin_full_scan")
summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
rows = [
    json.loads(line)
    for line in (out / "episode_results.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
keys = {(row["task"], row["episode_index"]) for row in rows}
assert len(rows) == len(keys), (len(rows), len(keys))
assert summary["episodes_scanned"] == len(rows), summary
assert summary["episodes_discovered"] == summary["episodes_scanned"], summary
assert summary["good"] + summary["bad"] == summary["episodes_scanned"], summary
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

Expected: all assertions pass. For a non-resumed uninterrupted run,
`episodes_skipped_resume` is zero. For a resumed run it records how many
descriptors were already terminal at startup without changing the unique
terminal total.

- [ ] **Step 6: Inspect failure evidence and report the validation boundary**

Summarize counts by stable error code from `summary.json`, include the first few concrete bad episode paths from `bad_episodes.jsonl`, and state explicitly:

- every discovered parquet was read;
- every declared video stream was decoded to EOF;
- bad episodes were reported without deletion;
- model preprocessing and distributed training were not exercised.

Do not claim the dataset is clean unless exit code is `0`, task failures are zero, bad episodes are zero, and the reconciliation assertions pass.
