# Simulation language-action and T5 sidecars

From the d0 project root, using the Python environment with torch, scipy,
pyarrow, h5py and the existing WAN dependencies:

```bash
python data/lerobot/prepare_sim_lap.py --stage preflight
python tests/test_prepare_sim_lap.py
python data/lerobot/prepare_sim_lap.py --limit 1 --max-episodes 2 \
  --output outputs/sim_lap_h16_smoke
mkdir -p outputs/sim_lap_h16
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -u data/lerobot/prepare_sim_lap.py --workers 8 \
  --output outputs/sim_lap_h16 > outputs/sim_lap_h16/run.log 2>&1
python data/lerobot/prepare_sim_lap.py --stage verify
```

The full command is resumable. `--datasets gr1 xemb cosmos` selects datasets;
`--stage language`, `--stage t5`, and `--stage verify` run individual stages.
`--limit` is a per-dataset task limit; `--max-episodes` limits work without
truncating metadata. Preflight is read-only. Other stages write reports.

## Contract

- Each task gets `language_action/episode_XXXXXX.txt` (one line per frame) and
  `t5_embedding/episode_XXXXXX.pt` (CPU `[S,4096]` tensor).
- `meta/episodes.jsonl` receives relative cache paths. Original tasks, parquet
  and video files are preserved. Metadata and replaced cache files are backed
  up with `.bak-sim-lap-<timestamp>` suffixes.
- Language windows are `[t,min(t+17,T))`: 17 pose samples, hence 16 position
  intervals. The last frame has zero motion. Descriptions report net motion,
  not total path length. Positions are converted from meters to centimeters.
- Rotations are `R_end * R_start.inv()`, decomposed as extrinsic XYZ angles in
  the source reference frame. Text uses explicit signed axes, avoiding an
  unsupported interpretation of world axes as robot forward/left.
- Panda hand text describes **changes in observed finger aperture** of at
  least 2 mm. It does not label stationary fingers as open/closed or assert
  grasp success. GR1 multi-finger open/close text is omitted and reported.

## Source adapters

- GR1: `trajectory_id` maps to the preserved HDF5 task and demo. The script
  checks the absolute/base/axis-angle controller configuration, right/left
  site order, frame counts, and both hand command trajectories against the
  44D LeRobot action. Motion describes absolute end-effector **targets**.
  A missing v2.1 parquet may be read from original LeRobot only after metadata
  identity checks; the normal trajectory checks still apply. This does not
  repair the missing v2.1 training file.
- X-Embodiment: absolute world EEF pose at state `[7:10]` / `[13:17]`, xyzw.
  Base and relative pose composition is checked on every frame to validate
  quaternion order and field interpretation. Finger aperture uses `[21:23]`.
- Cosmos: `observation.ee_pos` and axis-angle `observation.ee_ori`, world frame;
  concatenation is checked against `observation.ee_states`. Finger aperture
  uses state `[7:9]`. Axis-angle interpretation follows the
  [official regeneration code](https://github.com/NVlabs/cosmos-policy/blob/main/cosmos_policy/experiments/robot/robocasa/regenerate_robocasa_dataset.py)
  and was cross-checked against raw HDF5 `robot_states` quaternions.

## T5 and provenance

GR1 uses full `remarks`; the other datasets use `tasks[0]`. Empty instructions
fail explicitly. Cached tensors are compared against fresh encoding of the
full instruction (`atol=0.02, rtol=0.01`) before existing episode files are
reused. Original task labels are not rewritten: training integrations must
use the recorded full instruction if textual/T5 conditioning must agree.

One encoder is reused; unique instructions are encoded in batches. A pool
under the report directory is keyed by encoder identity and instruction hash.
Episode files are hard-linked to the pool where possible, copied across
filesystems otherwise. Treat caches as immutable; regenerate via atomic
replacement rather than editing a tensor file in place.

Per-task `meta/sim_lap_language.jsonl` records input fingerprints, source
semantics, window, and output SHA256. `meta/sim_lap_t5.jsonl` records exact text,
instruction hash, encoder identity and shape. Failures go to
`meta/sim_lap_language_failures.jsonl`. Full inventory and stage reports are
under `outputs/sim_lap_h16/`. Success requires a zero-failure final verification,
not merely a running process or completed encoder initialization.
