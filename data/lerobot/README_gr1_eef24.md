# GR1 native end-effector actions (LeRobot v2.1)

`PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_v21_eef24` is a separate,
source-preserving derivative of the active `LeRobot_v21` episode manifest.
It does not restore excluded episodes. Original HDF5 and LeRobot data remain unchanged.

## Action contract

| Slice | Meaning |
|---|---|
| `[0:3]` | Right end-effector absolute target position, meters |
| `[3:6]` | Right end-effector absolute target rotation vector, radians |
| `[6:9]` | Left end-effector absolute target position, meters |
| `[9:12]` | Left end-effector absolute target rotation vector, radians |
| `[12:18]` | Right hand six original control values |
| `[18:24]` | Left hand six original control values |

Positions and orientations use the robot base frame. These are controller
targets, not measured poses or frame-to-frame deltas. Hands retain their
native controls and are not reduced to a scalar grasp signal. `action` is
float32; `observation.state` remains the original 44D joint observation.
Other parquet columns and episode numbering are preserved.

`meta/modality.json`, action feature names, per-episode and aggregate statistics
describe the new contract. `meta/eef24_sources.jsonl` records source/demo mapping
and parquet hashes. Each task has a `COMPLETE.json` written only after verification.
Stale source joint-action metadata (`initial_actions.npz`, `metadata.json`) is not copied.

Videos, LAP text and T5 tensors are hardlinked (copied only across filesystems).
Treat these shared files as immutable. LAP still describes target-pose motion
over a 17-pose / 16-action-step window, omitting hand descriptions; T5 still encodes
full `remarks`. Neither cache changes when retaining the same HDF5 targets.

## Conversion and verification

From the d0 root, with the robot6 dependency environment:

```bash
export PYTHONPATH=/root/nasbak/cjy/robot_raw/motus_robot6_python_deps:$PWD
python scripts/data/convert_gr1_eef24.py --workers 8
python scripts/data/convert_gr1_eef24.py --workers 8 --verify \
  --report outputs/gr1_eef24_conversion/verify.json
python scripts/data/build_robot6_sim_stats.py \
  --config configs/multidataset_lap_robot6_24a_eef_norm.yaml \
  --output data/utils/robot6_sim_eef24_delta_leftfirst_stats.json
CONFIG=configs/multidataset_lap_robot6_24a_eef_norm.yaml \
OUTPUT_DIR=outputs/robot6_eef24_preflight python scripts/data/check_robot6_batches.py
```

Conversion checks HDF5 controller settings, arm ordering, frame/index/timestamp
alignment and both hand trajectories against the 44D source. Existing output
is compared before reuse; discrepancies fail rather than overwrite silently.
`--verify` recomputes comparisons and statistics for all selected episodes.
Use separate `--output` and `--report` locations for `--max-episodes` experiments.

## Training

The training config uses `sim_kind: gr1_eef24_delta` with 24 valid actions and 14 selected
arm-joint state values. Other five sources retain 7 valid actions, padded to
24 with loss masking. State inputs remain padded to 14. This is dimensional
padding, not a conversion of all sources into a shared control convention.

```bash
# 8 GPUs, microbatch 5, global batch 40, 100 steps, validation and checkpoint
bash scripts/mutildataset/train_robot6_24a_eef_norm_8gpu.sh
# Full training (explicitly opt in; not automatically run after smoke)
SMOKE=0 bash scripts/mutildataset/train_robot6_24a_eef_norm_8gpu.sh
```

Use a fresh RUN_NAME and OUTPUT_DIR for another smoke. Start from the configured
Motus pretraining checkpoint; the existing partial loader skips action input
and output adapters. Never resume a joint-action smoke checkpoint. The old
14D dataset mapping, config, statistics and launcher remain available.

For this conversion's acceptance run, smoke weights/optimizer/TensorBoard
artifacts are removed only after successful validation and checkpoint readback.
Small logs, resolved config, audit results and cleanup records remain under
`outputs/robot6_eef24_smoke`; final data and source files are not cleanup targets.

## Consecutive target deltas and four-node HyperTrain

The stored LeRobot action contract above remains absolute. Before normalization,
the loader converts both arm poses to consecutive target increments:
`delta_p[t] = p[t] - p[t-1]` and
`delta_R[t] = R[t] @ inverse(R[t-1])`, represented as a rotation vector in radians.
After computing deltas in native order, training reorders to left translation
`[0:3]`, left rotation `[3:6]`, left hand `[6:12]`, right translation `[12:15]`,
right rotation `[15:18]`, right hand `[18:24]`. Hands are unchanged.
The stored absolute Parquet order described above is unchanged.
Checkpoints trained with the previous right-first layout are not compatible with this ordering.
For a sample conditioned on frame t, the first action is target t+1 minus target t.
No difference crosses an episode boundary. Repeated final frames produce exactly
zero physical pose increments while retaining hand commands. Stride must be 1.
At inference, accumulate increments from the previous target in the base frame;
they are not measured-pose control errors. Normalized zeros need not equal physical zeros.

`data/utils/robot6_sim_eef24_delta_leftfirst_stats.json` contains rebuilt statistics from all
17,999 active GR1 episodes (4,480,196 frames), with a distinct transform identity
to prevent accidental reuse of absolute-pose normalization. LAP and T5 are reused.

On HyperTrain allocate four nodes with eight GPUs each. Run this same command
once per node, with platform-provided `MASTER_ADDR` (node 0 reachable address),
common `MASTER_PORT`, and distinct `NODE_RANK` values 0, 1, 2, 3:

```bash
bash /root/nas/code/d0/scripts/mutildataset/train_robot6_24a_eef_delta_4node_hypertrain.sh
```

Defaults: 32 ranks, batch 5 per rank, accumulation 1, global batch 160,
BF16, ZeRO-2, 200,000 steps. All nodes need the same code, dataset/model mounts
and Python dependencies; override `PROJECT_ROOT`, `PYTHON`, `ROBOT6_DEPS` if needed.
The script validates local asset paths and GPU count before launch.
`DRY_RUN=1` prints the command without training. The four-node launcher runs
the selected configuration directly and has no smoke or temporary-config branch.
The HyperTrain launcher activates `conda motus` and accepts `CONFIG_FILE` (or legacy `CONFIG`).
Logs: `outputs/motus-multidataset_lap_robot6_24a_norm_lap_hypertrain/train_lap_hypertrain_node{0,1,2,3}.log`.
Checkpoints follow the config basename and run name: `checkpoints/multidataset_lap_robot6_24a_norm/multidataset_lap_robot6_24a_norm_lap_hypertrain/checkpoint_step_*`.
Use a fresh common `RUN_NAME` and `OUTPUT_DIR` on retries; no old checkpoint resume.
The previous 8-GPU 100-step acceptance run tested absolute poses, not this delta
configuration. Four-node execution requires validation on the allocated cluster.
