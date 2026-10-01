# Current selection: original joint actions

The mixed config now selects `RoboDojo_joint24` with `action_signal: action_joint`.
It uses the original LeRobot 14D absolute joint targets without pose conversion
or differencing: left joints in 0:6, left gripper at 6, right joints in 12:18,
right gripper at 18. Other slots are zero and masked. States retain the original
14D left-joints/gripper/right-joints/gripper ordering. Both signals have independently
computed source min/max statistics. Repeated tail frames hold absolute targets.
`prepare_robodojo_joint24.py` checks original action[t] against state[t+1] for all
included episodes and reuses the verified media/LAP/T5 mapping. One episode with
incomplete media-side caches remains excluded. The original raw files are unchanged.
The existing four-node launcher loads this updated selection. Previous EEF output
below remains available but is not selected by the current config.

# RoboDojo in the 24D mixed training configuration

`configs/multidataset_lap_robot6_24a_eef_norm.yaml` now has seven equal-weight
sources. The retained filename keeps the existing four-node launcher working.
Only Bridge participates in validation. No training is launched by preprocessing.

RoboDojo uses 14 valid dimensions in the 24D action vector:

| Slots | Control |
|---|---|
| 0:3 | Left translation increment (meters) |
| 3:6 | Left relative rotation vector (radians) |
| 6 | Left continuous gripper target |
| 7:12 | Zero, masked from action loss |
| 12:15 | Right translation increment |
| 15:18 | Right relative rotation vector |
| 18 | Right continuous gripper target |
| 19:24 | Zero, masked from action loss |

The local derivative is `data/robot_data/RoboDojo_eef_delta24`.
`stats.json` has separate min/max for 14 native action channels and 14 joint-state
channels. Normalize first, then scatter action channels into the 24D slots.
`observation.state` remains left joints 6 + gripper + right joints 6 + gripper.
Grippers are continuous and not differenced or thresholded.

Action source is the local RoboDojo EEF v3 dataset's original target action.
`action[t]` targets frame `t+1`; the loader offsets target-frame indices by -1.
Deltas are consecutive targets, with the initial observed pose as the reference
for the first target of an episode. Rotations use `R_current @ R_previous.inv()`.
Translation and rotation vectors are transformed from world coordinates to the
robot-aligned convention (x=forward, y=left, z=up) with
`[[0,1,0],[-1,0,0],[0,0,1]]`, matching the canonical LAP sidecars.
This is an axis convention, not a claim of identical origins/control dynamics
across embodiments. Clamped repeated tail frames have zero physical pose delta
and held gripper targets. Stride is restricted to 1.

The preparation script verifies every included trajectory against existing
absolute EEF observations and raw gripper states before reusing videos/T5.
It links existing videos, qpos, full-instruction T5, and canonical 16-step LAP;
it writes only small target-delta tensors and metadata. Existing datasets remain
unchanged. `COMPLETE.json` reports included scale and any missing-asset exclusions;
`manifest.json` records the global-to-task-local episode mapping.

```bash
PYTHONPATH=/root/nasbak/cjy/robot_raw/motus_robot6_python_deps:$PWD \
  /opt/conda/bin/python scripts/data/prepare_robodojo_delta24.py
```

The completed output is not overwritten. Use `--output` for a separate rebuild.
A seven-source real-batch check is available through `scripts/data/check_robot6_batches.py`
with `CONFIG=configs/multidataset_lap_robot6_24a_eef_norm.yaml`.
Use a fresh run name when starting the updated mixture; already-running jobs
retain their original dataset configuration.
