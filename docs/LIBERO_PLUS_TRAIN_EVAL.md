# Motus LIBERO Training and LIBERO-plus Evaluation

This document only covers the current LIBERO training and LIBERO-plus evaluation workflow.

## 1. Server layout

```text
Motus repository:       /root/nas/code/d0
LIBERO training data:   /root/nasbak/cjy/robot_raw/libero
LIBERO-plus repository: /root/nas/code/LIBERO-plus
Motus environment:      /opt/conda/envs/motus
LIBERO-plus environment:/opt/conda/envs/liberoplus
LIBERO-plus config:      /root/.libero_plus/config.yaml
```

The policy and benchmark are decoupled. Motus runs as a policy server in the `motus` environment; the LIBERO-plus evaluator runs as a client in the `liberoplus` environment and communicates with the server over a local TCP port.

## 2. Current policy contract

```text
Action representation: raw OSC, 7 dimensions
Robot state:           8 dimensions
Action prediction:     16 steps (H16)
Action execution:      8 steps (E8)
Video frames:          8
Video/action ratio:    1:2
Image input:           third-person view above wrist view
Action normalization:  shared stats from the complete LIBERO training set
Validation split:      5%, seed 42
```

Training and evaluation must use the same config and action statistics. In particular, evaluating the new checkpoint with the old default config is incorrect.

## 3. Training

Current files:

```text
scripts/train_libero_lap_h16_gradacc4_40k.sh
configs/libero_raw_osc_lap_h16_gradacc4_40k.yaml
```

Current training scale:

```text
GPUs:                  8
Batch size per GPU:    5
Gradient accumulation: 4
Effective batch size:  160
Optimizer steps:       40000
Sample presentations:  6,400,000
Save/validation:       every 5000 steps
```

The script first regenerates raw OSC action statistics and H16 LAP labels, then starts distributed training. The cache is under:

```text
/root/nasbak/cjy/robot_raw/libero/motus_cache
```

Check that this directory is writable before starting. `Disk quota exceeded` is a storage quota error, not a GPU-memory error. The action statistics file currently exists and is non-empty.

Start training:

```bash
cd /root/nas/code/d0
bash scripts/train_libero_lap_h16_gradacc4_40k.sh
```

Training log:

```text
/root/nas/code/d0/outputs/motus-libero_raw_osc_h16/libero_raw_osc_h16_lap_after_pretrain_gradacc4_step40000/train.log
```

The current run completed normally at step 40000. The final validation metrics in the training log are `video_mse=0.0068`, `action_mse_loss=0.0303`, and `action_l2_error=0.0950`.

## 4. Current checkpoint

Use the DeepSpeed model directory, not `pytorch_model_0.bin`:

```bash
CKPT=/root/nas/code/d0/checkpoints/libero_raw_osc_lap_h16_gradacc4_40k/libero_raw_osc_h16_lap_after_pretrain_gradacc4_step40000/checkpoint_step_40000/pytorch_model
```

Checkpoints are available at steps 5000, 10000, 15000, 20000, 25000, 30000, 35000, and 40000.

## 5. Evaluation smoke test

Run one LIBERO-plus spatial task once before launching a full evaluation:

```bash
cd /root/nas/code/d0

CKPT=/root/nas/code/d0/checkpoints/libero_raw_osc_lap_h16_gradacc4_40k/libero_raw_osc_h16_lap_after_pretrain_gradacc4_step40000/checkpoint_step_40000/pytorch_model

bash scripts/run_libero_plus_eval.sh \
  --checkpoint "$CKPT" \
  --config configs/libero_raw_osc_lap_h16_gradacc4_40k.yaml \
  --task libero_spatial \
  --max-tasks 1 \
  --num-trials 1 \
  --run-name step40000_spatial_smoke
```

The evaluation chain was previously validated end to end with the step-20000 checkpoint. The new step-40000 checkpoint has not yet been evaluated, so its success rate should not be assumed before this smoke test passes.

Smoke-test result:

```text
outputs/libero_plus/step40000_spatial_smoke/results/libero_spatial/results.json
```

## 6. Four-GPU LIBERO-plus evaluation

The four suites are evaluated independently:

```text
libero_spatial: 2402 tasks
libero_object:  2518 tasks
libero_goal:    2591 tasks
libero_10:      2519 tasks
```

Example for `libero_spatial` on four GPUs:

```bash
cd /root/nas/code/d0

CKPT=/root/nas/code/d0/checkpoints/libero_raw_osc_lap_h16_gradacc4_40k/libero_raw_osc_h16_lap_after_pretrain_gradacc4_step40000/checkpoint_step_40000/pytorch_model

CONFIG_PATH=configs/libero_raw_osc_lap_h16_gradacc4_40k.yaml \
bash scripts/run_libero_plus_eval_multi_gpu.sh \
  --checkpoint "$CKPT" \
  --task libero_spatial \
  --gpu-ids 0,1,2,3 \
  --num-trials 1 \
  --video-mode failures \
  --output-root /root/nas/code/d0/outputs/libero_plus_4gpu \
  --run-name step40000_libero_spatial_4gpu
```

`CONFIG_PATH=...` is required here because the multi-GPU wrapper does not expose a `--config` argument and otherwise starts the policy server with the old 20000-step config.

To evaluate another suite, replace `libero_spatial` with `libero_object`, `libero_goal`, or `libero_10`. Start with `--num-trials 1`; this is the number of trials for every LIBERO-plus task, so larger values multiply the total episode count substantially.

Outputs:

```text
outputs/libero_plus_4gpu/<run-name>/results/<suite>/results.json
outputs/libero_plus_4gpu/<run-name>/logs/
```

Use `--resume` to continue an interrupted sharded evaluation. Use `--video-mode none` for maximum throughput, `failures` for failed episodes only, or `all` for every episode.

## 7. Quick checks and common failures

Check the two environments:

```bash
/opt/conda/envs/motus/bin/python -c "import torch; print(torch.cuda.is_available())"
/opt/conda/envs/liberoplus/bin/python -c "import libero, robosuite, wand; print('ok')"
```

Dry-run the four-GPU split without loading the model:

```bash
CONFIG_PATH=configs/libero_raw_osc_lap_h16_gradacc4_40k.yaml \
bash scripts/run_libero_plus_eval_multi_gpu.sh \
  --checkpoint "$CKPT" \
  --task libero_spatial \
  --gpu-ids 0,1,2,3 \
  --num-trials 1 \
  --dry-run
```

Common failures:

- `Checkpoint not found`: pass the `checkpoint_step_x/pytorch_model` directory.
- Policy server cannot start: inspect the corresponding `server_gpu*_port*.log` file.
- Port already in use: change `--base-port`, for example to `9983`.
- EGL initialization failure: confirm GPUs are attached and `MUJOCO_GL=egl` is available.
- Motion-blur/Wand failure: confirm `libmagickwand-dev` and the Wand Python package are available.
- Wrong model behavior with no load error: confirm the new config is supplied, H16/E8 is preserved, and the shared raw OSC action statistics file is used.
