# Motus Training Guide

This document explains the training code layout, how to create and edit a
training configuration, and how to launch Motus on a workstation or a
distributed cluster. Run all commands from the repository root unless noted
otherwise.

## Contents

- [Quick start](#quick-start)
- [Code architecture](#code-architecture)
- [Training configuration](#training-configuration)
- [DeepSpeed configuration](#deepspeed-configuration)
- [Launching training](#launching-training)
- [Resume and fine-tune](#resume-and-fine-tune)
- [Data preparation](#data-preparation)
- [Troubleshooting](#troubleshooting)

## Quick start

### 1. Prepare the environment

Follow the installation instructions in [README.md](README.md). A normal
training environment needs:

- Python 3.10 and the packages in `requirements.txt`;
- a CUDA-enabled PyTorch installation;
- FlashAttention for practical training speed;
- the Wan2.2 and Qwen3-VL checkpoints referenced by the selected YAML file;
- enough GPUs for the selected model and per-process batch size.

The launch scripts for LAP training also check that `peft` is installed.

### 2. Create a local configuration

Start from the closest checked-in configuration instead of editing every
field from scratch. For a RoboTwin fine-tuning run:

```bash
cp configs/robotwin.yaml configs/robotwin_local.yaml
```

At minimum, update these paths in `configs/robotwin_local.yaml`:

```yaml
dataset:
  type: "robotwin"
  dataset_dir: "/path/to/robotwin/dataset"

model:
  wan:
    config_path: "./pretrained_models/Wan2.2-TI2V-5B"
    checkpoint_path: "./pretrained_models/Wan2.2-TI2V-5B"
    vae_path: "./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  vlm:
    checkpoint_path: "./pretrained_models/Qwen3-VL-2B-Instruct"

finetune:
  checkpoint_path: "./pretrained_models/Motus"
```

Set `finetune.checkpoint_path` to `null` when the run should initialize from
the WAN and VLM backbone paths instead of a Motus checkpoint.

### 3. Launch a single-node run

The following example starts eight local processes and uses DeepSpeed ZeRO
stage 1:

```bash
mkdir -p outputs

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train/train.py \
  --deepspeed configs/zero1.json \
  --config configs/robotwin_local.yaml \
  --run_name robotwin_local \
  --report_to tensorboard \
  2>&1 | tee outputs/robotwin_local.log
```

`training.batch_size` is the DataLoader batch size for each process. With
data parallelism, the effective global batch size is approximately:

```text
batch_size * number_of_processes * gradient_accumulation_steps
```

### 4. Locate logs and checkpoints

The shell log in the example above is written to
`outputs/robotwin_local.log`. Model checkpoints use a different directory,
derived by `train/train.py` from the YAML and CLI values:

```text
<system.checkpoint_dir>/<config-file-stem>/<run-name>/checkpoint_step_<N>
```

For the quick-start command and the default `system.checkpoint_dir`, this is:

```text
checkpoints/robotwin_local/robotwin_local/checkpoint_step_<N>
```

TensorBoard logs are stored below the same run directory in
`logging.tensorboard_log_dir`.

## Code architecture

The training-related repository layout is:

```text
Motus/
|-- configs/                 # Model, dataset, optimizer, and DeepSpeed configs
|-- data/                    # Dataset factory, collator, and dataset implementations
|   |-- dataset.py           # create_dataset(), MultiDataset, and collate_fn
|   |-- lerobot/             # LeRobot, AgiBot, RoboCOIN, and InternData loaders
|   |-- robotwin2/           # RoboTwin loader and conversion utilities
|   |-- bridge/              # Bridge dataset loader
|   |-- droid/               # DROID dataset loader
|   `-- fractal/             # Fractal/RT-1-style dataset loader
|-- models/                  # Motus and expert model definitions
|-- train/
|   `-- train.py             # Main training entry point and training loop
|-- utils/                   # Scheduler and shared training utilities
|-- scripts/                 # Local, HyperTrain, download, and helper scripts
|   `-- slurm/               # Single-node and multi-node Slurm launchers
|-- inference/               # Real-world and RoboTwin inference code
|-- checkpoints/             # Default model-state output root
`-- outputs/                 # Shell logs and generated smoke-test configs
```

The main runtime flow is:

1. `train/train.py` parses CLI arguments and loads the YAML with OmegaConf.
2. `load_config()` validates common settings and derives
   `action_chunk_size = num_video_frames * video_action_freq_ratio`.
3. `data/dataset.py` builds either one dataset or a weighted `MultiDataset`.
4. `models/motus.py` constructs the WAN video expert, VLM/understanding
   expert, and action expert from the model section.
5. `train/train.py` creates AdamW parameter groups and the configured
   scheduler.
6. Hugging Face Accelerate wraps the model, optimizer, DataLoader, and
   scheduler; a DeepSpeed plugin is enabled when `--deepspeed` is provided.
7. `UniDiffuserTrainer` runs validation and saves complete Accelerate state
   at the configured intervals.

## Training configuration

Training YAML files are stored in `configs/`. They are plain OmegaConf YAML
files; there is no automatic base-config inheritance. Copy a close example
and keep all required top-level sections.

### Top-level sections

| Section | Purpose |
| --- | --- |
| `common` | Shared action/state dimensions, video shape, and sampling rates. |
| `dataset` | Dataset type, paths, task selection, normalization, and multi-dataset composition. |
| `model` | WAN, VLM, action expert, understanding expert, loss, and time-distribution settings. |
| `training` | Batch size, step count, optimizer values, scheduler, gradient settings, and LAP mode. |
| `system` | Checkpoint root, save/evaluation intervals, DataLoader workers, and final-save behavior. |
| `logging` | TensorBoard/WandB backend, project name, log directory, and run name. |
| `resume` | Full-state checkpoint restoration and optional scheduler reset. |
| `finetune` | Partial Motus weight initialization for fine-tuning. |

Some specialized configs also define `training_mode`, `profiler`, or model
options such as `flow_source` and VLM export settings.

### Common dimensions and sampling

```yaml
common:
  action_dim: 14
  state_dim: 14
  num_video_frames: 8
  video_height: 384
  video_width: 320
  global_downsample_rate: 3
  video_action_freq_ratio: 2
```

Important relationships:

- `action_dim` and `state_dim` must match the representation emitted by the
  dataset, or the configured canonical representation.
- `num_video_frames` controls the predicted video window.
- `video_action_freq_ratio` determines how many action positions correspond
  to each video frame.
- The derived action chunk size in this example is `8 * 2 = 16`.
- `video_height` and `video_width` directly affect GPU memory usage.

### Single-dataset configuration

`configs/robotwin.yaml` is the basic single-dataset example:

```yaml
dataset:
  type: "robotwin"
  dataset_dir: "/path/to/robotwin/dataset"
  data_mode: "both"       # clean, randomized, or both
  task_mode: "multi"      # single or multi
  task_name: null          # required when task_mode is single
  max_episodes: null       # use a small value for a data-loading check
  image_aug: false
```

The dataset types currently routed by `data/dataset.py` include RoboTwin,
Bridge, DROID, Fractal, AC-One, Aloha-Agilex, several LeRobot variants,
latent-action data, EgoVerse trimodal data, and image QA data. Each type has
type-specific fields; use an existing config for that loader as the source
of truth.

### Multi-dataset and LAP v2 configuration

`configs/multidataset_lap_v2.yaml` combines multiple robot datasets in the
`canonical55_v2` representation:

```yaml
common:
  action_dim: 55
  state_dim: 55

dataset:
  type: "multi"
  sampling: "weighted"
  canonical_format: "canonical55_v2"
  target_action_dim: 55
  target_state_dim: 55
  datasets:
    - name: "agibot_lerobot_split"
      type: "lerobot_agibot"
      weight: 0.2187
      use_for_val: false
      task_mode: "multi"
      use_language_action: true
      params:
        root: "/path/to/AgiBotWorld2026"
        repo_id: "AgiBotWorld2026"
      common:
        global_downsample_rate: 3
        video_action_freq_ratio: 2

    - name: "robotwin"
      type: "robotwin"
      weight: 0.1000
      use_for_val: true
      dataset_dir: "/path/to/robotwin/dataset"
      data_mode: "both"
      task_mode: "multi"
      common:
        global_downsample_rate: 3
        video_action_freq_ratio: 2
```

For multi-dataset runs:

- `weight` is a positive relative sampling weight. It does not need to be
  manually converted into an integer sample count.
- At least one child must have `use_for_val: true`; otherwise validation
  dataset construction fails.
- A child `common` block overrides global sampling settings for that child.
- Loader-specific constructor values belong in `params` when shown by the
  corresponding checked-in config.
- `canonical55_v2` pads/remaps heterogeneous state and action vectors and
  carries validity masks for dataset-specific slots.
- Set `training.train_lap: true` for the LAP training path.



### Model configuration

The minimum backbone and expert structure is:

```yaml
model:
  wan:
    config_path: "./pretrained_models/Wan2.2-TI2V-5B"
    checkpoint_path: "./pretrained_models/Wan2.2-TI2V-5B"
    vae_path: "./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    precision: "bfloat16"

  vlm:
    checkpoint_path: "./pretrained_models/Qwen3-VL-2B-Instruct"
    precision: "bfloat16"
    frozen: true

  action_expert:
    hidden_size: 1024
    ffn_dim_multiplier: 4
    norm_eps: 1e-5

  und_expert:
    hidden_size: 512
    ffn_dim_multiplier: 4
    norm_eps: 1e-5
    vlm:
      input_dim: 2048
      projector_type: "mlp3x_silu"

  loss_weights:
    video_loss_weight: 1.0
    action_loss_weight: 1.0
```

### Optimization and scheduler

```yaml
training:
  batch_size: 4
  max_steps: 400000
  learning_rate: 5.0e-5
  # wan_learning_rate: 1.0e-5
  weight_decay: 0.01
  scheduler_type: "linear"
  warmup_steps: 200
  cycle_length: 5000000
  f_max: 0.99
  f_min: 0.4
  grad_clip_norm: 0.5
  gradient_accumulation_steps: 1
  find_unused_parameters: false
  train_lap: true
```

The optimizer is AdamW. `learning_rate` applies to all trainable parameters
unless `wan_learning_rate` is present, in which case WAN parameters receive
their own learning rate. `gradient_accumulation_steps` defaults to `1` when
omitted.

### System and logging

```yaml
system:
  checkpoint_dir: "./checkpoints"
  log_interval: 1
  save_interval: 5000
  val_interval: 500
  num_workers: 16
  pin_memory: true
  save_final_checkpoint: true

logging:
  report_to: "tensorboard"  # wandb, tensorboard, all, or none
  wandb_project: "motus"
  tensorboard_log_dir: "tensorboard_logs"
  run_name: null
```

The following CLI options override their YAML counterparts:

| CLI option | YAML value overridden |
| --- | --- |
| `--checkpoint_dir` | `system.checkpoint_dir` |
| `--report_to` | `logging.report_to` |
| `--wandb_project` | `logging.wandb_project` |
| `--run_name` | `logging.run_name` |

`--log_level` controls Python logging directly. If no run name is provided,
the entry point generates one from the dataset type, batch size, and learning
rate.

## DeepSpeed configuration

DeepSpeed is enabled by passing a JSON file to `--deepspeed`:

| File | ZeRO stage | Typical use |
| --- | ---: | --- |
| `configs/zero1.json` | 1 | Shard optimizer state with the smallest behavioral change. |
| `configs/zero2.json` | 2 | Also shard gradients to reduce per-GPU memory. |

All three checked-in files enable bf16 and let Accelerate/DeepSpeed infer the
micro batch size, global batch size, and gradient accumulation values. Keep
the YAML and launcher process counts consistent with the intended global
batch size.

## Launching training


### Generic local script

`scripts/train.sh` is the basic eight-GPU wrapper:

```bash
bash scripts/train.sh
```

Before using it, edit its conda initialization path, environment name,
`TASK`, and `CONFIG_FILE`. The checked-in script contains paths from the
original development environment and is not portable without those changes.

### LAP multi-dataset v2

The v2 launcher accepts environment-variable overrides. Always pass the
config path explicitly because the current default in the script is missing
the dot before `yaml`:

```bash
CONFIG_FILE=configs/multidataset_lap_v2.yaml \
DEEPSPEED_CONFIG=configs/zero2.json \
RUN_NAME=multidataset_lap_v2_8gpu \
bash scripts/train_lap_multidataset_v2.sh
```

Common overrides include `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`,
`MASTER_PORT`, `OUTPUT_DIR`, `REPORT_TO`, and the Hugging Face cache/offline
variables. The shell log defaults to:

```text
outputs/motus-multidataset_lap_v2/train_lap_8gpu.log
```

Run the built-in small-data smoke mode before a long v2 job:

```bash
CONFIG_FILE=configs/multidataset_lap_v2.yaml \
SMOKE_TEST=1 \
SMOKE_MAX_STEPS=5 \
SMOKE_MAX_EPISODES=2 \
bash scripts/train_lap_multidataset_v2.sh
```



### HyperTrain or Kubernetes multi-node launch

Use the same command on every pod. The platform must provide a consistent
`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, a unique `NODE_RANK`, and the number
of local GPU processes:

```bash
CONFIG_FILE=configs/multidataset_lap_v2.yaml \
NNODES=4 \
NPROC_PER_NODE=8 \
NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" \
MASTER_PORT="$MASTER_PORT" \
bash scripts/train_lap_multidataset_v2_hypertrain.sh
```

Do not hard-code the same `NODE_RANK` on every pod. Each node writes its own
log to `outputs/motus-<TASK>/train_lap_hypertrain_node<NODE_RANK>.log`.



## Resume and fine-tune

`resume` and `finetune` have different semantics and should not point to the
same checkpoint unless that is intentional.

### Resume an interrupted run

```yaml
resume:
  checkpoint_path: "./checkpoints/robotwin_local/robotwin_local/checkpoint_step_10000"
  reset_scheduler: false

finetune:
  checkpoint_path: null
```





```yaml
resume:
  checkpoint_path: null

finetune:
  checkpoint_path: "./pretrained_models/Motus"
```

Fine-tuning loads compatible Motus weights with
`model.load_pretrain_weights()` but creates a new optimizer, scheduler, and
step counter. It is partial initialization, not a continuation of the old
run.

When either resume or fine-tune is configured, the entry point disables
separate pretrained WAN/VLM backbone loading to avoid overwriting checkpoint
weights. For backbone-only initialization, set both checkpoint paths to
`null` and verify `model.wan.*` and `model.vlm.checkpoint_path`.

## Data preparation

- Repository data layout and processed dataset notes:
  [DATA_STRUCTURE.md](DATA_STRUCTURE.md)
- RoboTwin conversion:
  [data/robotwin2/robotwin_data_convert/README.md](data/robotwin2/robotwin_data_convert/README.md)
- Bridge dataset notes: [data/bridge/README.md](data/bridge/README.md)
- LeRobot/RoboCOIN notes:
  [data/lerobot/README_robocoin.md](data/lerobot/README_robocoin.md)

Before launching a full run, verify that every dataset path exists, required
language-action/T5 files are present, and normalization statistics match the
selected representation.
