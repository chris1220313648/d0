# HyperTrain LAP Multi-Dataset v0 训练说明

本文档说明下面这个训练入口脚本：

```bash
/root/nas/code/d0/scripts/train_lap_multidataset_v0_hypertrain.sh
```

该脚本用于在 HyperTrain/Kubernetes 环境中通过 `torchrun` 启动 LAP multi-dataset v0 预训练，主要使用：

```bash
configs/multidataset_lap_v0.yaml
configs/zero2.json
```

## 启动方式

在每个 HyperTrain pod 上使用同一个脚本作为启动命令：

```bash
bash /root/nas/code/d0/scripts/train_lap_multidataset_v0_hypertrain.sh
```

脚本会尝试从以下路径加载 conda：

```bash
/opt/conda/etc/profile.d/conda.sh
/share/anaconda3/etc/profile.d/conda.sh
```

然后激活：

```bash
conda activate motus
```

最终执行：

```bash
torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes="$NNODES" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  train/train.py \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --config "$CONFIG_TO_RUN" \
  --run_name "$RUN_NAME" \
  --report_to "$REPORT_TO" \
  --log_level "$LOG_LEVEL"
```

## 脚本参数

脚本通过环境变量读取参数。当前默认值如下：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TASK` | `multidataset_lap_v0` | 任务名，用于输出目录和 run name。 |
| `CONFIG_FILE` | `configs/multidataset_lap_v0.yaml` | 主训练配置文件。 |
| `DEEPSPEED_CONFIG` | `configs/zero2.json` | DeepSpeed 配置文件。 |
| `RUN_NAME` | `${TASK}_lap_hypertrain` | 传给训练脚本的 run name。 |
| `REPORT_TO` | `tensorboard` | CLI 传入的日志后端。YAML 中也配置了 `wandb`。 |
| `OUTPUT_DIR` | `outputs/motus-${TASK}` | shell 日志输出目录。 |
| `LOG_LEVEL` | `INFO` | 传给 `train/train.py` 的日志级别。 |
| `NNODES` | `1` | 分布式节点数量。4 节点训练时在 HyperTrain 环境变量里设为 `4`。 |
| `NPROC_PER_NODE` | `auto` | 每个节点启动的进程数。也可以显式设为每节点 GPU 数，例如 `8`。 |
| `MASTER_PORT` | `29500` | 分布式 rendezvous 端口，和官方 HyperTrain 示例保持一致。 |
| `NODE_RANK` | `0` | 节点 rank。多机时由 HyperTrain 注入，或在每个节点环境变量里分别设置。 |
| `MASTER_ADDR` | `localhost` | 分布式 rendezvous 地址。多机时由 HyperTrain 注入，不再由脚本拼接 `*-master-0`。 |



日志文件路径：

```bash
$OUTPUT_DIR/train_lap_hypertrain_node${NODE_RANK}.log
```


## 训练阶段

当前脚本和 `configs/multidataset_lap_v0.yaml` 对应 ours 的 Stage 2 预训练阶段：

| 方案 | 阶段 | 数据级别 | 训练目标 |
| --- | --- | --- | --- |
| ours | Stage 1.1: VGM Training | Level 2: Egocentric Human Videos；Level 3: Synthetic Data；Level 5: Multi-Robot Task Trajectory | Only VGM |
| ours | Stage 1.2: LLM Training | Level 1: VQA data | Only LLM |
| ours | Stage 2: Pretraining | Level 5: Multi-Robot Task Trajectory；Level 2: Egocentric Human Videos | Motus all 3 experts, with language action |
| ours | Stage 3: SFT | Level 6: Target-Robot Task Trajectory | Motus all 3 experts, with actions |

本脚本使用的 Stage 2 配置要点：

- 数据来自 Level 5 多机器人轨迹和 Level 2 第一视角人类视频。
- robot 数据集启用 `use_language_action: true`，人类视频使用 zero action。
- `training.train_lap: true`，训练 d0 三个专家：VGM/WAN video expert、LLM/VLM understanding expert、action expert。

## 注意力模式

当前模型采用 MoT unified attention / trimodal joint self-attention：

- VGM/WAN video tokens、action tokens、understanding tokens 在 WAN self-attention 路径中做三模态联合注意力。
- WAN cross-attention 使用 T5/language embedding 作为条件上下文。
- Stage 2 中 language action 进入 VLM 输入，用于监督/约束 LLM understanding 分支。
- 底层注意力实现优先使用 `flash_attention`；环境中如果有 FlashAttention 3 则优先使用 FlashAttention 3，否则回退到 FlashAttention 2；如果 FlashAttention 不可用，则回退到 PyTorch `scaled_dot_product_attention`。

## 预训练数据集构成

当前配置使用：

```yaml
dataset:
  type: "multi"
  sampling: "weighted"
```

即多数据集加权采样。所有数据集权重之和为 `1.0`。

| 数据集名 | 类型 | 权重 | 路径 / manifest | 说明 |
| --- | --- | ---: | --- | --- |
| `agibot_lerobot_split` | `lerobot_agibot` | `0.4425` | `/root/nas/code/d0/data/robot_data/AgiBotWorld2026` | 自动发现所有 `_split` 结尾的本地 LeRobot task；使用三路相机和 language action 文件。 |
| `human_video_egoverse` | `egoverse_trimodal` | `0.5000` | `/root/nas/usbdata/D0_huamn_dataset/EgoVerse/motus_pretrain/manifests` | 人类视频数据，动作使用 zero action，并启用自适应下采样。 |
| `droid_pt_v5` | `droid` | `0.0498` | `/root/nas/code/d0/data/robot_data/droid_dataset` | DROID 多任务机器人数据，使用 `stats_key: droid` 做归一化。 |
| `fractal_pt_v5` | `fractal_bridge` | `0.0055` | `/root/nas/code/d0/data/robot_data/fractal` | Fractal/RT-1 风格机器人数据，使用 `stats_key: fractal`。 |
| `bridge_pt_v5` | `bridge` | `0.0022` | `/root/nas/code/d0/data/robot_data/bridge_dataset` | Bridge 数据，使用 `stats_key: bridge`。 |

统一目标维度：

```yaml
common:
  action_dim: 24
  state_dim: 14
  num_video_frames: 8
  video_height: 384
  video_width: 320
```

各数据集采样频率配置：

| 数据集 | `global_downsample_rate` | `video_action_freq_ratio` |
| --- | ---: | ---: |
| AgiBot | `3` | `2` |
| EgoVerse | `8` | `2` |
| DROID | `1` | `2` |
| Fractal | `1` | `2` |
| Bridge | `1` | `2` |

### AgiBot 数据说明

AgiBot 使用自动发现：

```yaml
task_discovery:
  enabled: true
  suffix: "_split"
  required:
    - "meta/info.json"
    - "data"
    - "videos"
```



## 模型和预训练参数

文件路径
```bash
configs/multidataset_lap_v0.yaml
```

WAN video backbone：

```yaml
model:
  wan:
    config_path: "./pretrained_models/Wan2.2-TI2V-5B"
    checkpoint_path: "./pretrained_models/Wan2.2-TI2V-5B"
    vae_path: "./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    precision: "bfloat16"
```

VLM backbone：

```yaml
model:
  vlm:
    checkpoint_path: "/root/nas/xicheng/qwen3vl_2b_repro/outputs/qwen3vl2b_full_40pct_8gpu_nframes16/checkpoint-74056"
    precision: "bfloat16"
    frozen: false
```

这个 VLM checkpoint 已经做过兼容性修复：

- 从原始 `pretrained_models/Qwen3-VL-2B-Instruct/config.json` 补齐了 `text_config.rope_scaling` 和 `text_config.rope_theta`。
- 从原始 `pretrained_models/Qwen3-VL-2B-Instruct` 复制了 tokenizer / processor 相关文件到 checkpoint 目录。

Action expert：

```yaml
action_expert:
  hidden_size: 1024
  ffn_dim_multiplier: 4
  norm_eps: 1e-5
```

UND expert：

```yaml
und_expert:
  hidden_size: 512
  ffn_dim_multiplier: 4
  norm_eps: 1e-5
  vlm:
    input_dim: 2048
    projector_type: "mlp3x_silu"
```

Loss 权重：

```yaml
loss_weights:
  video_loss_weight: 1.0
  action_loss_weight: 1.0
```

EMA 当前关闭：

```yaml
ema:
  enabled: false
```

## 训练超参数

当前 `configs/multidataset_lap_v0.yaml` 中的训练参数：

```yaml
training:
  batch_size: 6
  max_steps: 200000
  learning_rate: 5.0e-5
  weight_decay: 0.01
  scheduler_type: "linear"
  warmup_steps: 200
  cycle_length: 5000000
  f_max: 0.99
  f_min: 0.4
  grad_clip_norm: 0.5
  use_amp: true
  find_unused_parameters: false
  train_lap: true
```

优化器是 `AdamW`。主学习率来自：

```yaml
training.learning_rate
```

如果需要给 WAN 单独设置学习率，可以增加：

```yaml
training:
  wan_learning_rate: 1.0e-5
```

如果不设置 `wan_learning_rate`，WAN 使用和其它参数相同的 `training.learning_rate`。

学习率会随 scheduler 变化。当前配置下大致范围是：

```text
最高 lr ~= learning_rate * f_max = 5.0e-5 * 0.99 = 4.95e-5
最低 lr ~= learning_rate * f_min = 5.0e-5 * 0.4  = 2.0e-5
```

## 系统和日志配置

```yaml
system:
  checkpoint_dir: "./checkpoints"
  log_level: "INFO"
  log_interval: 1
  save_interval: 50000
  val_interval: 5000
  num_workers: 16
  pin_memory: true
```

```yaml
logging:
  report_to: "wandb"
  wandb_project: "motus"
  tensorboard_log_dir: "tensorboard_logs"
  run_name: null
```

注意：shell 脚本也会通过 CLI 传入：

```bash
--report_to "$REPORT_TO"
```

当前 shell 默认 `REPORT_TO=tensorboard`，而 YAML 中是 `wandb`。如果需要确认最终使用哪个日志后端，请看训练启动日志。

## DeepSpeed 配置

当前使用 `configs/zero2.json`，开启 bf16 和 ZeRO stage 2：

```json
{
  "bf16": {
    "enabled": true
  },
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "sub_group_size": 1000000000.0
  }
}
```

## Resume 和 Finetune

当前配置：

```yaml
resume:
  checkpoint_path: null
  reset_scheduler: false

finetune:
  checkpoint_path: null
```

行为说明：

- 如果设置 `resume.checkpoint_path`，会通过 `accelerator.load_state(...)` 恢复 model、optimizer、scheduler、dataloader 和 RNG 状态。
- 如果设置 `resume.reset_scheduler: true`，scheduler 会按当前 YAML 重新初始化相关参数。
- 如果设置了 `resume.checkpoint_path` 或 `finetune.checkpoint_path`，`train/train.py` 会在创建模型前设置 `model.load_pretrained_backbones = false`，避免重新加载 WAN/VLM 预训练权重覆盖 checkpoint 权重。
