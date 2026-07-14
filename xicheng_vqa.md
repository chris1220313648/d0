# VLA + VLM-QA 混合训练改动总结

- 原始 VLA 数据仍然走原来的 Motus 训练路径。
- 新增 Image-QA 数据走 Qwen3-VL 的直接 SFT 路径。
- VLA batch 和 QA batch 不混在同一个 batch 内，而是两个 dataloader 交替采样。
- QA batch 目前不进入 Video Gen Model、Action Expert、UndExpert、MoT joint attention、video decoder 或 action decoder。
- QA batch 只对 Qwen3-VL 的 answer token 计算 CE loss。

## 操作位置

```text
/root/nas/xicheng/d0_imageqa_adapt/d0
```

## 代码文件改动


| 文件                                             | 改动内容                                                                             |
| ---------------------------------------------- | -------------------------------------------------------------------------------- |
| `utils/vlm_utils.py`                           | 新增 `preprocess_vlm_messages_sft(...)`，用于 `image + question -> answer` 的 SFT 预处理。 |
| `utils/image_qa_vlm_utils.py`                  | 新增 QA SFT 预处理 wrapper。                                                           |
| `data/image_qa/image_qa_dataset.py`            | 新增 Image-QA dataset，读取 LLaVA/Qwen 格式 QA JSON。                                    |
| `data/dataset.py`                              | 增加 `image_qa` 和 `vla_vlmqa_mixed` dataset 构造逻辑。                                  |
| `train/train.py`                               | 增加 mixed dataloader 调度；增加 `qa_only=True` 分支。                                     |
| `train/train_vla_vlmqa_mixed.py`               | 新增混合训练入口 wrapper。                                                                |
| `models/motus.py`                              | 新增 `compute_vlm_sft_loss` / `vlm_sft_training_step`；兼容 Qwen3-VL 新 API。           |
| `configs/multidataset_lap_v0_vlmqa_mixed.yaml` | 新增 VLA+VLM-QA mixed 配置。                                                          |
| `configs/vla_vlmqa_mixed_qaonly_smoke.yaml`    | 新增 QA-only smoke 配置。                                                             |
| `scripts/train_lap_vlmqa_mixed_hypertrain.sh`  | Train 启动脚本。                                                                      |
| `tools/smoke_vla_vlmqa_mixed_dataset.py`       | 新增 dataloader smoke 检查脚本。                                                        |


具体训练逻辑如下。

VLA batch 仍然走原始 Motus training step：

```text
VLA batch
  -> model.training_step(...)
  -> video_loss + action_loss + optional LAP llm_loss
```

也就是说，原来的机器人 VLA 数据训练方式保持不变。

QA batch 会带上：

```python
qa_only = True
```

然后走 Qwen3-VL 的 VLM SFT 路径：

```text
QA batch
  -> model.vlm_sft_training_step(...)
  -> Qwen3-VL forward
  -> answer CE loss
```

QA batch 只监督 answer token，不监督 prompt token。

当前 QA batch 的主要作用是更新 Qwen3-VL 的 VLM 问答能力。

当前 QA batch 不进入 MoT 主干，也不经过 action/video 分支。

目前 QA batch 不会更新：

- Video Gen Model
- Action Expert
- UndExpert / MoT joint attention
- video decoder
- action decoder
更新 Qwen3-VL 本体。

## 关于 dummy video/action/mask 和主干更新的关键说明

这里的关键不只是有没有构造 `dummy video/action`，而是 **QA answer loss 的计算路径有没有经过 tri-modal joint attention**。

如果 QA batch 仍然是：

```text
image + question -> Qwen3-VL -> LM head -> QA answer CE loss
```

即使额外构造了 `dummy video`、`dummy action` 和 mask，并且设置：

```python
video_loss = 0
action_loss = 0
```

只要 QA answer token 没有参与 tri-modal joint attention，或者 QA loss 不是基于经过 tri-modal joint attention 更新后的 token 计算的，那么梯度就不会传到 Motus 主干。

这种情况下实际更新的主要还是：

```text
Qwen3-VL 以及和 Qwen LM loss 直接相连的部分
```

不会有效更新：

```text
Video Gen Model / Action Expert / Tri-modal Joint Attention / MoT 主干
```

如果希望 QA 数据真正进入 Motus 主干，需要让流程变成：

```text
QA answer tokens / und tokens
    -> 参与 tri-modal joint attention
    -> 得到 updated und tokens
    -> 基于 updated und tokens 计算 QA CE loss
```

只有这样，QA loss 才可能反传到 tri-modal joint attention，甚至间接影响 video/action token 相关分支。否则，仅仅构造 dummy video/action 再 mask 掉 video/action loss，本质上仍然是普通 Qwen3-VL SFT，不会带来 MoT 主干更新。
