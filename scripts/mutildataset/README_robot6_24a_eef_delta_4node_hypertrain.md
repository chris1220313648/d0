# Robot6 24D EEF HyperTrain

Launcher: `train_robot6_24a_eef_delta_4node_hypertrain.sh`  
Config: `../../configs/multidataset_lap_robot6_24a_norm.yaml`

统一设置：`action_dim=24`，`state_dim=14`，动作块长度 `16`。

## 数据集

| 数据集 | 有效维度（统一 24D） | 数据路径 |
|---|---|---|
| Bridge | `0–6`（7D） | `/root/nas/code/d0/data/robot_data/bridge_dataset` |
| DROID | `0–6`（7D） | `/root/nas/code/d0/data/robot_data/droid_dataset` |
| Fractal | `0–6`（7D） | `/root/nas/code/d0/data/robot_data/fractal` |
| Cosmos | `0–6`（7D） | `/root/nas/code/d0/data/robot_data/RoboCasa-Cosmos-Policy_lerobot_v21` |
| X-Embodiment | `0–6`（7D） | `/root/nas/code/d0/data/robot_data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim_v21` |
| GR1 | `0–23`（24D） | `/root/nas/code/d0/data/robot_data/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_v21_eef24` |
| RoboDojo | `0–6`、`12–18`（14D） | `/root/nas/code/d0/data/robot_data/RoboDojo_joint24` |

不足 24D 的数据会补零，并由 `action_mask` 屏蔽无效维度。

## 模型权重

```text
Wan:      ./pretrained_models/Wan2.2-TI2V-5B
Wan VAE:  ./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
Qwen3-VL: ./pretrained_models/Qwen3-VL-2B-Instruct
```
## 检查点

/root/nas/code/d0/checkpoints/multidataset_lap_robot6_24a_norm/multidataset_lap_robot6_24a_norm_lap_hypertrain/checkpoint_step_200000
