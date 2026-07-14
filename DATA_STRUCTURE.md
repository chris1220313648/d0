# Motus Data Structure Reference

This document summarizes the dataset directory layouts supported by
`data/dataset.py`. Use it as the checklist when preparing data under
`/root/nas/code/d0/data` or an external NAS path referenced by config.

## Common Sample Contract

Every loader should return samples compatible with `data/dataset.py::collate_fn`:

```python
{
    "first_frame": Tensor[C, H, W],
    "video_frames": Tensor[F, C, H, W],
    "action_sequence": Tensor[T, action_dim],
    "initial_state": Tensor[state_dim],          # optional for some loaders
    "language_embedding": Tensor[S, D],          # optional
    "vlm_inputs": dict | None,                   # optional
    "action_mask": Tensor[T, action_dim] | None, # optional
}
```

`MultiDataset` pads `action_sequence` and `initial_state` to
`dataset.target_action_dim` and `dataset.target_state_dim`.

## LeRobot Datasets

Dataset types:

- `lerobot`
- `lerobot_agibot`
- `lerobot_robocoin`

Single local repo layout:

```text
<root or dataset_dir>/<repo_id>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── chunk-001/
│   │   └── ...
│   └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── <video_feature_key>/
│   │   │   ├── episode_000000.mp4
│   │   │   ├── episode_000001.mp4
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── t5_embedding/
│   ├── episode_000000.pt
│   ├── episode_000001.pt
│   └── ...
└── language_action/
    ├── episode_000000.txt
    ├── episode_000001.txt
    └── ...
```

Important metadata fields:

```text
meta/episodes.jsonl:
  episode_index: int
  tasks: [str]
  length: int
  t5_embedding_path: "t5_embedding/episode_000000.pt"
  language_action_path: "language_action/episode_000000.txt"
```

Multi-task local root layout:

```text
<root>/
├── <task_or_repo_0>/
│   ├── meta/
│   ├── data/
│   └── videos/
├── <task_or_repo_1>/
│   ├── meta/
│   ├── data/
│   └── videos/
└── ...
```

Use `dataset.params.task_discovery` when tasks are nested or need filtering:

```yaml
params:
  root: "/path/to/local/root"
  repo_id: "local_root_name"
  task_discovery:
    enabled: true
    suffix: ""
    required:
      - "meta/info.json"
      - "data"
      - "videos"
```

### AgiBotWorld2026

Dataset type: `lerobot_agibot`.

Expected root:

```text
/root/nas/code/d0/data/robot_data/AgiBotWorld2026/
├── ImitationLearning/
│   └── .../
│       └── <task_split>/
│           ├── meta/
│           ├── data/
│           ├── videos/
│           ├── t5_embedding/
│           └── language_action/
└── RichInteraction/
    └── ...
```

The current config discovers split datasets by requiring `meta/info.json`,
`data`, and `videos`, and usually filters `suffix: "_split"`.

Loader behavior:

- Raw `action` is 40D, converted to compact 24D action.
- Raw `observation.state` can be wide, but loader uses 14D
  `state/joint/position`.
- Default camera keys:
  - `observation.images.top_head`
  - `observation.images.hand_left`
  - `observation.images.hand_right`

### RoboCOIN

Dataset type: `lerobot_robocoin`.

Expected root:

```text
/root/nasbak/cjy/robot_raw/robocoin/RoboCOIN/
├── <robot_task_0>/
│   ├── meta/
│   │   ├── info.json
│   │   ├── episodes.jsonl
│   │   ├── episodes_stats.jsonl
│   │   └── tasks.jsonl
│   ├── annotations/
│   │   ├── subtask_annotations.jsonl
│   │   ├── eef_direction_annotation.jsonl
│   │   ├── eef_velocity_annotation.jsonl
│   │   ├── eef_acc_mag_annotation.jsonl
│   │   ├── gripper_mode_annotation.jsonl
│   │   └── gripper_activity_annotation.jsonl
│   ├── data/
│   │   └── chunk-000/
│   │       ├── episode_000000.parquet
│   │       └── ...
│   ├── videos/
│   │   └── chunk-000/
│   │       ├── <camera_key>/
│   │       │   ├── episode_000000.mp4
│   │       │   └── ...
│   │       └── ...
│   ├── language_action/
│   │   ├── episode_000000.txt
│   │   └── ...
│   └── t5_embedding/
│       ├── episode_000000.pt
│       └── ...
├── <robot_task_1>/
└── ...
```

`prepare_robocoin_lerobot.py` generates `language_action/*.txt` from
`annotations/` and writes `language_action_path` into `meta/episodes.jsonl`.

Observed RoboCOIN dimensions:

- action: 14 to 54 dims, max 54
- state: 14 to 118 dims, max 118

The current loader pads heterogeneous raw action/state dimensions and returns
`action_mask` for valid action coordinates.

## RobotWin

Dataset type: `robotwin`.

Expected layout:

```text
<dataset_dir>/
├── clean/
│   ├── <task_name>/
│   │   ├── videos/
│   │   │   ├── episode0.mp4
│   │   │   └── ...
│   │   ├── qpos/
│   │   │   ├── episode0.pt
│   │   │   └── ...
│   │   ├── epos/
│   │   │   ├── episode0.pt
│   │   │   └── ...
│   │   ├── umt5_wan/
│   │   │   ├── episode0.pt
│   │   │   └── ...
│   │   └── language_action/
│   │       ├── episode0.txt
│   │       └── ...
│   └── ...
├── randomized/
│   ├── <task_name>/
│   │   ├── videos/
│   │   ├── qpos/
│   │   ├── epos/
│   │   ├── umt5_wan/
│   │   └── language_action/
│   └── ...
└── ...
```

`data_mode` can be `clean`, `randomized`, or `both`.

## Bridge / DROID / Fractal

Dataset types:

- `bridge`
- `droid` or `droid_bridge`
- `fractal_bridge`

These loaders use the same task-level layout:

```text
<dataset_dir>/
├── train/
│   ├── <task_name>/
│   │   ├── videos/
│   │   │   ├── <episode_id>.mp4
│   │   │   └── ...
│   │   ├── qpos/
│   │   │   ├── <episode_id>.pt
│   │   │   └── ...
│   │   ├── epos/
│   │   │   ├── <episode_id>.pt
│   │   │   └── ...
│   │   ├── umt5_wan/
│   │   │   ├── <episode_id>.pt
│   │   │   └── ...
│   │   ├── instructions/
│   │   │   ├── <task_or_episode>.txt
│   │   │   └── ...
│   │   └── language_action/
│   │       ├── <episode_id>.txt
│   │       └── ...
│   └── ...
├── test/
│   └── <task_name>/
│       ├── videos/
│       ├── qpos/
│       ├── epos/
│       ├── umt5_wan/
│       ├── instructions/
│       └── language_action/
└── ...
```

`data_mode` can be `train`, `test`, or `both`. `language_action/` is required
only when `use_language_action: true`.

## AC-One

Dataset type: `ac_one`.

Expected layout:

```text
<dataset_dir>/
├── <task_category>/
│   ├── <task_variant>.json
│   ├── <task_variant>/
│   │   ├── videos/
│   │   │   ├── 0.mp4
│   │   │   ├── 1.mp4
│   │   │   └── ...
│   │   ├── qpos/
│   │   │   ├── 0.pt
│   │   │   ├── 1.pt
│   │   │   └── ...
│   │   └── instructions/
│   │       ├── <task_variant>.txt
│   │       ├── <task_variant>.pt
│   │       └── ...
│   └── ...
└── ...
```

The loader recursively finds task folders that contain `videos`, `qpos`, and
`instructions`.

## Aloha-Agilex2

Dataset type: `aloha_agilex_2`.

Expected layout:

```text
<dataset_dir>/
├── <task_category>/
│   ├── <task_variant>.json
│   ├── <task_variant>/
│   │   ├── videos/
│   │   │   ├── 0.mp4
│   │   │   └── ...
│   │   ├── qpos/
│   │   │   ├── 0.pt
│   │   │   └── ...
│   │   └── instructions/
│   │       ├── <task_variant>.txt
│   │       ├── <task_variant>.pt
│   │       └── ...
│   └── ...
└── ...
```

This layout is similar to AC-One. The loader uses `data/utils/stat.json` for
normalization stats keyed by `aloha_agilex_2`.

## Latent Action

Dataset type: `latent_action`.

Expected leaf layout:

```text
<dataset_dir_or_nested_leaf>/
├── videos/
│   ├── <episode_id>.mp4
│   └── ...
├── umt5_wan/
│   ├── <episode_id>.pt
│   └── ...
└── latent_action_dim14/
    ├── <episode_id>.pt
    └── ...
```

`dataset.dataset_dir` is a list of roots. The loader recursively scans each
root until it finds leaf folders with `videos`, `umt5_wan`, and
`latent_action_dim14`.

It may write a scan cache:

```text
<root>/cached_episodes.v2.pkl
```

## EgoVerse Trimodal

Dataset type: `egoverse_trimodal`.

Expected config uses manifest files:

```yaml
dataset:
  type: "egoverse_trimodal"
  train_manifest: "/path/to/train.jsonl"
  val_manifest: "/path/to/val.jsonl"
  action_mode: "zeros"
```

Manifest records are consumed by
`human_data.motus_pretrain.loader.egoverse_vgm_dataset.EgoVerseTrimodalDataset`.
That loader is not under `data/` in this checkout, so keep the manifest schema
with the human-data package that provides it.

## Image QA

Dataset type: `image_qa`.

Expected layout:

```text
<image_root>/
├── image_000001.jpg
├── subdir/
│   ├── image_000002.png
│   └── ...
└── ...

<json_path>.json
```

JSON must be a list. Each record should provide an image path and QA text:

```json
[
  {
    "id": "sample_0",
    "image": "image_000001.jpg",
    "question": "What is shown?",
    "answer": "..."
  },
  {
    "id": "sample_1",
    "images": ["subdir/image_000002.png"],
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]
```

Supported image fields:

- `image`: string
- `image`: list of strings
- `images`: list of strings

The loader reduces multi-image records to one image per sample.

## Multi Dataset Config

Dataset type: `multi` wraps multiple child datasets:

```yaml
dataset:
  type: "multi"
  sampling: "weighted"
  target_action_dim: 24
  target_state_dim: 14
  datasets:
    - name: "agibot_lerobot_split"
      type: "lerobot_agibot"
      weight: 0.4
      task_mode: "multi"
      use_language_action: true
      params:
        root: "/root/nas/code/d0/data/robot_data/AgiBotWorld2026"
        repo_id: "AgiBotWorld2026"
    - name: "bridge"
      type: "bridge"
      weight: 0.1
      dataset_dir: "/root/nas/code/d0/data/robot_data/bridge_dataset"
```

The child loader returns its native action/state dimensions. The wrapper pads
them to `target_action_dim` and `target_state_dim`.

