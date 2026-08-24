import os
from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "train_lap_robocoin_v2.sh"
CONFIG = REPO_ROOT / "configs" / "robocoin_lap_v2.yaml"


def test_robocoin_v2_config_uses_only_eager_canonical55_robocoin():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["common"]["action_dim"] == 55
    assert config["common"]["state_dim"] == 55
    assert config["dataset"] == {
        "type": "lerobot_robocoin",
        "canonical_format": "canonical55_v2",
        "task_mode": "multi",
        "task_name": None,
        "max_episodes": None,
        "image_aug": False,
        "use_language_action": True,
        "canonical_normalization": {
            "enabled": True,
            "stats_path": "data/utils/stat.json",
            "stats_key": "robocoin",
            "normalize_state": True,
            "normalize_action": True,
        },
        "params": {
            "root": "/root/nas/code/d0/data/robot_data/robocoin",
            "repo_id": "RoboCOIN",
            "embodiment_type": "aloha_agilex_2",
            "language_action_dir_name": "language_action",
            "enable_t5_fallback": False,
            "t5_folder_name": "t5_embedding",
            "task_discovery": {
                "enabled": True,
                "validate_episodes": True,
            },
        },
    }


def test_robocoin_v2_config_inherits_multidataset_v2_training_defaults():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "multidataset_lap_v2.yaml").read_text(encoding="utf-8")
    )

    for section in ("common", "model", "training", "system", "logging", "resume", "finetune"):
        assert config[section] == source[section]


def test_robocoin_smoke_generates_single_task_config(tmp_path):
    output_dir = tmp_path / "smoke-output"
    env = os.environ.copy()
    env.update(
        {
            "SMOKE_TEST": "1",
            "GENERATE_CONFIG_ONLY": "1",
            "OUTPUT_DIR": str(output_dir),
            "SMOKE_ROBOCOIN_TASK": "AI2_Alphabot_2_arrange_teaset",
            "SMOKE_MAX_EPISODES": "3",
            "SMOKE_MAX_STEPS": "7",
            "SMOKE_BATCH_SIZE": "2",
            "SMOKE_NUM_WORKERS": "1",
            "SMOKE_REPORT_TO": "none",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    generated = output_dir / "robocoin_lap_v2_smoke.yaml"
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["dataset"]["type"] == "lerobot_robocoin"
    assert config["dataset"]["task_mode"] == "multi"
    assert config["dataset"]["task_name"] == "AI2_Alphabot_2_arrange_teaset"
    assert config["dataset"]["max_episodes"] == 3
    assert config["dataset"]["params"]["task_discovery"]["enabled"] is False
    assert config["training"]["max_steps"] == 7
    assert config["training"]["batch_size"] == 2
    assert config["system"]["num_workers"] == 1
    assert config["system"]["val_interval"] == 8
    assert config["system"]["save_interval"] == 8
    assert config["system"]["save_final_checkpoint"] is False
    assert config["logging"]["report_to"] == "none"
    assert config["resume"]["checkpoint_path"] is None
    assert str(generated) in result.stdout


def test_robocoin_launcher_passes_overrides_to_torchrun(tmp_path):
    capture_path = tmp_path / "torchrun-args.txt"
    output_dir = tmp_path / "formal-output"
    fake_torchrun = tmp_path / "fake-torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "TORCHRUN_BIN": str(fake_torchrun),
            "CUDA_VISIBLE_DEVICES": "2,3",
            "NPROC_PER_NODE": "2",
            "MASTER_ADDR": "127.0.0.9",
            "MASTER_PORT": "29666",
            "TASK": "custom-robocoin",
            "REPORT_TO": "none",
            "OUTPUT_DIR": str(output_dir),
        }
    )

    subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert capture_path.read_text(encoding="utf-8").splitlines() == [
        "--nnodes=1",
        "--nproc_per_node=2",
        "--node_rank=0",
        "--master_addr=127.0.0.9",
        "--master_port=29666",
        "train/train.py",
        "--deepspeed",
        "configs/zero2.json",
        "--config",
        "configs/robocoin_lap_v2.yaml",
        "--run_name",
        "custom-robocoin_lap_2gpu",
        "--report_to",
        "none",
        "--log_level",
        "INFO",
    ]
    assert (output_dir / "train_lap.log").is_file()


def test_robocoin_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
