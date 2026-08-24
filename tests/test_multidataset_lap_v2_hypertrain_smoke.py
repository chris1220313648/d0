import os
from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "train_lap_multidataset_v2_hypertrain_smoke.sh"
TASK = "multidataset_lap_v2_smoke"


def _generate_config(tmp_path: Path, node_rank: int) -> tuple[dict, Path, str]:
    env = os.environ.copy()
    env.update(
        {
            "GENERATE_CONFIG_ONLY": "1",
            "NODE_RANK": str(node_rank),
            "TMPDIR": str(tmp_path),
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
    config_path = tmp_path / f"motus-{TASK}-node{node_rank}" / f"{TASK}.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path, result.stdout


def test_hypertrain_smoke_generates_requested_100_step_config(tmp_path):
    config, config_path, stdout = _generate_config(tmp_path, node_rank=0)

    datasets = {item["name"]: item for item in config["dataset"]["datasets"]}
    assert set(datasets) == {
        "agibot_lerobot_split",
        "bridge_pt_v5",
        "droid_pt_v5",
        "fractal_pt_v5",
        "robocoin_lerobot",
        "interndata_a1",
        "robotwin",
    }
    assert all(item["max_episodes"] == 2 for item in datasets.values())
    assert config["training"]["max_steps"] == 100
    assert config["training"]["batch_size"] == 1
    assert config["system"]["num_workers"] == 0
    assert config["system"]["val_interval"] == 101
    assert config["system"]["save_interval"] == 100
    assert config["system"]["save_final_checkpoint"] is False
    assert config["logging"]["report_to"] == "none"
    assert config["resume"]["checkpoint_path"] is None
    assert str(config_path) in stdout


def test_hypertrain_smoke_configs_match_across_nodes(tmp_path):
    _, node0_path, _ = _generate_config(tmp_path, node_rank=0)
    _, node1_path, _ = _generate_config(tmp_path, node_rank=1)

    assert node0_path.name == node1_path.name == f"{TASK}.yaml"
    assert node0_path.read_bytes() == node1_path.read_bytes()


def test_hypertrain_smoke_accepts_auto_nproc_per_node(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "GENERATE_CONFIG_ONLY": "1",
            "NPROC_PER_NODE": "auto",
            "TMPDIR": str(tmp_path),
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


def test_hypertrain_smoke_script_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
