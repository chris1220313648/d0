import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "train_lap_multidataset_v2_lazy.sh"


def _make_fake_project(tmp_path: Path, *, include_config: bool = True, include_base: bool = True) -> Path:
    project_root = tmp_path / "project"
    scripts_dir = project_root / "scripts"
    configs_dir = project_root / "configs"
    scripts_dir.mkdir(parents=True)
    configs_dir.mkdir()

    shutil.copy2(SCRIPT, scripts_dir / SCRIPT.name)
    if include_config:
        (configs_dir / "multidataset_lap_v2_lazy.yaml").write_text("dataset: {}\n", encoding="utf-8")
    if include_base:
        base = scripts_dir / "train_lap_multidataset_v2.sh"
        base.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$CONFIG_FILE\" \"$TASK\" \"$RUN_NAME\" \"$OUTPUT_DIR\"\n",
            encoding="utf-8",
        )
        base.chmod(0o755)
    return project_root


def test_lazy_launcher_passes_lazy_defaults_to_base_launcher(tmp_path):
    project_root = _make_fake_project(tmp_path)
    result = subprocess.run(
        ["bash", str(project_root / "scripts" / SCRIPT.name)],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"CONFIG_FILE", "TASK", "RUN_NAME", "OUTPUT_DIR"}
        },
    )

    assert result.stdout.splitlines() == [
        "configs/multidataset_lap_v2_lazy.yaml",
        "multidataset_lap_v2_lazy",
        "multidataset_lap_v2_lazy_lap_8gpu",
        "outputs/motus-multidataset_lap_v2_lazy",
    ]


def test_lazy_launcher_preserves_environment_overrides(tmp_path):
    project_root = _make_fake_project(tmp_path)
    override_config = project_root / "configs" / "override.yaml"
    override_config.write_text("dataset: {}\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "CONFIG_FILE": "configs/override.yaml",
            "TASK": "custom-task",
            "RUN_NAME": "custom-run",
            "OUTPUT_DIR": "outputs/custom",
        }
    )

    result = subprocess.run(
        ["bash", str(project_root / "scripts" / SCRIPT.name)],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.stdout.splitlines() == [
        "configs/override.yaml",
        "custom-task",
        "custom-run",
        "outputs/custom",
    ]


def test_lazy_launcher_reports_missing_config(tmp_path):
    project_root = _make_fake_project(tmp_path, include_config=False)
    result = subprocess.run(
        ["bash", str(project_root / "scripts" / SCRIPT.name)],
        cwd=project_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Lazy config file not found" in result.stderr


def test_lazy_launcher_reports_missing_base_launcher(tmp_path):
    project_root = _make_fake_project(tmp_path, include_base=False)
    result = subprocess.run(
        ["bash", str(project_root / "scripts" / SCRIPT.name)],
        cwd=project_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Base launcher not found" in result.stderr


def test_lazy_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
