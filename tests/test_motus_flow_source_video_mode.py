import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_MOTUS_CONFIG_PATH = REPO_ROOT / "models" / "motus.py"
MOTUS_CONFIG_PATHS = [
    REPO_ROOT / "RoboTwin" / "policy" / "Motus" / "models" / "motus.py",
    ROOT_MOTUS_CONFIG_PATH,
]


def _motus_config_class(path: Path) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MotusConfig":
            return node
    raise AssertionError(f"MotusConfig not found in {path}")


def _field_defaults(config_class: ast.ClassDef) -> dict[str, object]:
    defaults = {}
    for node in config_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                defaults[node.target.id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                defaults[node.target.id] = None
    return defaults


def test_root_motus_config_inherits_action_flow_source_when_video_mode_is_omitted():
    defaults = _field_defaults(_motus_config_class(ROOT_MOTUS_CONFIG_PATH))
    source = ROOT_MOTUS_CONFIG_PATH.read_text()

    assert defaults["flow_source_video_mode"] is None
    assert "if self.flow_source_video_mode is None:" in source
    assert "self.flow_source_video_mode = self.flow_source_mode" in source


def test_motus_config_validates_flow_source_video_mode():
    for path in MOTUS_CONFIG_PATHS:
        source = path.read_text()

        assert 'self.flow_source_video_mode not in {"gaussian", "history"}' in source, path
        assert "flow_source_video_mode must be 'gaussian' or 'history'" in source, path


def test_vlm_loss_weight_is_configured_and_applied_to_llm_loss():
    defaults = _field_defaults(_motus_config_class(ROOT_MOTUS_CONFIG_PATH))
    motus_source = ROOT_MOTUS_CONFIG_PATH.read_text()
    train_source = (REPO_ROOT / "train" / "train.py").read_text()

    assert defaults["vlm_loss_weight"] == 1.0
    assert "total_loss += self.config.vlm_loss_weight * llm_loss" in motus_source
    assert "vlm_loss_weight=config.model.loss_weights.get('vlm_loss_weight', 1.0)" in train_source
