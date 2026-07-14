import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MOTUS_CONFIG_PATHS = [
    REPO_ROOT / "RoboTwin" / "policy" / "Motus" / "models" / "motus.py",
    REPO_ROOT / "models" / "motus.py",
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


def test_motus_config_accepts_flow_source_video_mode_with_gaussian_default():
    for path in MOTUS_CONFIG_PATHS:
        defaults = _field_defaults(_motus_config_class(path))

        assert defaults["flow_source_video_mode"] == "gaussian", path


def test_motus_config_validates_flow_source_video_mode():
    for path in MOTUS_CONFIG_PATHS:
        source = path.read_text()

        assert 'self.flow_source_video_mode not in {"gaussian", "history"}' in source, path
        assert "flow_source_video_mode must be 'gaussian' or 'history'" in source, path
