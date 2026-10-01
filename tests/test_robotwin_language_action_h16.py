"""Check horizon endpoints, integer rounding, grippers, and tail alignment."""
import importlib.util
from pathlib import Path
import sys
import tempfile

import torch


def test_h16():
    source = Path(__file__).resolve().parents[1] / "data/robotwin2/robotwin_data_convert/robotwin_generate_language_action.py"
    spec = importlib.util.spec_from_file_location("robotwin_h16_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pose = torch.zeros(20, 14)
        pose[:, 0] = torch.arange(20) * 0.01
        pose[:, 1] = torch.arange(20) * 0.0001
        pose[:, 6] = 1
        pose[15:, 6] = 0
        pose[:, 13] = 1
        input_path = root / "input.pt"
        output_path = root / "output.txt"
        torch.save(pose, input_path)
        runner = module.RobotWinLanguageActionBackfill(
            root, ["clean"], window_size=16, input_dir_name="epos",
            input_mode="absolute_xyzrpy", sum_decimal="0f",
        )
        assert runner._process_episode(input_path, output_path) == (True, 20)
        lines = output_path.read_text().splitlines()
        assert len(lines) == 20
        assert lines[0] == "Left arm: move forward 15 cm, close gripper. Right arm: open gripper"
        assert lines[-2] == "Left arm: move forward 1 cm, close gripper. Right arm: open gripper"
        assert lines[-1] == "Left arm: close gripper. Right arm: open gripper"
        # The default still retains one decimal for existing callers.
        assert "move left 0.2 cm" in module._summarize_window(pose[:16].numpy(), "absolute_xyzrpy", "epos", "wxyz")


if __name__ == "__main__":
    test_h16()
    print("h16 checks passed")
