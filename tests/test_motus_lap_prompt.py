from pathlib import Path


def test_robotwin_motus_lap_prompt_matches_training_english_template():
    source = (Path(__file__).resolve().parents[1] / "RoboTwin" / "policy" / "Motus" / "1.py").read_text()

    assert "Task：{instruction}\\nPlease provide a language description of the next action." in source
    assert "任务：{instruction}\\n请给出下一步动作语言描述。" not in source
