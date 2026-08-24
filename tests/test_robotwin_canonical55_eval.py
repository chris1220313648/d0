import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MOTUS_ROOT = REPO_ROOT / "RoboTwin" / "policy" / "Motus"
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))


def test_robotwin_qpos_to_canonical55_places_grippers_in_canonical_slots():
    from utils.robotwin_canonical55 import robotwin_qpos_to_canonical55

    qpos = np.arange(14, dtype=np.float32)
    canonical = robotwin_qpos_to_canonical55(qpos)

    assert canonical.shape == (55,)
    np.testing.assert_array_equal(canonical[:6], qpos[:6])
    assert canonical[6] == 0
    np.testing.assert_array_equal(canonical[7:13], qpos[7:13])
    assert canonical[13] == 0
    np.testing.assert_array_equal(canonical[26:28], qpos[[6, 13]])
    assert np.count_nonzero(canonical[14:26]) == 0
    assert np.count_nonzero(canonical[28:]) == 0


def test_canonical55_to_robotwin_qpos_extracts_only_robotwin_slots():
    from utils.robotwin_canonical55 import canonical55_to_robotwin_qpos

    canonical = np.full(55, -99.0, dtype=np.float32)
    canonical[0:6] = np.arange(0, 6, dtype=np.float32)
    canonical[7:13] = np.arange(7, 13, dtype=np.float32)
    canonical[26] = 6
    canonical[27] = 13

    qpos = canonical55_to_robotwin_qpos(canonical)

    assert qpos.shape == (14,)
    np.testing.assert_array_equal(qpos, np.arange(14, dtype=np.float32))


def test_robotwin_canonical55_mapping_preserves_batch_prefixes_and_rejects_bad_dims():
    from utils.robotwin_canonical55 import canonical55_to_robotwin_qpos, robotwin_qpos_to_canonical55

    qpos = np.arange(2 * 16 * 14, dtype=np.float32).reshape(2, 16, 14)
    canonical = robotwin_qpos_to_canonical55(qpos)
    roundtrip = canonical55_to_robotwin_qpos(canonical)

    assert canonical.shape == (2, 16, 55)
    assert roundtrip.shape == (2, 16, 14)
    np.testing.assert_array_equal(roundtrip, qpos)

    with pytest.raises(ValueError, match="last dimension 14"):
        robotwin_qpos_to_canonical55(np.zeros(13, dtype=np.float32))
    with pytest.raises(ValueError, match="last dimension 55"):
        canonical55_to_robotwin_qpos(np.zeros(54, dtype=np.float32))


def test_canonical55_eval_configs_point_to_55d_checkpoint_and_model_config():
    robotwin_cfg = yaml.safe_load((MOTUS_ROOT / "utils" / "robotwin_canonical55.yml").read_text())
    paths_cfg = yaml.safe_load((MOTUS_ROOT / "paths_config_canonical55.yml").read_text())

    assert robotwin_cfg["common"]["action_dim"] == 55
    assert robotwin_cfg["common"]["state_dim"] == 55
    assert paths_cfg["model_config"] == "robotwin_canonical55.yml"

    checkpoint_dir = Path(paths_cfg["checkpoint_path"])
    assert checkpoint_dir.name == "pytorch_model"
    assert (checkpoint_dir / "mp_rank_00_model_states.pt").is_file()
