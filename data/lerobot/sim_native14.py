"""Explicit native-control subsets. Equal width does not imply equal semantics."""
import numpy as np
import torch
from data.utils.norm import normalize_masked

# Cosmos: EEF pose + gripper; X-Embodiment: skip base/mode prefix.
# GR1 retains both complete 7-joint arms. Raw files are not rewritten.
LAYOUTS = {
    'cosmos': {'action': list(range(7)), 'state': list(range(9)), 'raw_dims': (12, 9)},
    'xemb': {'action': list(range(5, 12)), 'state': list(range(25, 32)) + [21, 22], 'raw_dims': (12, 53)},
    'gr1': {'action': list(range(7)) + list(range(22, 29)), 'state': list(range(7)) + list(range(22, 29)), 'raw_dims': (44, 44)},
    'gr1_eef24': {'action': list(range(24)), 'state': list(range(7)) + list(range(22, 29)), 'raw_dims': (24, 44)},
    'gr1_eef24_delta': {'action': list(range(6, 12)) + list(range(18, 24)) + list(range(6)) + list(range(12, 18)), 'state': list(range(7)) + list(range(22, 29)),
                      'raw_dims': (24, 44), 'action_transform': 'consecutive_target_base_rotvec_v1'},
}


def eef24_target_delta(current, previous):
    """Base-frame target-to-target pose increments; hand controls stay absolute.

    R_current = R_delta @ R_previous. This is not Euler/rotvec subtraction
    and not a target-minus-measured-pose control error. Identical frame pairs
    produce zero pose increments, including repeated end-of-episode padding.
    """
    from scipy.spatial.transform import Rotation
    current, previous = np.asarray(current), np.asarray(previous)
    if current.ndim != 2 or current.shape != previous.shape or current.shape[1] != 24:
        raise ValueError('EEF24 delta requires matching [T,24] current/previous actions')
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError('Nonfinite EEF24 target pose')
    out = current.astype(np.float32, copy=True)
    for start in (0, 6):
        out[:, start:start+3] = current[:, start:start+3] - previous[:, start:start+3]
        rot = slice(start+3, start+6)
        out[:, rot] = (Rotation.from_rotvec(current[:, rot]) *
                       Rotation.from_rotvec(previous[:, rot]).inv()).as_rotvec()
        identical = (current[:, start:start+6] == previous[:, start:start+6]).all(axis=1)
        out[identical, start:start+6] = 0
    return out

def select_native(values, kind, signal):
    layout = LAYOUTS[kind]
    expected = layout['raw_dims'][signal == 'state']
    if values.shape[-1] != expected:
        raise ValueError(f'{kind}/{signal}: expected {expected}, got {values.shape}')
    return values[..., layout[signal]]


def normalize_native(values, stats):
    return normalize_masked(values, torch.ones_like(values, dtype=torch.bool),
                            np.asarray(stats['min']), np.asarray(stats['max']),
                            np.ones(values.shape[-1], dtype=bool))
