"""RoboDojo target deltas with left/right 12D slots and explicit masks."""
import json
from pathlib import Path

import torch

from data.lerobot.sim_native14 import normalize_native
from data.robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset

ACTION_SLOTS = list(range(7)) + list(range(12, 19))


class RoboDojoDataset(RobotWinTaskDataset):
    def __init__(self, action_signal="action_delta", **kwargs):
        if action_signal not in ("action_delta", "action_joint"):
            raise ValueError(action_signal)
        self.action_signal = action_signal
        root = Path(kwargs['dataset_dir'])
        self.stats = json.loads((root / 'stats.json').read_text())
        if self.stats['action_slots'] != ACTION_SLOTS:
            raise ValueError('RoboDojo action statistics layout mismatch')
        if not (root / 'COMPLETE.json').exists():
            raise ValueError('RoboDojo preprocessing is incomplete')
        expected = 'absolute_joint_targets_v1' if action_signal == 'action_joint' else 'consecutive_target_world_to_robot_rotvec_v1'
        if self.stats['transform'] != expected:
            raise ValueError('RoboDojo action signal/statistics mismatch')
        super().__init__(**kwargs)
        if self.global_downsample_rate != 1 or self.include_history_actions:
            raise ValueError('RoboDojo target deltas require stride 1 and no history flow')

    def _scan_task_folder(self, task_path):
        return [e for e in super()._scan_task_folder(task_path)
                if (task_path / self.action_signal / (e['episode_name'] + '.pt')).is_file()
                and Path(e['lang_action_path']).is_file()]

    def _load_robot_data(self, qpos_path, action_indices, initial_state_idx=0,
                         history_action_indices=None):
        path = Path(qpos_path)
        states = torch.load(path, map_location='cpu', weights_only=True)
        deltas = torch.load(path.parent.parent / getattr(self, 'action_signal', 'action_delta') / path.name,
                            map_location='cpu', weights_only=True)
        if len(states) != len(deltas):
            raise ValueError('RoboDojo state/action length mismatch')
        # Raw action[t] targets state[t+1]; sampler indices denote target frames.
        actions = deltas[[max(i - 1, 0) for i in action_indices]].clone()
        previous = [initial_state_idx] + action_indices[:-1]
        repeated = torch.tensor([i == p for i, p in zip(action_indices, previous)])
        if getattr(self, 'action_signal', 'action_delta') == 'action_delta':
            actions[repeated, :6] = 0
            actions[repeated, 7:13] = 0
        actions = normalize_native(actions, self.stats['action'])
        output = torch.zeros(len(actions), 24)
        output[:, ACTION_SLOTS] = actions
        state = normalize_native(states[initial_state_idx].float(), self.stats['state'])
        return state, output, None

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        mask = torch.zeros_like(sample['action_sequence'], dtype=torch.bool)
        mask[:, ACTION_SLOTS] = True
        sample['action_mask'] = mask
        return sample
