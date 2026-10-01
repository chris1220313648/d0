"""Episode-lazy simulation loader with explicit native14 mapping and independent stats."""
import json
from pathlib import Path
import torch
from .lerobot_lazy_dataset import LazyLeRobotMixin
from .lerobot_robocoin_dataset import LeRobotRoboCOINDataset
from .sim_native14 import LAYOUTS, select_native, normalize_native, eef24_target_delta
from data.utils.image_utils import tensor_to_pil
from utils.vlm_utils import preprocess_vlm_messages_lap

class LeRobotSimDataset(LazyLeRobotMixin, LeRobotRoboCOINDataset):
    def __init__(self, *, sim_kind, stats_path, visual_keys, normalize_actions=True,
                 use_language_action=True, **kwargs):
        if not normalize_actions or not use_language_action:
            raise ValueError('Simulation native14 requires normalization and LAP')
        self.sim_kind = sim_kind
        self.visual_keys_override = tuple(visual_keys)
        self._visual_keys_cache = {}
        self.language_action_dir_name = 'language_action'
        self._language_action_cache = {}
        stats = json.loads(Path(stats_path).read_text())
        self.stats = stats['datasets'][sim_kind]
        expected_layout = dict(LAYOUTS[sim_kind], raw_dims=list(LAYOUTS[sim_kind]['raw_dims']))
        if self.stats['layout'] != expected_layout:
            raise ValueError('Statistics mapping differs from loader mapping')
        if kwargs.get('task_name') is None:
            root = Path(kwargs.get('root') or kwargs['dataset_dir'])
            kwargs['task_name'] = sorted(p.parent.parent.name for p in root.glob('*/meta/info.json'))
        self._init_lazy_common(**kwargs)
        if sim_kind == 'gr1_eef24_delta' and self.global_downsample_rate != 1:
            raise ValueError('GR1 delta statistics require consecutive frames: global_downsample_rate=1')
        if self.vlm_processor is None:
            raise RuntimeError('Simulation LAP requires a working VLM processor')

    def __getitem__(self, idx):
        episode = self._sample_lazy_episode()
        self._active_lazy_episode = episode
        columns = self._read_episode_columns(episode)  # fail with the actual path, never recursively resample
        if len(columns['action']) != episode.length:
            raise ValueError(f'Episode length mismatch: {episode.data_path}')
        t, video_indices, action_indices = self._calculate_sampling_indices(episode.length)
        media = self._lazy_media_for(episode)
        first, video = self._load_visual_frames(media, episode.episode_index,
            self._timestamps_at(columns, [t] + video_indices))
        state = select_native(torch.as_tensor(columns['observation.state'][t]).float(), self.sim_kind, 'state')
        raw_actions = self._values_at(columns, 'action', action_indices)
        if self.sim_kind == 'gr1_eef24_delta':
            # Anchor the first increment at the conditioning target, and later
            # increments at the previous action frame. Repeated padding holds.
            previous = self._values_at(columns, 'action', [t] + action_indices[:-1])
            raw_actions = eef24_target_delta(raw_actions, previous)
        actions = select_native(torch.as_tensor(raw_actions).float(), self.sim_kind, 'action')
        state = normalize_native(state, self.stats['state'])
        actions = normalize_native(actions, self.stats['action'])
        meta = episode.episode_meta
        instruction = meta['remarks'] if self.sim_kind in ('gr1', 'gr1_eef24', 'gr1_eef24_delta') else self._task_text(meta)
        if not str(instruction).strip():
            raise ValueError(f'Empty instruction: {episode.data_path}')
        item = self._row_at(columns, t, episode)
        embedding = self._load_language_embedding(item, episode.task_idx)
        lap_path = episode.dataset_root / meta['language_action_path']
        lap_lines = lap_path.read_text().splitlines()
        if len(lap_lines) != episode.length or not lap_lines[t].strip():
            raise ValueError(f'Invalid frame-aligned LAP: {lap_path}')
        lap = lap_lines[t].strip()
        tokens = preprocess_vlm_messages_lap(instruction, tensor_to_pil(first), self.vlm_processor,
                                            lap, supervise_answer=True)
        return dict(first_frame=first, video_frames=video, initial_state=state,
                    action_sequence=actions, language_embedding=embedding, vlm_inputs=tokens)
