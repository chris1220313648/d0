import os
import json
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.dataset import create_dataset, collate_fn
from data.lerobot.sim_native14 import LAYOUTS
config=OmegaConf.load(os.environ.get('CONFIG', 'configs/multidataset_lap_robot6_14a_norm.yaml'))
action_dim, state_dim = config.common.action_dim, config.common.state_dim
rows=[]; samples=[]
for entry in config.dataset.datasets:
 if os.environ.get("SIM_ONLY") and entry.type != "lerobot_sim": continue
 c=OmegaConf.create(OmegaConf.to_container(config,resolve=True))
 c.dataset.datasets=[entry]
 # Inspect real samples without indexing the full traditional robot corpus here.
 c.dataset.datasets[0].max_episodes=2
 ds=create_dataset(c)
 s=ds[0]; samples.append(s)
 assert s['action_sequence'].shape==(16,action_dim)
 assert s['initial_state'].shape==(state_dim,)
 assert torch.isfinite(s['action_sequence']).all() and torch.isfinite(s['initial_state']).all()
 expected_valid = len(LAYOUTS[entry.params.sim_kind]['action']) if entry.type == 'lerobot_sim' else (14 if entry.type == 'robodojo' else 7)
 assert (s['action_mask'].sum(dim=-1) == expected_valid).all()
 assert (s['action_sequence'][~s['action_mask']] == 0).all()
 if entry.type == 'robodojo':
  assert s['action_mask'][0].nonzero().flatten().tolist() == list(range(7)) + list(range(12,19))
 row=dict(name=entry.name,action_shape=list(s['action_sequence'].shape),state_shape=list(s['initial_state'].shape),
          action_range=[s['action_sequence'].min().item(),s['action_sequence'].max().item()],
          state_range=[s['initial_state'].min().item(),s['initial_state'].max().item()],
          action_valid=int(s['action_mask'][0].sum()),state_valid=int(s['state_mask'].sum()),
          video_shape=list(s['video_frames'].shape),vlm_labels=int((s['vlm_inputs']['labels']!=-100).sum()))
 rows.append(row);print(json.dumps(row),flush=True)
 del ds
for subset in (samples[:5],samples[-5:]) if len(samples)>=5 else ():
 b=collate_fn(subset)
 assert b['action_sequence'].shape==(5,16,action_dim)
 assert b['initial_state'].shape==(5,state_dim)
output_dir = Path(os.environ.get('OUTPUT_DIR', 'outputs/robot6_arm7_preflight'))
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / ('sim_preflight.json' if os.environ.get('SIM_ONLY') else 'batch_preflight.json')).write_text(json.dumps(rows,indent=2))
print('SIM SOURCES PASS' if os.environ.get('SIM_ONLY') else f'{len(samples)} SOURCES AND MIXED BATCHES PASS',flush=True)
