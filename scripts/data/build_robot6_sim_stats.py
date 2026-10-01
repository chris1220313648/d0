#!/usr/bin/env python3
"""Aggregate exact per-episode extrema, restricted to the active episode manifest."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.lerobot.sim_native14 import LAYOUTS, eef24_target_delta

def build(config='configs/multidataset_lap_robot6_14a_norm.yaml', output='data/utils/robot6_sim_arm7_stats.json'):
    result = dict(schema_version=1, method='source_minmax_0_1', datasets={})
    for ds in yaml.safe_load(Path(config).read_text())['dataset']['datasets']:
        if ds['type'] != 'lerobot_sim': continue
        kind = ds['params']['sim_kind']; layout=LAYOUTS[kind]
        if kind == 'gr1_eef24_delta' and ds.get('common', {}).get('global_downsample_rate', 1) != 1:
            raise ValueError('GR1 delta stats require global_downsample_rate=1')
        minimum={s:np.full(len(layout[s]),np.inf) for s in ('action','state')}
        maximum={s:-v for s,v in minimum.items()}
        tasks=[]; count=0; frames=0
        for info in sorted(Path(ds['dataset_dir']).glob('*/meta/info.json')):
            root=info.parent
            metadata=json.loads(info.read_text())
            manifest=(root/'episodes.jsonl').read_bytes()
            eps={e['episode_index']:e for e in map(json.loads,manifest.splitlines())}
            seen=set()
            for line in (root/'episodes_stats.jsonl').open():
                e=json.loads(line); i=e['episode_index']
                if i not in eps: continue
                if i in seen: raise ValueError(f'Duplicate stats: {root}/{i}')
                seen.add(i)
                for signal,key in [('action','action'),('state','observation.state')]:
                    st=e['stats'][key]; ids=layout[signal]
                    lo=np.asarray(st['min'])[ids]; hi=np.asarray(st['max'])[ids]
                    if kind == 'gr1_eef24_delta' and signal == 'action':
                        # Absolute extrema cannot yield extrema of differences.
                        # Read each active trajectory and use the loader's exact transform.
                        import pyarrow.parquet as pq
                        path=root.parent/metadata['data_path'].format(
                            episode_chunk=i//metadata['chunks_size'], episode_index=i)
                        table=pq.read_table(path, columns=['action'])
                        values=np.asarray(table['action'].to_pylist(), dtype=np.float32)
                        if len(values) != eps[i]['length'] or not len(values):
                            raise ValueError(f'Invalid trajectory length: {path}')
                        delta=eef24_target_delta(values, np.concatenate([values[:1], values[:-1]]))
                        lo=delta[:, ids].min(axis=0); hi=delta[:, ids].max(axis=0)
                    if not np.isfinite([lo,hi]).all() or np.any(hi<lo): raise ValueError(f'Invalid stats {root}/{i}')
                    minimum[signal]=np.minimum(minimum[signal],lo)
                    maximum[signal]=np.maximum(maximum[signal],hi)
            if seen != set(eps): raise ValueError(f'Missing stats: {root}: {set(eps)-seen}')
            count+=len(eps); frames+=sum(e['length'] for e in eps.values())
            tasks.append(dict(root=str(root.parent),episodes=len(eps),manifest_sha256=hashlib.sha256(manifest).hexdigest()))
        if not count: raise ValueError(f'Empty dataset {kind}')
        result['datasets'][kind]=dict(layout=layout,episodes=count,frames=frames,tasks=tasks,
            **{s:dict(min=minimum[s].tolist(),max=maximum[s].tolist()) for s in minimum})
        print(kind,count,frames,flush=True)
    target=Path(output)
    tmp=target.with_name(target.name+'.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    tmp.replace(target)
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/multidataset_lap_robot6_14a_norm.yaml')
    parser.add_argument('--output', default='data/utils/robot6_sim_arm7_stats.json')
    args = parser.parse_args()
    build(args.config, args.output)
