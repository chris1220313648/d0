#!/usr/bin/env python3
"""Build small target-delta sidecars; link existing videos, states, T5 and LAP."""
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.robotwin2.robodojo_dataset import ACTION_SLOTS

EEF = Path('/root/nas/code/d0-zy/dojo_sample/data/RoboDojo_ee_lerobot_v30_video')
PT = Path('/root/nas/zehua/d0_new_robodojo/data/robodojo/dataset_raw_gripper')
V21 = Path('/root/nas/zehua/RoboDojo/data/RoboDojo_lerobot_v21_video')
LAP = Path('/root/nas/code/d0-zy/Motus/data/robodojo_lap_canonical/language_action')
WORLD_TO_ROBOT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)


def target_deltas(actions, initial_state):
    """Target[t]-target[t-1]; first target references initial measured pose."""
    previous = np.concatenate([initial_state[:1], actions[:-1]])
    result = np.zeros((len(actions), 14), dtype=np.float32)
    for raw, dst in ((0, 0), (8, 7)):
        result[:, dst:dst+3] = (actions[:, raw:raw+3] - previous[:, raw:raw+3]) @ WORLD_TO_ROBOT.T
        order = [raw+4, raw+5, raw+6, raw+3]  # source wxyz -> scipy xyzw
        delta = Rotation.from_quat(actions[:, order]) * Rotation.from_quat(previous[:, order]).inv()
        result[:, dst+3:dst+6] = delta.as_rotvec() @ WORLD_TO_ROBOT.T
        result[:, dst+6] = actions[:, raw+7]
    if not np.isfinite(result).all(): raise ValueError('Nonfinite RoboDojo action')
    return result


def build(output):
    output = Path(output)
    if (output / 'COMPLETE.json').exists():
        raise FileExistsError(f'Already complete: {output}')
    episodes = {r['episode_index']:r for r in map(json.loads, (V21/'meta/episodes.jsonl').read_text().splitlines())}
    tasks = {}
    task_counts = {}
    mapping = {}
    for eid, row in sorted(episodes.items()):
        text = row['tasks'][0]
        if text not in tasks:
            matches = [p.parent.parent for p in (PT/'clean').glob('*/metas/0.txt') if p.read_text().strip().endswith(text)]
            if len(matches) != 1: raise ValueError(f'Ambiguous task: {text}: {matches}')
            tasks[text] = matches[0]
        task = tasks[text]
        local = task_counts.get(text, 0); task_counts[text] = local+1
        mapping[eid] = (task, str(local))
    minimum = {k:np.full(14, np.inf) for k in ('action','state')}
    maximum = {k:np.full(14, -np.inf) for k in minimum}
    manifest = []; seen=set(); excluded=[]
    for path in sorted((EEF/'data').rglob('*.parquet')):
        table = pq.read_table(path, columns=['episode_index','frame_index','action','observation.state'])
        ids=np.asarray(table['episode_index']); frames=np.asarray(table['frame_index'])
        action=np.asarray(table['action'].to_pylist(), dtype=np.float32)
        state=np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        for eid in np.unique(ids):
            eid=int(eid)
            if eid in seen: raise ValueError(f'Episode spans files: {eid}')
            seen.add(eid); take=ids==eid; a,s=action[take],state[take]
            assert np.array_equal(frames[take], np.arange(len(a)))
            assert len(a)==episodes[eid]['length']
            task,local=mapping[eid]
            required=[task/'qpos'/f'{local}.pt',task/'epos'/f'{local}.pt',task/'videos'/f'{local}.mp4',task/'umt5_wan'/f'{local}.pt',task/'metas'/f'{local}.txt']
            missing=[str(p) for p in required if not p.is_file()]
            if missing:
                excluded.append(dict(episode_index=eid,task=task.name,local_episode=local,missing=missing))
                continue
            q=torch.load(task/'qpos'/f'{local}.pt',weights_only=True).numpy()
            e=torch.load(task/'epos'/f'{local}.pt',weights_only=True).numpy()
            assert q.shape==e.shape==(len(a),14), (eid,task,local)
            # Verify the complete trajectory, not only a matching task name/length.
            for src,dst in ((0,0),(8,7)):
                np.testing.assert_allclose(e[:,dst:dst+3], s[:,src:src+3], atol=1e-5)
                r=Rotation.from_quat(s[:,[src+4,src+5,src+6,src+3]])
                error=(r.inv()*Rotation.from_euler('xyz',e[:,dst+3:dst+6])).magnitude()
                assert error.max()<1e-4, (eid,error.max())
                np.testing.assert_allclose(q[:,dst+6],s[:,src+7],atol=1e-5)
            delta=target_deltas(a,s)
            for key,values in [('action',delta),('state',q)]:
                assert np.isfinite(values).all()
                minimum[key]=np.minimum(minimum[key],values.min(0))
                maximum[key]=np.maximum(maximum[key],values.max(0))
            dest=output/'clean'/task.name
            dest.mkdir(parents=True,exist_ok=True)
            for name in ('videos','qpos','umt5_wan','metas'):
                link=dest/name
                if not link.exists(): link.symlink_to((task/name).resolve(),target_is_directory=True)
                ext='.mp4' if name=='videos' else '.txt' if name=='metas' else '.pt'
                assert (link/f'{local}{ext}').is_file()
            lap=LAP/f'episode_{eid:06d}.txt'
            assert len(lap.read_text().splitlines())==len(a)
            (dest/'language_action').mkdir(exist_ok=True)
            link=dest/'language_action'/f'{local}.txt'
            if not link.exists():link.symlink_to(lap)
            (dest/'action_delta').mkdir(exist_ok=True)
            torch.save(torch.from_numpy(delta),dest/'action_delta'/f'{local}.pt')
            manifest.append(dict(episode_index=eid,task=task.name,local_episode=local,frames=len(a),source=str(path)))
        print(f'{path.name}: {len(seen)} episodes',flush=True)
    assert seen==set(episodes)
    # Repeated tail frames hold pose (physical delta zero), hands remain commanded.
    for indexes in (list(range(6)),list(range(7,13))):
        minimum['action'][indexes]=np.minimum(minimum['action'][indexes],0)
        maximum['action'][indexes]=np.maximum(maximum['action'][indexes],0)
    stats=dict(action_slots=ACTION_SLOTS,transform='consecutive_target_world_to_robot_rotvec_v1',
               **{k:dict(min=minimum[k].tolist(),max=maximum[k].tolist()) for k in minimum})
    (output/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=dict(episodes=len(manifest),excluded=excluded,tasks=len(tasks),frames=sum(x['frames'] for x in manifest),
                 eef_root=str(EEF),pt_root=str(PT),lap_root=str(LAP),action_slots=ACTION_SLOTS)
    (output/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/robot_data/RoboDojo_eef_delta24')
    build(parser.parse_args().output)
