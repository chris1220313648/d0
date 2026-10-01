"""Reuse verified RoboDojo media mapping with original LeRobot joint actions."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch


def build():
    source=Path('/root/nas/zehua/RoboDojo/data/RoboDojo_lerobot_v21_video')
    verified=Path('data/robot_data/RoboDojo_eef_delta24').resolve()
    output=Path('data/robot_data/RoboDojo_joint24')
    if (output/'COMPLETE.json').exists(): raise FileExistsError(output)
    info=json.loads((source/'meta/info.json').read_text())
    manifest=json.loads((verified/'manifest.json').read_text())
    summary=json.loads((verified/'COMPLETE.json').read_text())
    lo={k:np.full(14,np.inf) for k in ('action','state')}
    hi={k:np.full(14,-np.inf) for k in lo}
    for n,row in enumerate(manifest):
        eid=row['episode_index'];local=row['local_episode']
        path=source/info['data_path'].format(episode_chunk=eid//info['chunks_size'],episode_index=eid)
        t=pq.read_table(path,columns=['action','observation.state','frame_index','episode_index'])
        assert np.array_equal(np.asarray(t['frame_index']),np.arange(row['frames']))
        assert (np.asarray(t['episode_index'])==eid).all()
        a=np.asarray(t['action'].to_pylist(),dtype=np.float32)
        q=np.asarray(t['observation.state'].to_pylist(),dtype=np.float32)
        assert a.shape==q.shape==(row['frames'],14)
        # Validate the original dataset's next-frame target convention.
        np.testing.assert_allclose(a[:-1],q[1:],atol=1e-5)
        dest=output/'clean'/row['task'];dest.mkdir(parents=True,exist_ok=True)
        old=verified/'clean'/row['task']
        for folder in ('videos','umt5_wan','metas','language_action'):
            link=dest/folder
            if not link.exists():link.symlink_to((old/folder).resolve(),target_is_directory=True)
        for folder,values in [('qpos',q),('action_joint',a)]:
            (dest/folder).mkdir(exist_ok=True)
            torch.save(torch.from_numpy(values),dest/folder/f'{local}.pt')
        for key,values in [('action',a),('state',q)]:
            assert np.isfinite(values).all()
            lo[key]=np.minimum(lo[key],values.min(0));hi[key]=np.maximum(hi[key],values.max(0))
        if (n+1)%500==0:print(n+1,flush=True)
    stats=dict(action_slots=list(range(7))+list(range(12,19)),transform='absolute_joint_targets_v1',
               **{k:dict(min=lo[k].tolist(),max=hi[k].tolist()) for k in lo})
    (output/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary.update(action_source=str(source),action_transform='absolute_joint_targets_v1')
    (output/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':build()
