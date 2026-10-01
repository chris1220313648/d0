"""Require completed eight-rank optimizer steps and report observed evidence."""
import argparse
from collections import Counter
import json
import math
import re
import yaml
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
args=parser.parse_args(); run=args.run
assert (run/'exit_code').read_text().strip()=='0', 'Training did not exit successfully'
ranks=[];counts=Counter()
for rank in range(8):
 rows=[json.loads(x) for x in (run/f'rank{rank}_batches.jsonl').read_text().splitlines()]
 assert [x['step'] for x in rows]==list(range(100)), (rank,len(rows))
 assert all(x['batch_size']==5 and math.isfinite(x['loss']) for x in rows)
 assert all(math.isfinite(v) for x in rows for v in x['metrics'].values() if isinstance(v,(float,int)))
 local=Counter(n for x in rows for n in x['datasets']);counts.update(local)
 ranks.append(dict(rank=rank,steps=len(rows),samples=sum(local.values()),sources=dict(local),
  first_loss=rows[0]['loss'],last_loss=rows[-1]['loss'],
  peak_cuda_gib=max(x['cuda_peak_bytes'] for x in rows)/2**30))
config=yaml.safe_load((run/'resolved_config.yaml').read_text())
expected={ds['name'] for ds in config['dataset']['datasets']}
assert set(counts)==expected and sum(counts.values())==4000
log=(run/'train.log').read_text()
assert 'Validation - Step 100' in log
assert '100 steps' in log
checkpoint=Path(re.findall(r'Checkpoint saved to (.+)',log)[-1])
assert checkpoint.is_dir()
files=[dict(path=str(p),bytes=p.stat().st_size) for p in sorted(checkpoint.rglob('*')) if p.is_file()]
assert any(x['path'].endswith('pytorch_model_0.bin') and x['bytes']>0 for x in files)
result=dict(status='passed',run=str(run),world_size=8,micro_batch=5,global_batch=40,steps=100,
            samples=dict(counts),ranks=ranks,checkpoint=str(checkpoint),checkpoint_files=files)
(run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('ranks','checkpoint_files')},indent=2))
