"""Validate corrected task semantics before resuming the isolated workers."""
import json
from pathlib import Path
import subprocess
import sys
import torch
from tasks_extended import build

torch.set_num_threads(2)
here=Path(__file__).resolve().parent
out={}
for name in ['lowrank_128','lowrank_256','lowrank_512','recsys_movielens']:
    torch.manual_seed(42)
    t=build(name)
    if name.startswith('lowrank_'):assert t.meta['target_rank']==8
    else:
        assert t.meta['n_users']>5000
        assert t.meta['subset'].startswith('all input users')
    loss=t.loss(t.model,next(iter(t.train)))
    assert torch.isfinite(loss)
    loss.backward()
    out[name]={'meta':t.meta,'initial_loss':float(loss.detach()),'passed':True}
    print(name,t.meta,flush=True)
(here/'coverage_validation.json').write_text(json.dumps(out,indent=2))
if '--launch' in sys.argv:
    subprocess.run([sys.executable,str(here/'launch.py')],check=True)
