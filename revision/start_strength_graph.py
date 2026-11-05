"""Three bounded CPU workers; no existing process or GPU reservation modified."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
jobs=[]
for task in ['graph_pubmed','graph_cora','graph_citeseer']:
    # Avoid a register-file creation race: the first worker records the protocol
    # before the other workers launch. This launcher itself does not run training.
    if jobs:
        for _ in range(40):
            if (here/'strength_results/graph_protocol.json').exists():break
            time.sleep(.25)
        else:raise RuntimeError('First worker did not record its protocol')
    with open(here/f'strength_{task}.log','a') as f:
        p=subprocess.Popen([sys.executable,str(here/'strength_graph.py'),'--tasks',task],
            cwd=here.parent,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append(dict(task=task,pid=p.pid,device='cpu',threads=2))
(here/'strength_graph_launch.json').write_text(json.dumps(jobs,indent=2))
print(json.dumps(jobs))
