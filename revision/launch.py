"""Launch only this revision's workers; never stop other users' processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tasks_extended import TASKS

here=Path(__file__).resolve().parent
priority=['sciml_poisson','sciml_burgers','sciml_darcy','sciml_fno']
priority += [x for x in TASKS if x.startswith(('matreg_','lowrank_','graph_','recsys_'))]
priority += [x for x in TASKS if x not in priority]
workers=[]
for gpu in (0,1,2):
    log=here/f'worker_gpu{gpu}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with open(log,'a') as f:
        p=subprocess.Popen([sys.executable,str(here/'suite.py'),'--tasks',*priority],
                           cwd=here.parent,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append({'gpu':gpu,'pid':p.pid,'log':str(log)})
out={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'workers':workers,
     'tasks':priority,'n_task_method_pairs':len(priority)*6,'policy':'one worker per GPU; all complete formal epochs; no other processes modified'}
(here/'launch.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
