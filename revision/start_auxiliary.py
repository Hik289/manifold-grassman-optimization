import json
import os
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
jobs=[]
for filename,gpu in [('preflight.py',1),('cifar_stability.py',2)]:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with open(here/(filename.removesuffix('.py')+'.log'),'a') as f:
        p=subprocess.Popen([sys.executable,str(here/filename)],cwd=here.parent,env=env,
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append({'script':filename,'pid':p.pid,'gpu':gpu})
(here/'auxiliary_launch.json').write_text(json.dumps(jobs,indent=2))
print(json.dumps(jobs,indent=2))
