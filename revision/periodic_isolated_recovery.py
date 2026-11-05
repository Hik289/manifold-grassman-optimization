"""One fresh-process retry of numerically failed CiteSeer periodic candidates.

An unchanged candidate completed when diagnosed in a fresh process. Retest every
failed candidate once in isolation; preserve all original failures and never
retry in response to a test score. Arithmetic and the nine unique HPs unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import strength_graph_rescue as recovery
original=recovery.original
TASK='graph_citeseer'
METHOD='PeriodicDiag'
FOLDER=original.OUT/'graph'/TASK/METHOD


def child(hp,epochs,path,formal):
    original.dump=recovery.atomic
    recovery.safe_run(TASK,METHOD,hp,epochs,Path(path),formal)


def run_child(hp,epochs,path,formal):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'summary.json').exists():return json.loads((path/'summary.json').read_text())
    cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--hp',json.dumps(hp),
         '--epochs',str(epochs),'--path',str(path)]
    if formal:cmd.append('--formal')
    with open(path/'process.log','a') as log:
        subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((path/'summary.json').read_text())


def main():
    protocol=dict(task=TASK,method=METHOD,candidates=original.candidates(METHOD),
                  original_protocol='graph_protocol.json',
                  reason='An unchanged failed screening candidate completed in a fresh diagnostic process',
                  numerical_retry_limit_per_candidate=1,optimizer_arithmetic_changed=False,
                  selection='Highest final screening-validation accuracy; then full fresh-process formal run',
                  formal_failure='Try next validation-ranked finite candidate; preserve every attempt',
                  test_driven_retry=False,seed=42,screening_epochs=100,formal_epochs=1000,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    pp=original.OUT/'graph_process_retry_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['source_sha256']==protocol['source_sha256']
    else:recovery.atomic(pp,protocol)
    if (FOLDER/'formal/summary.json').exists():
        if json.loads((FOLDER/'formal/summary.json').read_text()).get('status')=='complete':return
    results=[]
    for i,hp in enumerate(protocol['candidates']):
        old=FOLDER/f'screen_{i}'/'summary.json'
        previous=json.loads(old.read_text()) if old.exists() else None
        if previous and previous.get('status')=='complete':
            result=previous
        else:
            result=run_child(hp,100,FOLDER/'process_retry'/f'screen_{i}',False)
        results.append(result)
        print('ISOLATED SCREEN',i,hp,result['status'],result.get('validation'),flush=True)
    good=sorted([x for x in results if x['status']=='complete'],
                key=lambda x:x['validation']['score'],reverse=True)
    for attempt,best in enumerate(good):
        dest=FOLDER/'process_retry'/f'formal_attempt_{attempt}'
        result=run_child(best['hp'],1000,dest,True)
        if result['status']!='complete':continue
        canonical=FOLDER/'formal';canonical.mkdir(parents=True,exist_ok=True)
        result=dict(result,execution_policy='fresh-process-numerical-retry-v1',
                    source_manifest='graph_process_retry_protocol.json',recovery_attempt=attempt)
        shutil.copy2(dest/'trajectory.json',canonical/'trajectory.json')
        recovery.atomic(canonical/'summary.json',result)
        recovery.atomic(FOLDER/'selection.json',dict(hp=best['hp'],
          criterion=protocol['selection'],candidates=[dict(hp=x['hp'],status=x['status'],
          validation=x.get('validation'),error=x.get('error')) for x in results],
          formal_fallback_attempt=attempt,policy='graph_process_retry_protocol.json'))
        recovery.atomic(FOLDER/'recovered.json',dict(status='complete',
            original_failure_records_retained=True,policy='graph_process_retry_protocol.json'))
        recovery.aggregate();print('ISOLATED FORMAL COMPLETE',result,flush=True);break
    else:
        recovery.atomic(FOLDER/'process_retry'/'failed.json',dict(status='unresolved',
             reason='No isolated validation-ranked candidate completed formal training',
             original_failures_retained=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--child',action='store_true')
    ap.add_argument('--hp');ap.add_argument('--epochs',type=int);ap.add_argument('--path')
    ap.add_argument('--formal',action='store_true');a=ap.parse_args()
    child(json.loads(a.hp),a.epochs,a.path,a.formal) if a.child else main()
