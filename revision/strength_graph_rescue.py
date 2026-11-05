"""Resume numerically interrupted graph searches without changing update rules.

Original optimizer/trainer/protocol sources and completed runs are immutable.
Numerical screening failures are retained and excluded; unrelated errors abort.
Formal numerical failures use the next validation-ranked candidate, never test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import strength_graph as original

HERE, OUT = original.HERE, original.OUT


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def safe_run(task, method, hp, epochs, path, formal):
    try:
        return original.run(task,method,hp,epochs,path,formal)
    except RuntimeError as error:
        msg=str(error).lower()
        if not any(s in msg for s in ['non-finite','nonfinite','nan','linalg.svd','linalg.eigh']):raise
        result=dict(status='diverged',task=task,method=method,hp=hp,formal=formal,
                    requested_epochs=epochs,error=str(error),traceback=traceback.format_exc(),
                    failure_class='numerical optimizer failure; update rule unchanged',
                    recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic(path/'numerical_failure.json',result)
        atomic(path/'summary.json',result)
        print('NUMERICAL CANDIDATE FAILURE',task,method,hp,str(error),flush=True)
        return result


def aggregate():
    results=[]
    for p in sorted((OUT/'graph').glob('*/*/formal/summary.json')):
        x=json.loads(p.read_text())
        if x.get('status')=='complete':results.append(x)
    protocol=json.loads((OUT/'graph_protocol.json').read_text())
    atomic(OUT/'graph_summary.json',dict(protocol=protocol,results=results,
           recovery_policy='graph_rescue_protocol.json; numerical failures only, no test-driven fallback'))


def main(tasks):
    # The shared original dump used one temporary name across processes.
    # Unique temporary files also make aggregate updates safe across workers.
    original.dump=atomic
    policy=dict(version='numerical-recovery-v1',
                original_protocol='graph_protocol.json',
                optimizer_update_changed=False,candidate_grids_changed=False,
                screening_failure='record the numerical failure, exclude that candidate, continue every remaining candidate',
                formal_failure='record failed full run; try candidates in descending screening-validation order',
                test_selection=False,completed_endpoints_overwritten=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                original_source_sha256=hashlib.sha256((HERE/'strength_graph.py').read_bytes()).hexdigest(),
                recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    policy_path=OUT/'graph_rescue_protocol.json'
    if policy_path.exists():assert json.loads(policy_path.read_text())['source_sha256']==policy['source_sha256']
    else:atomic(policy_path,policy)
    for task in tasks:
        for method in original.METHODS:
            folder=OUT/'graph'/task/method
            canonical=folder/'formal'
            if (canonical/'summary.json').exists():
                x=json.loads((canonical/'summary.json').read_text())
                if x.get('status')=='complete':continue
            screens=[safe_run(task,method,hp,100,folder/f'screen_{i}',False)
                     for i,hp in enumerate(original.candidates(method))]
            good=sorted([x for x in screens if x['status']=='complete'],
                        key=lambda x:x['validation']['score'],reverse=True)
            candidates=[dict(hp=x['hp'],status=x['status'],validation=x.get('validation'),
                             error=x.get('error')) for x in screens]
            if not good:
                atomic(folder/'failed.json',dict(reason='All nine candidates failed numerically',candidates=candidates))
                continue
            for attempt,best in enumerate(good):
                dest=folder/'numerical_recovery'/f'formal_attempt_{attempt}'
                result=safe_run(task,method,best['hp'],1000,dest,True)
                if result['status']!='complete':continue
                result=dict(result,execution_policy='numerical-recovery-v1',
                            recovery_attempt=attempt,source_manifest='graph_rescue_protocol.json')
                if canonical.exists() and any(canonical.iterdir()):
                    backup=folder/'numerical_recovery'/'pre_rescue_formal'
                    if not backup.exists():shutil.copytree(canonical,backup)
                canonical.mkdir(parents=True,exist_ok=True)
                shutil.copy2(dest/'trajectory.json',canonical/'trajectory.json')
                atomic(canonical/'summary.json',result)
                atomic(folder/'selection.json',dict(hp=best['hp'],
                       criterion='Highest final screening-validation accuracy among numerically completing candidates',
                       candidates=candidates,formal_fallback_attempt=attempt,
                       policy='graph_rescue_protocol.json; no test score used'))
                print('RESUMED COMPLETE',task,method,result,flush=True)
                break
            else:atomic(folder/'failed.json',dict(reason='All validation-ranked formal attempts failed numerically',candidates=candidates))
            aggregate()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',required=True)
    a=ap.parse_args();main(a.tasks)
