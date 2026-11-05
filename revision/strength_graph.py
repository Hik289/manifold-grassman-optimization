"""Equal-budget Planetoid follow-up, including official SubTrack++.

CPU-only companion to the existing GPU suite. Original source and runs unchanged.
Validation is the sole selection criterion. All complete endpoints are retained.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from strength_study import PeriodicDiag, GrassRoute, seeded, dump, memory, HERE, OUT
from suite import Official, groups
from tasks_extended import build

torch.set_num_threads(2)
METHODS=['SGD','AdamW','Fixed','PeriodicDiag','GrassRoute','SubTrackOfficial']


def candidates(method):
    if method=='AdamW':return [dict(lr=x) for x in [.0003,.0005,.001,.002,.003,.005,.01,.02,.03]]
    if method=='SGD':return [dict(lr=x) for x in [.001,.003,.01,.02,.03,.05,.1,.2,.3]]
    key,vals={'Fixed':('residual_ratio',[.3,1.,3.]),'PeriodicDiag':('period',[5,25,100]),
              'GrassRoute':('rho',[.3,1.,3.]),'SubTrackOfficial':('rho',[.1,1.,3.])}[method]
    return [dict(lr=lr,**{key:v}) for lr,v in itertools.product([.001,.003,.01],vals)]


def run(taskname,method,hp,epochs,path,formal):
    if (path/'summary.json').exists():return json.loads((path/'summary.json').read_text())
    seeded();task=build(taskname);gs,routed=groups(task,hp['lr'])
    if method=='SGD':opt=torch.optim.SGD(task.model.parameters(),lr=hp['lr'],momentum=.9)
    elif method=='AdamW':opt=torch.optim.AdamW(task.model.parameters(),lr=hp['lr'],weight_decay=0.)
    elif method=='SubTrackOfficial':opt=Official(gs,hp['lr'],0.,hp['rho'])
    elif method=='PeriodicDiag':opt=PeriodicDiag(gs,lr=hp['lr'],svd_every=hp['period'],reset_mode='diagonal')
    else:opt=GrassRoute(gs,lr=hp['lr'],grassmann_lr=hp.get('rho',0.),reset_mode='diagonal')
    records=[];start=time.monotonic()
    for ep in range(epochs):
        task.model.train();rate=hp['lr']*(.01+.99*.5*(1+math.cos(math.pi*ep/epochs)))
        for g in opt.param_groups:
            g['lr']=rate
            if 'lr_perp' in g:g['lr_perp']=rate*hp.get('residual_ratio',1.)
        opt.zero_grad();loss=task.loss(task.model,None)
        if not torch.isfinite(loss):
            x=dict(status='diverged',epoch=ep+1,method=method,hp=hp,formal=formal)
            dump(path/'summary.json',x);return x
        loss.backward();rng=torch.get_rng_state();opt.step();torch.set_rng_state(rng)
        rec=dict(epoch=ep+1,train_loss=float(loss.detach()),wall_s=time.monotonic()-start)
        if ep==0 or (ep+1)%50==0 or ep==epochs-1:rec['validation']=task.evaluate(task.model,'val')
        records.append(rec)
        if (ep+1)%100==0:dump(path/'trajectory.json',records)
    x=dict(task=taskname,method=method,status='complete',formal=formal,epochs=epochs,n_epochs=len(records),
           seed=42,hp=hp,routed_layers=routed,meta=task.meta,validation=task.evaluate(task.model,'val'),
           test=task.evaluate(task.model,'test') if formal else None,state_bytes=memory(opt),wall_s=time.monotonic()-start)
    dump(path/'trajectory.json',records);dump(path/'summary.json',x);return x


def main(selected_tasks=None):
    protocol=dict(tasks=['graph_cora','graph_citeseer','graph_pubmed'],methods=METHODS,
                  search={m:candidates(m) for m in METHODS},screening_epochs=100,formal_epochs=1000,
                  seed=42,selection='maximum FINAL validation accuracy; fresh formal initialization',
                  paired_initialization=True,paired_dropout=True,device='cpu',no_early_stopping=True,
                  scope='Independent Planetoid follow-up; transductive public splits; not iid generalization validation',
                  rationale='Check qualified PubMed result against transport-corrected periodic and two other Planetoid controls',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependencies={f:hashlib.sha256((HERE.parent/f).read_bytes()).hexdigest() for f in
                    ['revision/strength_study.py','revision/tasks_extended.py','revision/suite.py','experiments/grassroute.py','experiments/baselines.py','experiments/tasks.py']},
                  recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    pp=OUT/'graph_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['source_sha256']==protocol['source_sha256']
    else:dump(pp,protocol)
    for task,method in itertools.product(selected_tasks or protocol['tasks'],METHODS):
        folder=OUT/'graph'/task/method
        screens=[run(task,method,hp,100,folder/f'screen_{i}',False) for i,hp in enumerate(candidates(method))]
        good=[x for x in screens if x['status']=='complete']
        if not good:
            dump(folder/'failed.json',dict(reason='All screening candidates diverged'));continue
        best=max(good,key=lambda x:x['validation']['score'])
        dump(folder/'selection.json',dict(hp=best['hp'],criterion=protocol['selection'],
             candidates=[dict(hp=x['hp'],status=x['status'],validation=x.get('validation')) for x in screens]))
        result=run(task,method,best['hp'],1000,folder/'formal',True)
        results=[json.loads(p.read_text()) for p in (OUT/'graph').glob('*/*/formal/summary.json')]
        dump(OUT/'graph_summary.json',dict(protocol=protocol,results=results))
        print(task,method,result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',choices=['graph_cora','graph_citeseer','graph_pubmed'])
    a=ap.parse_args();main(a.tasks)
