"""Independent endpoint, selection, stream and transport consistency checks."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from strength_study import OUT, HERE, PeriodicDiag, seeded
from baselines import PeriodicSVDRouter
from grassroute import _polar_orthogonal


def main():
    checks={}
    protocol=json.loads((OUT/'pinn_protocol.json').read_text())
    assert protocol['source_sha256']==hashlib.sha256((HERE/'strength_study.py').read_bytes()).hexdigest()
    complete=[];initial={};screen_count=0
    for task in protocol['tasks']:
        initial[task]=[]
        for method in protocol['methods']:
            root=OUT/'pinn'/task/method
            selected=json.loads((root/'selection.json').read_text())
            actual=[]
            for i,hp in enumerate(protocol['search'][method]):
                folder=root/f'screen_{i}';s=json.loads((folder/'summary.json').read_text())
                assert s['hp']==hp
                if s['status']=='complete':
                    tr=json.loads((folder/'trajectory.json').read_text())
                    assert len(tr)==s['epochs']==s['n_epochs']==100
                    actual.append(s)
                screen_count+=1
            best=min(actual,key=lambda x:x['validation']['score'])
            assert selected['hp']==best['hp']
            s=json.loads((root/'formal/summary.json').read_text());tr=json.loads((root/'formal/trajectory.json').read_text())
            assert s['formal'] and s['seed']==42 and s['test'] is not None
            assert s['hp']==best['hp']
            assert len(tr)==s['epochs']==s['n_epochs']==2000
            assert all(x['epoch']==i+1 and math.isfinite(x['train_loss']) for i,x in enumerate(tr))
            initial[task].append(tr[0]['train_loss'])
            complete.append(dict(task=task,method=method,path=str(root.relative_to(HERE)),
                                 sha256=hashlib.sha256((root/'formal/summary.json').read_bytes()).hexdigest()))
    for vals in initial.values():assert max(vals)-min(vals)<1e-10
    checks['pinn']=dict(verified_formal=complete,screen_candidates=screen_count,paired_initial_losses=initial)
    ex=json.loads((OUT/'tracking_extended_summary.json').read_text())
    assert len(ex['conditions'])==10
    maxdelta=0.;conditions=0
    for condition in ex['conditions']:
        p=OUT/'tracking_extended'/(condition['key']+'.json');a=json.loads(p.read_text());rr=a['records']
        assert len(rr)==2000 and all(r['step']==i+1 for i,r in enumerate(rr))
        for method in ex['protocol']['methods']:
            errors=[r['methods'][method]['sin2'] for r in rr]
            assert min(errors)>=0 and max(errors)<=1+1e-6
            assert abs(np.mean(errors[-500:])-a['summary'][method]['mean_sin2_last500'])<1e-12
        for name in ['EMA5_90','EMA25_90','EMA5_99','EMA25_99']:
            assert a['summary'][name]['state_bytes']-a['summary']['GrassRoute']['state_bytes']==64*64*4
        old=OUT/'tracking'/(condition['key']+'.json')
        if old.exists():
            b=json.loads(old.read_text())
            for method in ['GrassRoute','Periodic5','Periodic25']:
                delta=max(abs(x['methods'][method]['sin2']-y['methods'][method]['sin2']) for x,y in zip(rr,b['records']))
                maxdelta=max(maxdelta,delta)
            conditions+=1
    assert maxdelta<1e-7
    checks['tracking']=dict(extended_conditions=10,steps_each=2000,repeated_conditions=conditions,
                            max_repeated_stream_error=maxdelta,full_covariance_extra_bytes=64*64*4)
    # Exact covariance-diagonal transport arithmetic at a basis refresh.
    seeded();m,d,r=7,12,3
    u=torch.linalg.qr(torch.randn(d,r)).Q;grad=torch.randn(m,d)
    opts=[]
    for cls in [PeriodicSVDRouter,PeriodicDiag]:
        p=torch.nn.Parameter(torch.zeros(m,d));o=cls([p],rank=r,lr=0.,svd_every=5,reset_mode='diagonal')
        o._init_routed_state(o.state[p],p,r,'cpu',p.dtype);o.state[p]['U']=u.clone();o.state[p]['svd_ctr']=4
        o.state[p]['M']=torch.full((m,r),.2);o.state[p]['V']=torch.arange(1,m*r+1).reshape(m,r).float()
        p.grad=grad;o.step();opts.append(o.state[p])
    old,new=opts;rot=_polar_orthogonal(new['U'].T@u)
    expected=old['V']@rot.T.square();error=float((new['V']-expected).abs().max())
    assert error<1e-6
    assert old['v_step']==new['v_step'] and torch.allclose(old['M'],new['M'])
    checks['periodic_transport']=dict(max_absolute_error=error,moment_age_preserved=True)
    (HERE/'strength_verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='verified_formal'}) for k,v in checks.items()},indent=2))


if __name__=='__main__':main()
