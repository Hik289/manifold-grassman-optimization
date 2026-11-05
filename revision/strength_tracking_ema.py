"""Stronger controls: full-covariance EMA SVD and a nonadiabatic drift regime.

Independent protocol recorded before this experiment; first study immutable.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from strength_study import OUT, HERE, GrassRoute, PeriodicDiag, seeded, dump, memory
from grassroute import _polar_orthogonal


class EMAPeriodic(PeriodicDiag):
    """Same parameter/moment skeleton; full normalized covariance EMA oracle.

    Reimplements only the basis-refresh operation. The ordinary periodic
    routine is prevented from refreshing; M,V are transported once below.
    Full covariance memory and computation are counted, not treated as free.
    """
    def __init__(self, params, period, beta, **kwargs):
        super().__init__(params, svd_every=10**12, **kwargs)
        self.period, self.beta = period, beta

    @torch.no_grad()
    def _step_routed(self, group, p):
        super()._step_routed(group, p)
        st = self.state[p]
        g = p.grad.reshape(p.shape[0],-1).float()
        c = g.T@g/(g.square().sum()+group['eps_res'])
        if 'covariance_ema' not in st: st['covariance_ema'] = torch.zeros_like(c)
        st['covariance_ema'].mul_(self.beta).add_(c,alpha=1-self.beta)
        if st['svd_ctr']%self.period==0:
            unew = torch.linalg.eigh(st['covariance_ema']).eigenvectors[:,-group['rank']:].contiguous()
            rot = _polar_orthogonal(unew.T@st['U'])
            st['M'].copy_(st['M']@rot.T)
            st['V'].copy_(st['V']@rot.T.square())
            st['U'] = unew


def main():
    methods=['GrassRoute','Periodic5','Periodic25','EMA5_90','EMA25_90','EMA5_99','EMA25_99']
    protocol=dict(d=64,rank=4,rows=[16,128],drift=[0.,.001,.005,.02,.1],steps=2000,
                  methods=methods,signal_eigenvalue=1.,tail_eigenvalue=.03,seed=42,
                  grassmann_lr=3.,momentum=.9,
                  ema_betas=[.9,.99],ema_periods=[5,25],
                  evaluation='pre-update mean sin^2 over steps 1501:2000',
                  initialization='common random QR',
                  scope='Known-covariance synthetic gradient stream; not end-to-end task accuracy',
                  rationale='Strong smoothing controls and nonadiabatic drift; retains all original drift conditions',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependencies={f:hashlib.sha256((HERE.parent/f).read_bytes()).hexdigest() for f in
                                ['revision/strength_study.py','experiments/grassroute.py','experiments/baselines.py']},
                  recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    pp=OUT/'tracking_extended_protocol.json'
    if pp.exists(): assert json.loads(pp.read_text())['source_sha256']==protocol['source_sha256']
    else: dump(pp,protocol)
    results=[]
    for m,omega in itertools.product(protocol['rows'],protocol['drift']):
        key=f'm{m}_omega{omega:g}';path=OUT/'tracking_extended'/(key+'.json')
        if path.exists():results.append({k:v for k,v in json.loads(path.read_text()).items() if k!='records'});continue
        seeded();d,r=protocol['d'],protocol['rank'];initial=torch.linalg.qr(torch.randn(d,r)).Q
        opts={}
        for method in methods:
            p=torch.nn.Parameter(torch.zeros(m,d));kw=dict(lr=0.,rank=r,grassmann_lr=3.,reset_mode='diagonal')
            if method=='GrassRoute':opt=GrassRoute([p],**kw)
            elif method.startswith('Periodic'):opt=PeriodicDiag([p],svd_every=int(method[8:]),**kw)
            else:
                parts=method[3:].split('_');opt=EMAPeriodic([p],period=int(parts[0]),beta=int(parts[1])/100,**kw)
            opt._init_routed_state(opt.state[p],p,r,'cpu',p.dtype);opt.state[p]['U']=initial.clone()
            if method!='GrassRoute':opt.state[p]['svd_ctr']=0
            opts[method]=(p,opt)
        gen=torch.Generator().manual_seed(4201);records=[];elapsed={n:0. for n in methods};maxorth=0.
        for t in range(protocol['steps']):
            a=omega*t;c,s=math.cos(a),math.sin(a)
            target=torch.zeros(d,r);target[:r]=c*torch.eye(r);target[r:2*r]=s*torch.eye(r)
            z=torch.randn(m,d,generator=gen);z[:,r:]*=math.sqrt(.03)
            x,y=z[:,:r].clone(),z[:,r:2*r].clone();z[:,:r]=c*x-s*y;z[:,r:2*r]=s*x+c*y
            rec={'step':t+1,'methods':{}}
            for name,(p,opt) in opts.items():
                u=opt.state[p]['U'];err=max(0.,1-float((u.T@target).square().sum())/r)
                p.grad=z;start=time.perf_counter();opt.step();elapsed[name]+=time.perf_counter()-start
                un=opt.state[p]['U'];maxorth=max(maxorth,float((un.T@un-torch.eye(r)).norm()))
                rec['methods'][name]={'sin2':err}
            records.append(rec)
        summary={n:dict(mean_sin2_last500=float(np.mean([x['methods'][n]['sin2'] for x in records[-500:]])),
                        state_bytes=memory(o),optimizer_seconds=elapsed[n]) for n,(_,o) in opts.items()}
        assert maxorth<1e-4
        result=dict(key=key,rows=m,drift=omega,summary=summary,max_orthogonality_error=maxorth)
        dump(path,dict(result,records=records));results.append(result)
        dump(OUT/'tracking_extended_summary.json',dict(protocol=protocol,conditions=results))
        print(key,summary,flush=True)


if __name__=='__main__':main()
