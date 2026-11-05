"""Frozen six-task extension: full epochs, equal validation budgets, all outcomes.

New CPU jobs use existing unmodified GrassRoute plus an efficient thin-SVD
periodic comparator with the same residual and diagonal-moment formulas.
"""
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'experiments'))
from grassroute import GrassRoute,_polar_orthogonal
from suite import state_bytes
from official_cpu_resume import DeviceOfficial

torch.set_num_threads(2)
DATA=HERE/'newtask_data'
OUT=HERE/'newtask_results'
TASKS=['image_denoise','image_inpaint','tab_spambase','tab_adult','text_sms','text_20news']
METHODS=['SGD','AdamW','Fixed','PeriodicThinSVD','GrassRoute','SubTrackOfficial']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)


def seeded():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)


class StaticPeriodic(GrassRoute):
    """No unused routing momentum in fixed/periodic state; no d-by-d covariance.

    Direct SVD of G yields the same top right eigenspace as G^T G while avoiding
    condition-number squaring. M and diag(V) both follow Procrustes rotations.
    """
    def __init__(self,params,period=None,**kw):
        super().__init__(params,grassmann_lr=0.,**kw);self.period=period

    @torch.no_grad()
    def _step_routed(self,group,p):
        st=self.state[p];m=p.shape[0];d=math.prod(p.shape[1:]);r=group['rank']
        if 'U' not in st:
            self._init_routed_state(st,p,r,p.device,p.dtype);st.pop('H')
        st['step']+=1;st['v_step']+=1;t=st['step'];u=st['U'];g=p.grad.reshape(m,d).float()
        z=g@u;gp=g-z@u.T;b1,b2=group['beta1'],group['beta2'];M,V=st['M'],st['V']
        M.mul_(b1).add_(z,alpha=1-b1);V.mul_(b2).addcmul_(z,z,value=1-b2)
        direction=(M/(1-b1**t))/((V/(1-b2**st['v_step'])).sqrt()+group['eps_adam'])
        scale=(g.norm()/(gp.norm()+group['eps_res'])).clamp(group['s_min'],group['s_max'])
        w=p.data.reshape(m,d)
        if group['weight_decay']:w.mul_(1-group['lr']*group['weight_decay'])
        w.add_(direction@u.T,alpha=-group['lr']);w.add_(scale*gp,alpha=-group['lr_perp'])
        if self.period and t%self.period==0:
            try:v=torch.linalg.svd(g,full_matrices=False).Vh
            except torch.linalg.LinAlgError:
                v=torch.linalg.svd(g.double(),full_matrices=False).Vh.float()
            unew=v[:r].T.contiguous();rotation=_polar_orthogonal(unew.T@u)
            M.copy_(M@rotation.T);V.copy_(V@rotation.T.square());st['U']=unew


class SparseHead(nn.Module):
    def __init__(self,d,c):
        super().__init__();self.fc1=nn.Linear(d,64);self.out=nn.Linear(64,c)
    def forward(self,x):
        h=torch.sparse.mm(x,self.fc1.weight.T)+self.fc1.bias if x.layout!=torch.strided else self.fc1(x)
        return self.out(F.dropout(F.relu(h),.1,self.training))


def load_x(folder,split):
    if (folder/f'x_{split}.npz').exists():
        a=sparse.load_npz(folder/f'x_{split}.npz').tocoo()
        return torch.sparse_coo_tensor(np.vstack([a.row,a.col]),a.data,a.shape,dtype=torch.float32).coalesce()
    return torch.from_numpy(np.load(folder/f'x_{split}.npy'))


class Task:
    def __init__(self,name):
        self.name=name;folder=DATA/name;self.meta=json.loads((folder/'meta.json').read_text())
        for fn,digest in self.meta['file_sha256'].items():assert sha(folder/fn)==digest
        self.x={s:load_x(folder,s) for s in ['train','val','test']}
        self.y={s:torch.from_numpy(np.load(folder/f'y_{s}.npy')) for s in ['train','val','test']}
        d=self.x['train'].shape[1];self.family=self.meta['family'];self.higher=self.meta['higher']
        if self.family=='inverse_imaging':
            inp=d*(2 if name.endswith('inpaint') else 1)
            self.model=nn.Sequential(nn.Linear(inp,128),nn.Tanh(),nn.Linear(128,32),nn.Tanh(),
                                     nn.Linear(32,128),nn.Tanh(),nn.Linear(128,d),nn.Sigmoid())
        elif self.family=='text':self.model=SparseHead(d,int(self.y['train'].max())+1)
        else:self.model=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,2))
        self.evaluation_corruptions={}
        if self.family=='inverse_imaging':
            for split,seed in [('val',42001),('test',42002)]:
                self.evaluation_corruptions[split]=self.corrupt(self.x[split],torch.Generator().manual_seed(seed))

    def corrupt(self,x,generator=None):
        if self.name.endswith('denoise'):
            noise=torch.randn(x.shape,generator=generator)*.3
            return (x+noise).clamp(0,1),None
        mask=(torch.rand(x.shape,generator=generator)>=.5).float()
        return torch.cat([x*mask,mask],1),mask

    def loss(self):
        if self.family!='inverse_imaging':return F.cross_entropy(self.model(self.x['train']),self.y['train'])
        x,mask=self.corrupt(self.x['train']);pred=self.model(x);err=(pred-self.y['train']).square()
        return err.mean() if mask is None else (err*(1-mask)).sum()/(1-mask).sum().clamp_min(1)

    @torch.no_grad()
    def evaluate(self,split):
        self.model.eval();y=self.y[split]
        if self.family=='inverse_imaging':
            x,mask=self.evaluation_corruptions[split];pred=self.model(x);err=(pred-y).square()
            mse=float(err.mean()) if mask is None else float((err*(1-mask)).sum()/(1-mask).sum().clamp_min(1))
            full=mse if mask is None else float((err*(1-mask)).mean())
            return dict(score=mse,mse=mse,full_image_mse=full,psnr=-10*math.log10(max(full,1e-30)),n=len(y))
        logits=self.model(self.x[split]);prob=logits.softmax(-1).numpy();target=y.numpy();pred=prob.argmax(1)
        c=prob.shape[1];cm=np.bincount(c*target+pred,minlength=c*c).reshape(c,c)
        denominator=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(c),where=denominator!=0).mean()
        result=dict(accuracy=float((target==pred).mean()),macro_f1=float(f1),
                    cross_entropy=float(F.cross_entropy(logits,y)),n=len(y))
        if c==2:
            positive=target==1;npos=int(positive.sum());nneg=len(target)-npos
            result['auroc']=float((rankdata(prob[:,1])[positive].sum()-npos*(npos+1)/2)/(npos*nneg))
        result['score']=result[self.meta['metric']];return result


def groups(task,lr):
    routed=[];other=[];names=[]
    for name,p in task.model.named_parameters():
        d=math.prod(p.shape[1:]) if p.ndim>=2 else 0
        r=min(8,d-1,max(1,p.shape[0]//2)) if d else 0
        if p.ndim>=2 and min(p.shape[0],d)>=4 and r<d:
            routed.append(dict(params=[p],routed=True,rank=r,lr=lr));names.append(dict(name=name,shape=list(p.shape),rank=r))
        else:other.append(p)
    if other:routed.append(dict(params=other,routed=False,lr=lr))
    return routed,names


def candidates(method):
    if method=='AdamW':return [dict(lr=x) for x in [.0003,.0005,.001,.002,.003,.005,.01,.02,.03]]
    if method=='SGD':return [dict(lr=float(x)) for x in np.geomspace(.001,3.,9)]
    if method=='SubTrackOfficial':return [dict(lr=x,rho=r) for x,r in itertools.product([.001,.003,.01],[.1,1.,3.])]
    grid=[]
    for i,lr in enumerate([.001,.003,.01]):
        for j,ratio in enumerate([1.,10.,100.]):
            hp=dict(lr=lr,residual_ratio=ratio)
            if method=='GrassRoute':hp['rho']=[.3,1.,3.][(i+j)%3]
            if method=='PeriodicThinSVD':hp['period']=[5,25,100][(i+j)%3]
            grid.append(hp)
    return grid


def optimizer(task,method,hp):
    gs,names=groups(task,hp['lr'])
    if method=='AdamW':o=torch.optim.AdamW(task.model.parameters(),lr=hp['lr'],weight_decay=0.)
    elif method=='SGD':o=torch.optim.SGD(task.model.parameters(),lr=hp['lr'],momentum=.9)
    elif method=='SubTrackOfficial':o=DeviceOfficial(gs,hp['lr'],0.,hp['rho'])
    else:
        kw=dict(lr=hp['lr'],lr_perp=hp['lr']*hp['residual_ratio'],reset_mode='diagonal',weight_decay=0.)
        if method=='GrassRoute':o=GrassRoute(gs,grassmann_lr=hp['rho'],**kw)
        else:o=StaticPeriodic(gs,period=hp.get('period'),**kw)
    return o,names


def freeze():
    protocol=dict(seed=42,tasks=TASKS,methods=METHODS,screening_epochs=200,formal_epochs=1000,
        search={m:candidates(m) for m in METHODS},selection='final validation primary metric; fresh formal initialization; one final test evaluation',
        training='one full-data batch per epoch; all examples on every pass; no early stopping',
        family_count=3,families={'text':['text_sms','text_20news'],'tabular':['tab_spambase','tab_adult'],'inverse_imaging':['image_denoise','image_inpaint']},
        repetitions='one seed, no statistical-significance claims',
        retention='all attempted candidates, failed runs, and formal endpoints retained; no task dropped after observing test',
        primary_win='strictly better primary test metric than both AdamW and thin periodic SVD, after all six methods complete',
        memory_tradeoff='reported separately: >=50% state saving, within 1 percentage point (classification) or 5% relative MSE (imaging) of AdamW; not called an accuracy win',
        rank_policy='rank 8 capped by input dimension minus one and half the output dimension; small matrices full Adam fallback',
        periodic='direct thin SVD of G every selected interval; Procrustes M plus diagonal V transport; no unused H state',
        budget_note='nine equally costly candidate runs; routed grids balance residual ratio with rho/refresh rather than full Cartesian tuning',
        source_sha256={f:sha(ROOT/f) for f in ['revision/new_tasks_suite.py','experiments/grassroute.py','revision/official_cpu_resume.py','revision/suite.py']},
        data_meta_sha256={t:sha(DATA/t/'meta.json') for t in TASKS},
        recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['source_sha256']==protocol['source_sha256'];assert old['data_meta_sha256']==protocol['data_meta_sha256']
        return old
    atomic(path,protocol);return protocol


def run(name,method,hp,epochs,folder,formal):
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    seeded();task=Task(name);opt,routed=optimizer(task,method,hp);records=[];start_ep=0;elapsed=0.
    cfg=dict(task=name,method=method,hp=hp,epochs=epochs,seed=42,formal=formal,routed_layers=routed,
             data=task.meta,protocol_sha256=sha(OUT/'protocol.json'))
    atomic(folder/'config.json',cfg)
    cp=folder/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,weights_only=False,map_location='cpu')
        task.model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        records=state['records'];start_ep=len(records);elapsed=state['elapsed']
        torch.set_rng_state(state['rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    begin=time.monotonic()
    try:
        for ep in range(start_ep,epochs):
            task.model.train();rate=hp['lr']*(.01+.99*.5*(1+math.cos(math.pi*ep/epochs)))
            for g in opt.param_groups:
                g['lr']=rate
                if 'lr_perp' in g:g['lr_perp']=rate*hp.get('residual_ratio',1.)
            opt.zero_grad();loss=task.loss()
            if not torch.isfinite(loss):raise FloatingPointError('non-finite training objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in task.model.parameters()):
                raise FloatingPointError('non-finite parameter gradient')
            rng=torch.get_rng_state();opt.step();torch.set_rng_state(rng)
            rec=dict(epoch=ep+1,train_loss=float(loss.detach()),wall_s=elapsed+time.monotonic()-begin)
            if ep==0 or (ep+1)%50==0 or ep==epochs-1:rec['validation']=task.evaluate('val')
            records.append(rec)
            if (ep+1)%100==0 or ep==epochs-1:
                atomic(folder/'trajectory.json',records)
                state=dict(model=task.model.state_dict(),optimizer=opt.state_dict(),records=records,
                           elapsed=rec['wall_s'],rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
                temp=cp.with_suffix('.pt.tmp');torch.save(state,temp);temp.replace(cp)
                print(name,method,'formal' if formal else 'screen',ep+1,rec.get('validation'),flush=True)
        result=dict(cfg,status='complete',n_epochs=len(records),validation=task.evaluate('val'),
                    test=task.evaluate('test') if formal else None,state_bytes=state_bytes(opt),
                    final_train_loss=records[-1]['train_loss'],wall_s=elapsed+time.monotonic()-begin)
    except (FloatingPointError,torch.linalg.LinAlgError) as error:
        atomic(folder/'trajectory.json',records)
        result=dict(cfg,status='diverged',completed_epochs=len(records),error=str(error),traceback=traceback.format_exc(),test=None)
    atomic(folder/'summary.json',result);return result


def worker(tasks):
    protocol=freeze()
    for name,method in itertools.product(tasks,METHODS):
        folder=OUT/name/method;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'formal/summary.json').exists():continue
        lock=open(folder/'worker.lock','a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        try:
            trials=[run(name,method,hp,200,folder/f'screen_{i}',False) for i,hp in enumerate(candidates(method))]
            valid=[x for x in trials if x['status']=='complete']
            if not valid:
                atomic(folder/'failed.json',dict(reason='All nine validation candidates failed numerically'));continue
            meta=json.loads((DATA/name/'meta.json').read_text())
            best=(max if meta['higher'] else min)(valid,key=lambda x:x['validation']['score'])
            atomic(folder/'selection.json',dict(hp=best['hp'],criterion=protocol['selection'],
                    candidates=[dict(hp=x['hp'],status=x['status'],validation=x.get('validation'),error=x.get('error')) for x in trials]))
            result=run(name,method,best['hp'],1000,folder/'formal',True)
            print('FORMAL ENDPOINT',name,method,result.get('test'),result['status'],flush=True)
        except Exception as error:
            atomic(folder/'failed.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
            traceback.print_exc()
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',default=TASKS,choices=TASKS)
    ap.add_argument('--register',action='store_true');ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.register:print(json.dumps(freeze(),indent=2))
    elif a.smoke:
        freeze()
        for name,method in itertools.product(a.tasks,METHODS):
            result=run(name,method,candidates(method)[4],2,OUT/'smoke'/name/method,False)
            assert result['status']=='complete'
    else:worker(a.tasks)
