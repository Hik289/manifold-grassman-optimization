"""Resumable, validation-selected experiment completion on the original HPC.

One GPU worker claims a task/method atomically. All formal epochs are complete
passes (PDEs/graphs use one full-batch step), seed 42, no early stopping.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'experiments'))
from grassroute import GrassRoute
from baselines import PeriodicSVDRouter
from tasks_extended import build, TASKS, DEV

OUT=HERE/'results'
METHODS=['SGD','AdamW','Fixed','PeriodicSVD','GrassRoute','SubTrackOfficial']


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)


def groups(task,lr):
    from routing import is_routable_matrix,FAMILY_RANK
    rank=FAMILY_RANK.get(task.family,16);out=[];other=[];names=[]
    for name,p in task.model.named_parameters():
        d=math.prod(p.shape[1:]) if p.ndim>=2 else 0
        r=min(rank,d,max(1,p.shape[0]//2)) if d else 0
        if is_routable_matrix(name,p,task.family) and min(p.shape[0],d)>=4 and r<d:
            out.append({'params':[p],'routed':True,'rank':r,'lr':lr})
            names.append({'name':name,'shape':list(p.shape),'rank':r})
        else:other.append(p)
    if other:out.append({'params':other,'routed':False,'lr':lr})
    return out,names


class Official:
    """Upstream optimizer, with a shape-only adapter for convolution tensors."""
    def __init__(self,gs,lr,wd,glr):
        path=HERE/'third_party/SubTrack/low_rank_torch'
        if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
        # Upstream __init__ eagerly imports optional 8-bit packages. Load only
        # its AdamW module without modifying any upstream source.
        if 'low_rank_torch' not in sys.modules:
            package=types.ModuleType('low_rank_torch');package.__path__=[str(path)]
            sys.modules['low_rank_torch']=package
        cls=importlib.import_module('low_rank_torch.adamw').AdamW
        official=[]
        for g in gs:
            z={'params':g['params'],'lr':lr,'weight_decay':wd}
            if g.get('routed'):
                z.update(rank=g['rank'],scale=1.,proj_type='std',
                         st_init_step_size=glr,subspace_update_method='subtrack',
                         st_step_size_scheduler='iterative_decrease',st_step_size_coef=.1,
                         st_noise_sigma2=0.,st_subspace_coef=1.,subspace_update_interval=25,
                         adaptive_optimizer=True,recovery_scaling=True)
            official.append(z)
        self.opt=cls(official,lr=lr,weight_decay=wd,no_deprecation_warning=True,adaptive_optimizer=True)
        self.param_groups,self.state=self.opt.param_groups,self.opt.state
    def zero_grad(self):self.opt.zero_grad(set_to_none=True)
    def state_dict(self):return self.opt.state_dict()
    def load_state_dict(self,x):self.opt.load_state_dict(x)
    def step(self):
        restore=[]
        try:
            for g in self.param_groups:
                if 'rank' not in g:continue
                for p in g['params']:
                    if p.ndim>2 and p.grad is not None:
                        shape=p.shape;grad=p.grad;p.grad=None;p.data=p.data.reshape(shape[0],-1)
                        p.grad=grad.reshape(shape[0],-1);restore.append((p,shape,grad))
            self.opt.step()
        finally:
            for p,shape,grad in restore:p.grad=None;p.data=p.data.reshape(shape);p.grad=grad


def optimizer(task,method,lr,glr):
    gs,names=groups(task,lr)
    wd=.01 if task.family in ('cnn','vit') else 0.
    if method=='SGD':opt=torch.optim.SGD(task.model.parameters(),lr=lr,momentum=.9,weight_decay=wd)
    elif method=='AdamW':opt=torch.optim.AdamW(task.model.parameters(),lr=lr,weight_decay=wd)
    elif method=='SubTrackOfficial':opt=Official(gs,lr,wd,glr)
    else:
        kw=dict(lr=lr,weight_decay=wd,reset_mode='diagonal',normalize_cov=True)
        if method=='PeriodicSVD':opt=PeriodicSVDRouter(gs,svd_every=25,**kw)
        else:opt=GrassRoute(gs,grassmann_lr=0. if method=='Fixed' else glr,**kw)
    return opt,names


def state_bytes(opt):
    seen=set();objects=set()
    def visit(x):
        if id(x) in objects:return 0
        objects.add(id(x))
        if torch.is_tensor(x):
            key=(str(x.device),x.untyped_storage().data_ptr())
            if key in seen:return 0
            seen.add(key);return x.untyped_storage().nbytes()
        if isinstance(x,dict):return sum(visit(v) for v in x.values())
        if isinstance(x,(tuple,list)):return sum(visit(v) for v in x)
        if hasattr(x,'__dict__'):return visit(vars(x))
        return 0
    return visit(opt.state)


def diagnostics(task,opt):
    # Independent layerwise snapshot. Full covariance eigenvalues, NOT the
    # r x r compressed Gram gap. Radius is to this batch oracle, not population.
    for name,p in task.model.named_parameters():
        st=opt.state.get(p,{})
        if 'U' not in st or p.grad is None:continue
        u=st['U'];g=p.grad.detach().reshape(p.shape[0],-1).float();d=g.shape[1]
        if d>1024:continue
        with torch.no_grad():
            c=g.T@g/(g.square().sum()+1e-12)
            lam,v=torch.linalg.eigh(c);r=u.shape[1]
            oracle=v[:,-r:];s=torch.linalg.svdvals(u.T@oracle).clamp(0,1);angles=s.acos()
            gap=float(lam[-r]-lam[-r-1]) if r<d else None
            captured=float((g@u).square().sum()/(g.square().sum()+1e-30))
            return {'layer':name,'r':r,'d':d,'batch_oracle_gap':gap,
                    'batch_oracle_radius':float(angles.norm()),'batch_oracle_theta_max':float(angles.max()),
                    'captured_energy':captured,'oracle_rank_tail':float(lam[:-r].sum()),
                    'routing_excess':float(lam[-r:].sum())-captured,
                    'reference':'current gradient only; not a population eigenspace estimate'}
    return None


def run(name,method,lr,glr,epochs,folder,formal=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    done=folder/'summary.json'
    if done.exists():return json.loads(done.read_text())
    seed();task=build(name);opt,names=optimizer(task,method,lr,glr)
    # Capture configuration and exact source hashes BEFORE training.
    cfg={'task':name,'method':method,'lr':lr,'grassmann_lr':glr,'epochs':epochs,'seed':42,
         'formal':formal,'algorithm_version':'diagonal-covariance-transport-v1',
         'routed_layers':names,'meta':task.meta,
         'sources':{str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [HERE/'suite.py',HERE/'tasks_extended.py',HERE.parent/'experiments/grassroute.py',HERE.parent/'experiments/baselines.py']}}
    write_json(folder/'config.json',cfg)
    records=[];start=0;elapsed=0.;acc_matrix=[]
    checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location=DEV,weights_only=False)
        task.model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer'])
        start=c['epoch']+1;records=c['records'];elapsed=c['elapsed'];acc_matrix=c.get('acc_matrix',[])
        torch.set_rng_state(c['rng_cpu'].cpu());np.random.set_state(c['rng_numpy']);random.setstate(c['rng_python'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in c['rng_cuda']])
        dls=task.tasks or [task.train]
        for dl,state in zip(dls,c['loader_rng']):
            if hasattr(dl,'generator') and dl.generator is not None and state is not None:dl.generator.set_state(state.cpu())
    begin=time.monotonic()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    for ep in range(start,epochs):
        warmup=max(1,int(.05*epochs)) if task.family=='vit' else 0
        rate=lr*(ep+1)/warmup if warmup and ep<warmup else lr*(.01+.99*.5*(1+math.cos(math.pi*max(0,ep-warmup)/max(1,epochs-warmup))))
        for group in opt.param_groups:
            group['lr']=rate
            if 'lr_perp' in group:group['lr_perp']=rate
        if task.tasks:
            phase=min(len(task.tasks)-1,ep*len(task.tasks)//epochs);train=task.tasks[phase]
        else:phase=0;train=task.train
        task.model.train();loss_total=0.;count=0;gnorm=0.
        for batch in train:
            opt.zero_grad();loss=task.loss(task.model,batch)
            if not torch.isfinite(loss):raise FloatingPointError(f'non-finite loss at epoch {ep}')
            loss.backward()
            gnorm=sum(float(p.grad.detach().square().sum()) for p in task.model.parameters() if p.grad is not None)
            # Optimizer random initialization cannot change subsequent dropout,
            # collocation or negative-sampling streams across methods.
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            opt.step();torch.set_rng_state(cpu_rng)
            if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
            n=len(batch[0]) if batch is not None else 1
            loss_total+=float(loss)*n;count+=n
        rec={'epoch':ep+1,'phase':phase,'train_loss':loss_total/count,'gradient_norm_last_batch':math.sqrt(gnorm),
             'lr':rate,'wall_s':elapsed+time.monotonic()-begin,'state_bytes':state_bytes(opt)}
        boundary=task.tasks and (ep==epochs-1 or (ep+1)*len(task.tasks)//epochs!=phase)
        if ep==0 or (ep+1)%50==0 or ep==epochs-1 or boundary:
            rec['validation']=task.evaluate(task.model,'val')
            rec['diagnostics']=diagnostics(task,opt)
            if boundary:acc_matrix.append(rec['validation'].get('per_task',[]))
        records.append(rec)
        if (ep+1)%50==0 or ep==epochs-1:
            write_json(folder/'trajectory.json',records)
            dls=task.tasks or [task.train]
            c={'model':task.model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,
               'records':records,'elapsed':rec['wall_s'],'acc_matrix':acc_matrix,
               'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),
               'loader_rng':[dl.generator.get_state() if hasattr(dl,'generator') and dl.generator is not None else None for dl in dls]}
            temp=folder/'checkpoint.pt.tmp';torch.save(c,temp);temp.replace(checkpoint)
            print(name,method,ep+1,rec['train_loss'],rec.get('validation'),flush=True)
    result={**cfg,'status':'complete','n_epochs':len(records),'validation':task.evaluate(task.model,'val'),
            'test':task.evaluate(task.model,'test') if formal else None,'final_train_loss':records[-1]['train_loss'],
            'state_bytes':state_bytes(opt),'wall_s':elapsed+time.monotonic()-begin,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            'continual_validation_matrix':acc_matrix}
    if formal and acc_matrix and result['test'].get('per_task'):
        final=result['test']['per_task']
        # Validation forgetting is separate from held-out final test accuracy.
        last=acc_matrix[-1]
        result['validation_forgetting']=float(np.mean([max(row[i] for row in acc_matrix[i:])-last[i] for i in range(len(last)-1)]))
    write_json(done,result);return result


def configs(task_name,method):
    if method=='SGD':lrs=[.01,.1,.3]
    elif task_name.startswith(('sciml_','matreg_','lowrank_','graph_')):lrs=[.001,.003,.01]
    else:lrs=[.0003,.001,.003]
    glr=3. if task_name.startswith('graph_') else .3
    # Official SubTrack uses a raw-gradient geodesic step with iterative decrease
    # in its paper; its scale is distinct from normalized GrassRoute.
    return [(lr,1. if method=='SubTrackOfficial' else glr) for lr in lrs]


def job(task,method):
    folder=OUT/task/method
    if (folder/'formal/summary.json').exists():return
    tuning=[]
    screening=100 if task.startswith(('sciml_','matreg_','lowrank_','graph_')) else 30
    for i,(lr,glr) in enumerate(configs(task,method)):
        try:
            r=run(task,method,lr,glr,screening,folder/f'screen_{i}')
            tuning.append(r)
        except FloatingPointError as ex:
            write_json(folder/f'screen_{i}'/'diverged.json',{'lr':lr,'error':str(ex)})
    if not tuning:raise FloatingPointError('all validation-screen configurations diverged')
    is_high=task.startswith(('graph_','mlp_','cnn_','vit_','recsys_','continual_'))
    best=(max if is_high else min)(tuning,key=lambda r:r['validation']['score'])
    write_json(folder/'selection.json',{'criterion':'final held-out validation score after equal screening budget',
               'selected_lr':best['lr'],'selected_grassmann_lr':best['grassmann_lr'],
               'candidates':[{'lr':r['lr'],'validation':r['validation']} for r in tuning]})
    epochs=2000 if task.startswith(('matreg_','lowrank_','sciml_','mlp_')) else 1000
    run(task,method,best['lr'],best['grassmann_lr'],epochs,folder/'formal',True)


def worker(tasks,methods):
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(t,m) for t in tasks for m in methods]
    for t,m in jobs:
        folder=OUT/t/m;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'formal/summary.json').exists() or (folder/'failed.json').exists():continue
        lock=folder/'worker.lock'
        fd=open(lock,'a+')
        try:fcntl.flock(fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:fd.close();continue
        fd.seek(0);fd.truncate();fd.write(json.dumps({'pid':os.getpid(),'host':os.uname().nodename,'time':time.time()}));fd.flush()
        try:job(t,m)
        except Exception as ex:
            write_json(folder/'failed.json',{'type':type(ex).__name__,'error':str(ex),'traceback':traceback.format_exc()})
            traceback.print_exc()
        finally:
            fcntl.flock(fd.fileno(),fcntl.LOCK_UN);fd.close()
            if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',default=TASKS)
    ap.add_argument('--methods',nargs='+',default=METHODS);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.smoke:
        for t in a.tasks:
            for m in a.methods:
                run(t,m,.001,.3,2,OUT/'smoke'/t/m,False)
    else:worker(a.tasks,a.methods)
