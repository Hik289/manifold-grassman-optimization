from pathlib import Path
import torch, math, json, time
import tasks
from baselines import build_optimizer
from routing import build_param_groups
from trajectory import TrajectoryLogger
import torch.nn.functional as F
DEV='cuda'
def run_partial(epochs=300, glr=0.3):
    torch.manual_seed(42)
    model,ts,ev,loader,meta=tasks.build_cnn()
    groups,names=build_param_groups(model,'cnn',base_lr=3e-3,cnn_partial=True)
    n_routed=sum(len(g['params']) for g in groups if g.get('routed'))
    print(f'partial-routing 路由层数={n_routed} (只 fc+1x1conv)',flush=True)
    opt=build_optimizer('B5',groups,rank=8,lr=3e-3,grassmann_lr=glr)  # fc/1x1 用 r=8 足够
    best=0
    for ep in range(epochs):
        f=0.01+0.99*0.5*(1+math.cos(math.pi*ep/epochs))
        for g in opt.param_groups:
            g['lr']=3e-3*f
            if 'lr_perp' in g: g['lr_perp']=3e-3*f
        model.train()
        for si,(x,y) in enumerate(loader):
            opt.zero_grad(); loss=F.cross_entropy(model(x.to(DEV)),y.to(DEV)); loss.backward(); opt.step()
        if ep%25==0 or ep==epochs-1:
            acc=ev(model)['top1']; best=max(best,acc)
            print(f'ep{ep}: top1={acc:.4f}',flush=True)
    json.dump({'cnn_B5_partial_top1':best},open(Path(__file__).resolve().parent / 'runs' / 'cnn_partial_verdict.json','w'))
    print('PARTIAL done best=',best,flush=True)
run_partial()
