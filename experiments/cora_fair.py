from pathlib import Path
import torch,math,json,tasks
from baselines import build_optimizer
from routing import build_param_groups
res={}
def run(b,glr=None,lr=1e-2):
    torch.manual_seed(42)
    m,ts,ev,ld,meta=tasks.build_graph('Cora')
    if b in ('B1','B2'):
        opt=build_optimizer(b,m.parameters(),lr=lr)
    else:
        g,_=build_param_groups(m,'graph',base_lr=lr)
        opt=build_optimizer(b,g,rank=8,lr=lr,grassmann_lr=glr or 0.3)
    for ep in range(1000):
        f=0.01+0.99*0.5*(1+math.cos(math.pi*ep/1000))
        for gg in opt.param_groups:
            gg['lr']=lr*f
            if 'lr_perp' in gg: gg['lr_perp']=lr*f
        opt.zero_grad(); loss=ts(m,None); loss.backward(); opt.step()
    return ev(m)['test']
print('Cora 公平天花板:',flush=True)
for b,glr in [('B2',None),('B4',None),('B4',None),('B5',10.0),('B5',20.0),('B6',10.0)]:
    pass
res['B2']=run('B2'); print('B2 AdamW:',res['B2'],flush=True)
for lr in [1e-2,3e-2]:
    res[f'B4_lr{lr}']=run('B4',lr=lr); print(f'B4 lr={lr}:',res[f'B4_lr{lr}'],flush=True)
res['B5_glr10']=run('B5',10.0); print('B5 glr=10:',res['B5_glr10'],flush=True)
res['B5_glr20']=run('B5',20.0); print('B5 glr=20:',res['B5_glr20'],flush=True)
res['B6_glr10']=run('B6',10.0); print('B6 glr=10:',res['B6_glr10'],flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'cora_fair.json','w'),indent=1); print('DONE',flush=True)
