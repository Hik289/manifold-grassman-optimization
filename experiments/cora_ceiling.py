from pathlib import Path
import torch,math,json,tasks
from baselines import build_optimizer
from routing import build_param_groups
res={}
def run(b,glr=None,lr=1e-2):
    torch.manual_seed(42)
    m,ts,ev,ld,meta=tasks.build_graph('Cora')
    if b in ('B2',):
        opt=build_optimizer(b,m.parameters(),lr=lr)
    else:
        g,_=build_param_groups(m,'graph',base_lr=lr); opt=build_optimizer(b,g,rank=8,lr=lr,grassmann_lr=glr or 0.3)
    for ep in range(1000):
        f=0.01+0.99*0.5*(1+math.cos(math.pi*ep/1000))
        for gg in opt.param_groups:
            gg['lr']=lr*f
            if 'lr_perp' in gg: gg['lr_perp']=lr*f
        opt.zero_grad(); loss=ts(m,None); loss.backward(); opt.step()
    return ev(m)['test']
for glr in [10,20,50]:
    res[f'B5_glr{glr}']=run('B5',glr); print(f'Cora B5 glr={glr}: {res[f"B5_glr{glr}"]:.4f}',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'cora_ceiling.json','w'),indent=1); print('DONE',flush=True)
