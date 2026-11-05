from pathlib import Path
import train, json
res={}
print('=== graph B5 glr 天花板 (推到饱和/发散) ===', flush=True)
for glr in [3.0,5.0,10.0,20.0,50.0]:
    try:
        s=train.train_one('graph','B5',epochs=1000,rank=8,lr=1e-2,grassmann_lr=glr,out_dir='/tmp/glr')
        t=s['final_metrics']['test']; res[f'glr{glr}']=t
        print(f'glr={glr}: test={t:.4f} loss={s["final_train_loss"]:.4f}',flush=True)
    except Exception as e:
        res[f'glr{glr}']='DIVERGE'; print(f'glr={glr}: DIVERGE {type(e).__name__}',flush=True)
# 也试 Cora GCN 更大 glr (中漂移可能翻盘)
print('=== Cora GCN B5 glr 天花板 ===', flush=True)
import torch,math,tasks
from baselines import build_optimizer
from routing import build_param_groups
for glr in [1.0,3.0,10.0]:
    torch.manual_seed(42)
    m,ts,ev,ld,meta=tasks.build_graph('Cora')
    g,_=build_param_groups(m,'graph',base_lr=1e-2)
    opt=build_optimizer('B5',g,rank=8,lr=1e-2,grassmann_lr=glr)
    for ep in range(1000):
        f=0.01+0.99*0.5*(1+math.cos(math.pi*ep/1000))
        for gg in opt.param_groups:
            gg['lr']=1e-2*f
            if 'lr_perp' in gg: gg['lr_perp']=1e-2*f
        opt.zero_grad(); loss=ts(m,None); loss.backward(); opt.step()
    met=ev(m); res[f'cora_glr{glr}']=met['test']; print(f'Cora B5 glr={glr}: test={met["test"]:.4f}',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'glr_ceiling.json','w'),indent=1); print('DONE',flush=True)
