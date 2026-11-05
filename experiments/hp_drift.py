from pathlib import Path
import train, json
res={}
print('=== continual (RotMNIST, 大漂移) B4 vs B5 vs B2 ===', flush=True)
for b,hp in [('B2',{}),('B4',{'rank':16}),('B5',{'rank':16,'grassmann_lr':0.3}),('B5',{'rank':16,'grassmann_lr':1.0})]:
    lr=1e-2
    s=train.train_one('continual',b,epochs=200,lr=lr,eval_every=199,out_dir='/tmp/hp',**hp)
    tag=f'{b}_glr{hp.get("grassmann_lr","-")}'
    res[f'continual_{tag}']=s['final_metrics']; print(f'continual {tag}: {s["final_metrics"]} loss={s["final_train_loss"]:.4f}',flush=True)
print('=== Cora GCN B5 vs B4 vs B2 (中漂移) ===', flush=True)
for b,glr in [('B2',None),('B4',None),('B5',0.3),('B5',1.0)]:
    hp={'rank':8}
    if glr: hp['grassmann_lr']=glr
    import tasks
    # 用 Cora 需改 build_graph dataset; 直接调
    m,ts,ev,ld,meta=tasks.build_graph('Cora')
    # 手动跑 (train_one 固定 ogbn); 简化: 复用 train_one 需 dataset 参数, 这里直接内联
    import torch
    from baselines import build_optimizer
    from routing import build_param_groups
    torch.manual_seed(42)
    m,ts,ev,ld,meta=tasks.build_graph('Cora')
    if b in ('B1','B2'):
        opt=build_optimizer(b,m.parameters(),lr=1e-2)
    else:
        g,_=build_param_groups(m,'graph',base_lr=1e-2)
        opt=build_optimizer(b,g,rank=8,lr=1e-2,grassmann_lr=glr or 0.3)
    import math
    for ep in range(1000):
        f=0.01+0.99*0.5*(1+math.cos(math.pi*ep/1000))
        for gg in opt.param_groups:
            gg['lr']=1e-2*f
            if 'lr_perp' in gg: gg['lr_perp']=1e-2*f
        opt.zero_grad(); loss=ts(m,None); loss.backward(); opt.step()
    met=ev(m); res[f'Cora_{b}_glr{glr}']=met; print(f'Cora {b} glr={glr}: {met}',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'hp_drift.json','w'),indent=1); print('DONE',flush=True)
