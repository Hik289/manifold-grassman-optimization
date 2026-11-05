from pathlib import Path
import train, json
res={}
print('=== sciml B5 lr sweep (2000ep) ===', flush=True)
for lr,glr in [(3e-3,0.1),(1e-2,0.3),(1e-2,1.0),(3e-3,1.0),(1e-3,0.3)]:
    s=train.train_one('sciml','B5',epochs=2000,rank=8,lr=lr,grassmann_lr=glr,eval_every=1999,out_dir='/tmp/hp')
    m=s['final_metrics']; res[f'B5_sciml_lr{lr}_glr{glr}']=s['best_train_loss']
    print(f'B5 lr={lr} glr={glr}: best_loss={s["best_train_loss"]:.4e} rel_l2={m.get("rel_l2_proxy")}', flush=True)
print('=== sciml B4 lr sweep ===', flush=True)
for lr in [3e-3,1e-2,3e-2]:
    s=train.train_one('sciml','B4',epochs=2000,rank=8,lr=lr,eval_every=1999,out_dir='/tmp/hp')
    res[f'B4_sciml_lr{lr}']=s['best_train_loss']
    print(f'B4 lr={lr}: best_loss={s["best_train_loss"]:.4e}', flush=True)
print('=== graph B5 glr 拐点 (lr=1e-2) ===', flush=True)
for glr in [1.0,2.0,3.0]:
    s=train.train_one('graph','B5',epochs=1000,rank=8,lr=1e-2,grassmann_lr=glr,eval_every=999,out_dir='/tmp/hp')
    res[f'B5_graph_glr{glr}']=s['final_metrics']['test']
    print(f'graph B5 glr={glr}: test={s["final_metrics"]["test"]:.4f}', flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'hp_probe3.json','w'),indent=1)
print('DONE', flush=True)
