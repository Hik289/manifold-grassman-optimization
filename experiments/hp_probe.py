from pathlib import Path
import train, json
res={}
print('=== graph B5 HP (1000ep) ===', flush=True)
for lr,glr in [(3e-3,0.1),(1e-2,0.1),(1e-2,0.3),(3e-2,0.3),(1e-2,1.0),(3e-2,1.0)]:
    s=train.train_one('graph','B5',epochs=1000,rank=8,lr=lr,grassmann_lr=glr,eval_every=999,out_dir='/tmp/hp')
    t=s['final_metrics']['test']; res[f'B5_lr{lr}_glr{glr}']=t
    print(f'lr={lr} glr={glr}: test={t:.4f}', flush=True)
print('=== graph B2 AdamW HP ===', flush=True)
for lr in [3e-3,1e-2,3e-2]:
    s=train.train_one('graph','B2',epochs=1000,lr=lr,eval_every=999,out_dir='/tmp/hp')
    t=s['final_metrics']['test']; res[f'B2_lr{lr}']=t
    print(f'lr={lr}: test={t:.4f}', flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'hp_probe_graph.json','w'),indent=1)
print('DONE', flush=True)
