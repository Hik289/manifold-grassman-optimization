from pathlib import Path
import train, json
res={}
print('=== B4 Periodic-SVD HP (1000ep) ===', flush=True)
for lr in [3e-3,1e-2,3e-2]:
    s=train.train_one('graph','B4',epochs=1000,rank=8,lr=lr,eval_every=999,out_dir='/tmp/hp')
    t=s['final_metrics']['test']; res[f'B4_lr{lr}']=t; print(f'B4 lr={lr}: test={t:.4f}', flush=True)
print('=== B6 SubTrack++ HP ===', flush=True)
for lr,glr in [(1e-2,0.3),(1e-2,1.0),(3e-2,1.0)]:
    s=train.train_one('graph','B6',epochs=1000,rank=8,lr=lr,grassmann_lr=glr,eval_every=999,out_dir='/tmp/hp')
    t=s['final_metrics']['test']; res[f'B6_lr{lr}_glr{glr}']=t; print(f'B6 lr={lr} glr={glr}: test={t:.4f}', flush=True)
print('=== B3 fixed-random HP ===', flush=True)
for lr in [1e-2,3e-2]:
    s=train.train_one('graph','B3',epochs=1000,rank=8,lr=lr,eval_every=999,out_dir='/tmp/hp')
    t=s['final_metrics']['test']; res[f'B3_lr{lr}']=t; print(f'B3 lr={lr}: test={t:.4f}', flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'hp_probe_graph2.json','w'),indent=1)
print('DONE', flush=True)
