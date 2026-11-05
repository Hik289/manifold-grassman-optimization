from pathlib import Path
import train, json, time
res={}
print('=== SciML B4 周期SVD 全 lr sweep (2000ep) ===', flush=True)
for lr in [3e-3,1e-2,3e-2]:
    t0=time.time()
    s=train.train_one('sciml','B4',epochs=2000,rank=8,lr=lr,out_dir='/tmp/scimlg2')
    res[f'B4_lr{lr}']=s['best_train_loss']
    print(f'B4 lr={lr}: best_loss={s["best_train_loss"]:.4e} ({time.time()-t0:.0f}s)',flush=True)
print('=== SciML B5 GrassRoute 确认最优 (glr sweep) ===', flush=True)
for lr,glr in [(1e-2,0.3),(1e-2,1.0),(3e-2,0.3),(1e-2,0.1)]:
    s=train.train_one('sciml','B5',epochs=2000,rank=8,lr=lr,grassmann_lr=glr,out_dir='/tmp/scimlg2')
    res[f'B5_lr{lr}_glr{glr}']=s['best_train_loss']
    print(f'B5 lr={lr} glr={glr}: best_loss={s["best_train_loss"]:.4e}',flush=True)
# 也扫 B4 更细 rank (周期SVD 可能不同 rank 更优)
print('=== SciML B4 rank sweep (lr 最优处) ===', flush=True)
for r in [4,16,32]:
    s=train.train_one('sciml','B4',epochs=2000,rank=r,lr=1e-2,out_dir='/tmp/scimlg2')
    res[f'B4_r{r}_lr0.01']=s['best_train_loss']
    print(f'B4 r={r} lr=0.01: best_loss={s["best_train_loss"]:.4e}',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'sciml_g2.json','w'),indent=1); print('DONE',flush=True)
