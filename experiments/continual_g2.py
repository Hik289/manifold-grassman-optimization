from pathlib import Path
import continual_run as C
import json, time
res={}
print('=== Continual RotMNIST G2: B4 全lr sweep vs B5 vs B2 (突变漂移) ===', flush=True)
# B4 周期SVD 全 lr 公平扫
for lr in [0.003,0.01,0.03]:
    t0=time.time()
    r=C.run('B4', epochs_per_task=15, steps_per_epoch=100, lr=lr, rank=16)
    res[f'B4_lr{lr}']={'avg_acc':r['avg_acc'],'forgetting':r['forgetting'],'drift':r['mean_subspace_drift']}
    print(f'B4 lr={lr}: avg_acc={r["avg_acc"]:.4f} forget={r["forgetting"]:.4f} drift={r["mean_subspace_drift"]} ({time.time()-t0:.0f}s)',flush=True)
# B5 GrassRoute 扫 lr/glr
for lr,glr in [(0.01,0.3),(0.01,1.0),(0.03,1.0),(0.03,0.3)]:
    r=C.run('B5', epochs_per_task=15, steps_per_epoch=100, lr=lr, glr=glr, rank=16)
    res[f'B5_lr{lr}_glr{glr}']={'avg_acc':r['avg_acc'],'forgetting':r['forgetting'],'drift':r['mean_subspace_drift']}
    print(f'B5 lr={lr} glr={glr}: avg_acc={r["avg_acc"]:.4f} forget={r["forgetting"]:.4f} drift={r["mean_subspace_drift"]}',flush=True)
# B2 AdamW 参考
for lr in [0.001,0.003]:
    r=C.run('B2', epochs_per_task=15, steps_per_epoch=100, lr=lr, rank=16)
    res[f'B2_lr{lr}']={'avg_acc':r['avg_acc'],'forgetting':r['forgetting']}
    print(f'B2 lr={lr}: avg_acc={r["avg_acc"]:.4f} forget={r["forgetting"]:.4f}',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'continual_g2.json','w'),indent=1); print('DONE',flush=True)
