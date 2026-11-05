from pathlib import Path
import train, json
res={}
print('=== B6 sciml + graph B5 glr 拐点 ===', flush=True)
for lr,glr in [(1e-2,0.3),(1e-2,1.0)]:
    s=train.train_one('sciml','B6',epochs=2000,rank=8,lr=lr,grassmann_lr=glr,eval_every=1999,out_dir='/tmp/hp')
    res[f'B6_sciml_lr{lr}_glr{glr}']=s['best_train_loss']; print(f'B6 sciml lr={lr} glr={glr}: best={s["best_train_loss"]:.4e}',flush=True)
for glr in [1.0,2.0,3.0]:
    s=train.train_one('graph','B5',epochs=1000,rank=8,lr=1e-2,grassmann_lr=glr,eval_every=999,out_dir='/tmp/hp')
    res[f'B5_graph_glr{glr}']=s['final_metrics']['test']; print(f'graph B5 glr={glr}: test={s["final_metrics"]["test"]:.4f}',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'hp_probe5.json','w'),indent=1); print('DONE',flush=True)
