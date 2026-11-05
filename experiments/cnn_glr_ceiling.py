from pathlib import Path
import train, json, time
res={}
print('=== CNN full-routing B5 glr 天花板 (200ep r32) ===', flush=True)
for glr in [0.3,1.0,3.0,10.0]:
    t0=time.time()
    s=train.train_one('cnn','B5',epochs=200,rank=32,lr=3e-3,grassmann_lr=glr,out_dir='/tmp/cnnglr')
    import json as J
    d=J.load(open('/tmp/cnnglr/traj_cnn_B5_r32.json'))['records']
    v=[r['val_metric'] for r in d if r.get('val_metric') is not None]
    best=max(v) if v else None
    res[f'glr{glr}']=best
    print(f'glr={glr}: best_top1={best:.4f} ({time.time()-t0:.0f}s)',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'cnn_glr_ceiling.json','w'),indent=1); print('DONE',flush=True)
