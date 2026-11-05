from pathlib import Path
import train, json, time
res={}
print('=== CNN full-routing B5 glr 确认 (r8 快, 150ep) ===', flush=True)
for glr in [0.3,3.0,10.0]:
    t0=time.time()
    s=train.train_one('cnn','B5',epochs=150,rank=8,lr=3e-3,grassmann_lr=glr,out_dir='/tmp/cglrf')
    import json as J
    d=J.load(open('/tmp/cglrf/traj_cnn_B5_r8.json'))['records']
    v=[r['val_metric'] for r in d if r.get('val_metric') is not None]
    res[f'glr{glr}']=max(v) if v else None
    print(f'glr={glr}: top1={res[f"glr{glr}"]:.4f} ({time.time()-t0:.0f}s)',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'cnn_glr_fast.json','w'),indent=1); print('DONE',flush=True)
