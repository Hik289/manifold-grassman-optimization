from pathlib import Path
import train,json
s=train.train_one('cnn','B5',epochs=300,rank=32,lr=3e-3,grassmann_lr=0.3,out_dir='/tmp/cnnv')
v=[r['val_metric'] for r in json.load(open('/tmp/cnnv/traj_cnn_B5_r32.json'))['records'] if r.get('val_metric') is not None]
json.dump({'cnn_B5_top1':max(v) if v else None},open(Path(__file__).resolve().parent / 'runs' / 'cnn_B5_verdict.json','w'))
print('B5 done',max(v) if v else None)
