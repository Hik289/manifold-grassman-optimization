from pathlib import Path
import train, json, time
res={}
print('=== RecSys B5 意外胜 — INTERVENE 机制隔离 (1000ep, lr=0.01) ===', flush=True)
# 1. 残差分支作用: A4 无残差 vs 正常 B5
configs=[
  ('B5_full',{'baseline':'B5','rank':16,'grassmann_lr':0.3}),
  ('B5_no_residual',{'baseline':'B5','rank':16,'grassmann_lr':0.3,'extra':{'use_residual':0}}),
  ('B5_main_sgd',{'baseline':'B5','rank':16,'grassmann_lr':0.3,'extra':{'main_sgd':1}}),
  ('B5_rank4',{'baseline':'B5','rank':4,'grassmann_lr':0.3}),
  ('B5_rank32',{'baseline':'B5','rank':32,'grassmann_lr':0.3}),
  ('B1_sgd',{'baseline':'B1'}),
]
import os
for name,cfg in configs:
    extra=cfg.pop('extra',None)
    if extra: os.environ['GR_EXTRA']=json.dumps(extra)
    elif 'GR_EXTRA' in os.environ: del os.environ['GR_EXTRA']
    b=cfg.pop('baseline')
    s=train.train_one('recsys',b,epochs=1000,lr=0.01,out_dir='/tmp/recprobe',**cfg)
    m=s['final_metrics']
    res[name]={'recall@20':m.get('recall@20'),'ndcg@20':m.get('ndcg@20'),'rmse':m.get('rmse')}
    print(f'{name}: recall@20={m.get("recall@20"):.4f} ndcg={m.get("ndcg@20"):.4f} rmse={m.get("rmse"):.3f}',flush=True)
if 'GR_EXTRA' in os.environ: del os.environ['GR_EXTRA']
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'recsys_probe.json','w'),indent=1); print('DONE',flush=True)
