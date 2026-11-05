from pathlib import Path
import train, json, time
res={}
print('=== RecSys MF B1-B6 全公平 (fixed MF, 1000ep) ===', flush=True)
# per-optimizer lr: B1 SGD 高, 其余低
lrmap={'B1':[0.03,0.05,0.1],'B2':[0.01,0.03],'B3':[0.01,0.03],'B4':[0.01,0.03],'B5':[0.01,0.03],'B6':[0.01,0.03]}
for b in ['B1','B2','B3','B4','B5','B6']:
    best=None
    for lr in lrmap[b]:
        kw={'rank':16} if b not in ('B1','B2') else {}
        if b in ('B3','B4','B5','B6'): kw['grassmann_lr']=0.3
        s=train.train_one('recsys',b,epochs=1000,lr=lr,out_dir='/tmp/recfull',**kw)
        m=s['final_metrics']
        cur=(m.get('recall@20') or 0, m.get('ndcg@20'), m.get('rmse'), lr)
        if best is None or cur[0]>best[0]: best=cur
        print(f'{b} lr={lr}: recall@20={m.get("recall@20"):.4f} ndcg={m.get("ndcg@20"):.4f} rmse={m.get("rmse"):.3f}',flush=True)
    res[b]={'best_recall@20':best[0],'ndcg@20':best[1],'rmse':best[2],'best_lr':best[3]}
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'recsys_full.json','w'),indent=1); print('DONE',flush=True)
