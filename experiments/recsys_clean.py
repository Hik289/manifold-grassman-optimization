from pathlib import Path
import train, json, time
res={}
print('=== RecSys MF 1000ep 干净负面对照 (各自最优lr) ===', flush=True)
for b,rank in [('B2',None),('B4',16),('B5',16)]:
    best=None
    for lr in [0.01,0.03,0.05]:
        kw={'rank':rank} if rank else {}
        if b=='B5': kw['grassmann_lr']=0.3
        s=train.train_one('recsys',b,epochs=1000,lr=lr,out_dir='/tmp/recclean',**kw)
        m=s['final_metrics']
        cur=(m.get('recall@20'),m.get('ndcg@20'),m.get('rmse'),lr)
        if best is None or (cur[0] or 0)>(best[0] or 0): best=cur
        print(f'{b} lr={lr}: recall@20={m.get("recall@20"):.4f} ndcg={m.get("ndcg@20"):.4f} rmse={m.get("rmse"):.3f}',flush=True)
    res[b]={'best_recall@20':best[0],'best_ndcg@20':best[1],'rmse':best[2],'best_lr':best[3]}
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'recsys_clean.json','w'),indent=1); print('DONE',flush=True)
