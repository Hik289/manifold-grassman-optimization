from pathlib import Path
import train, json, time
res={}
print('=== RecSys MF 负面对照 (embedding近满秩, 预期B5≤AdamW) ===', flush=True)
for b,hp in [('B2',{}),('B1',{}),('B4',{'rank':16}),('B5',{'rank':16,'grassmann_lr':0.3})]:
    for lr in ([0.01,0.03] if b in ('B1','B2','B4','B5') else [0.01]):
        t0=time.time()
        kw=dict(hp);
        s=train.train_one('recsys',b,epochs=300,lr=lr,out_dir='/tmp/recneg',**kw)
        m=s['final_metrics']
        tag=f'{b}_lr{lr}'
        res[tag]={'rmse':m.get('rmse'),'recall@20':m.get('recall@20'),'ndcg@20':m.get('ndcg@20')}
        print(f'{tag}: rmse={m.get("rmse"):.4f} recall@20={m.get("recall@20")} ndcg@20={m.get("ndcg@20")} ({time.time()-t0:.0f}s)',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'recsys_neg.json','w'),indent=1); print('DONE',flush=True)
