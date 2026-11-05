from pathlib import Path
import train, json, time
# 短 epoch: 快族(graph/sciml/mlp/recsys)用中等ep, 慢族(cnn/vit)用短ep看趋势
EP={'graph':500,'sciml':500,'mlp':200,'recsys':300,'continual':100,'cnn':100,'vit':100}
res={}
for fam in ['graph','sciml','mlp','cnn','vit','recsys','continual']:
    res[fam]={}
    for lr in [0.01,0.03,0.1]:
        try:
            t0=time.time()
            s=train.train_one(fam,'B1',epochs=EP[fam],lr=lr,out_dir='/tmp/b1sweep')
            m=s['final_metrics']
            metric=m.get('top1') or m.get('acc') or m.get('val') or m.get('test') or (-s['best_train_loss'])
            # 统一取一个可比标量
            key=[k for k in ['top1','acc','val','test','rmse','rel_l2_proxy'] if k in m]
            val=m.get(key[0]) if key else s['best_train_loss']
            res[fam][f'lr{lr}']=val
            print(f'{fam} B1 lr={lr}: {key[0] if key else "best_loss"}={val} ({time.time()-t0:.0f}s ep={EP[fam]})',flush=True)
        except Exception as e:
            res[fam][f'lr{lr}']=f'ERR:{type(e).__name__}'; print(f'{fam} lr={lr}: ERR {e}',flush=True)
    json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'b1_sgd_sweep.json','w'),indent=1)
print('DONE',flush=True)
