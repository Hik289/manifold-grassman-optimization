from pathlib import Path
import train, json, time
res={}
print('=== CNN CIFAR-100 G2 verdict (300ep r32, informed glr) ===', flush=True)
for b,hp in [('B2',{}),('B4',{'rank':32}),('B5',{'rank':32,'grassmann_lr':0.3}),('B5',{'rank':32,'grassmann_lr':1.0})]:
    t0=time.time()
    s=train.train_one('cnn',b,epochs=300,lr=3e-3,out_dir='/tmp/cnnv',**hp)
    tag=f'{b}_glr{hp.get("grassmann_lr","-")}'
    m=s['final_metrics']; res[tag]={'top1':m.get('top1'),'best_test':s.get('best_test')}
    print(f'{tag}: top1={m.get("top1"):.4f} loss={s["final_train_loss"]:.4f} ({time.time()-t0:.0f}s)',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'cnn_verdict.json','w'),indent=1); print('DONE',flush=True)
