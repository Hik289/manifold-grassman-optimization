from pathlib import Path
import train, json
res={}
print('=== CNN CIFAR-100 快速 G2 探针 (150ep, r32) ===', flush=True)
for b,hp in [('B2',{}),('B4',{'rank':32}),('B5',{'rank':32,'grassmann_lr':0.3}),('B5',{'rank':32,'grassmann_lr':1.0})]:
    s=train.train_one('cnn',b,epochs=150,lr=3e-3,eval_every=149,out_dir='/tmp/cnnprobe',**hp)
    tag=f'{b}_glr{hp.get("grassmann_lr","-")}'
    res[tag]=s['final_metrics']; print(f'{tag}: top1={s["final_metrics"].get("top1"):.4f} loss={s["final_train_loss"]:.4f}',flush=True)
json.dump(res,open(Path(__file__).resolve().parent / 'runs' / 'cnn_probe.json','w'),indent=1); print('DONE',flush=True)
