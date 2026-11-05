import train,json
s=train.train_one('vit','B2',epochs=80,lr=1e-3,weight_decay=0.05,out_dir='/tmp/vitfix')
d=json.load(open('/tmp/vitfix/traj_vit_B2_r16.json'))['records']
v=[r['val_metric'] for r in d if r.get('val_metric') is not None]
print(f'ViT B2 (lr=1e-3+warmup+wd) 80ep: top1={max(v) if v else None}, final_loss={s["final_train_loss"]:.3f}')
