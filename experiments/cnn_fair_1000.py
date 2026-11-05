from pathlib import Path
import train, json, time, sys
which=sys.argv[1]
if which=='sgd':
    t0=time.time()
    s=train.train_one('cnn','B1',epochs=1000,lr=0.1,out_dir='/tmp/cnnfair')
    import json as J; d=J.load(open('/tmp/cnnfair/traj_cnn_B1_r32.json'))['records']
    v=[r['val_metric'] for r in d if r.get('val_metric') is not None]
    json.dump({'cnn_B1_sgd_lr0.1_1000ep':max(v) if v else None},open(Path(__file__).resolve().parent / 'runs' / 'cnn_sgd1000.json','w'))
    print(f'CNN B1 SGD lr=0.1 1000ep: {max(v) if v else None} ({time.time()-t0:.0f}s)',flush=True)
elif which=='partial':
    import torch,math,tasks
    from baselines import build_optimizer
    from routing import build_param_groups
    import torch.nn.functional as F
    torch.manual_seed(42)
    m,ts,ev,loader,meta=tasks.build_cnn()
    g,_=build_param_groups(m,'cnn',base_lr=3e-3,cnn_partial=True)
    opt=build_optimizer('B5',g,rank=8,lr=3e-3,grassmann_lr=0.3)
    best=0
    for ep in range(1000):
        f=0.01+0.99*0.5*(1+math.cos(math.pi*ep/1000))
        for gg in opt.param_groups:
            gg['lr']=3e-3*f
            if 'lr_perp' in gg: gg['lr_perp']=3e-3*f
        m.train()
        for x,y in loader:
            opt.zero_grad(); F.cross_entropy(m(x.cuda()),y.cuda()).backward(); opt.step()
        if ep%25==0 or ep==999:
            best=max(best,ev(m)['top1'])
    mem=opt.state_memory_bytes()
    json.dump({'cnn_partial_B5_1000ep':best,'state_mem_MB':mem/1e6},open(Path(__file__).resolve().parent / 'runs' / 'cnn_partial1000.json','w'))
    print(f'CNN partial-B5 1000ep: {best} mem={mem/1e6:.1f}MB',flush=True)
