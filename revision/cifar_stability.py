"""Neighboring-data CIFAR-100 linear-probe sensitivity, 1000 full epochs.

Frozen random ResNet features are an auxiliary controlled model. This is NOT
the end-to-end ResNet benchmark and cannot be counted as an extra family win.
"""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms as T

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'experiments'))
from grassroute import GrassRoute
from tasks_extended import ROOT,DEV,EXTRA

torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
EXTRA.mkdir(parents=True,exist_ok=True)
cache=EXTRA/'cifar100_random_resnet_features_seed42.pt'
if not cache.exists():
    tf=T.Compose([T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
    tr=tv.datasets.CIFAR100(str(ROOT),train=True,download=True,transform=tf)
    te=tv.datasets.CIFAR100(str(ROOT),train=False,download=False,transform=tf)
    net=tv.models.resnet18(num_classes=100);net.conv1=nn.Conv2d(3,64,3,1,1,bias=False);net.maxpool=nn.Identity();net.fc=nn.Identity();net.to(DEV).eval()
    def extract(ds,ids):
        dl=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=256,num_workers=0)
        xx=[];yy=[]
        with torch.no_grad():
            for x,y in dl:xx.append(net(x.to(DEV)).cpu());yy.append(y)
        return torch.cat(xx),torch.cat(yy)
    ids=torch.randperm(len(tr),generator=torch.Generator().manual_seed(42))[:2561].tolist()
    x,y=extract(tr,ids);xt,yt=extract(te,list(range(1000)))
    mean=x[:2048].mean(0);std=x[:2048].std(0).clamp_min(.01)
    torch.save({'x':(x-mean)/std,'y':y,'xt':(xt-mean)/std,'yt':yt,'train_indices':ids,'feature_model':'untrained frozen ResNet-18, seed42'},cache)
d=torch.load(cache,weights_only=False);x=d['x'].to(DEV);y=d['y'].to(DEV);xt=d['xt'].to(DEV);yt=d['yt'].to(DEV)
xx=x[:2048];yy=y[:2048];xp=xx.clone();yp=yy.clone();xp[0]=x[2048];yp[0]=y[2048]
out={}
for method in ['AdamW','GrassRoute','HardReset','Fixed']:
    torch.manual_seed(42)
    a=nn.Linear(512,100).to(DEV);b=nn.Linear(512,100).to(DEV);b.load_state_dict(a.state_dict())
    def opt(model):
        if method=='AdamW':return torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        gs=[{'params':[model.weight],'routed':True,'rank':16},{'params':[model.bias],'routed':False}]
        return GrassRoute(gs,lr=.003,weight_decay=.01,grassmann_lr=0. if method=='Fixed' else .3,
                          reset_mode='hard' if method=='HardReset' else 'diagonal')
    oa,ob=opt(a),opt(b);records=[];start=time.time()
    for ep in range(1000):
        lr=.003*(.01+.99*.5*(1+math.cos(math.pi*ep/1000)))
        for optimizer in (oa,ob):
            for g in optimizer.param_groups:
                g['lr']=lr
                if 'lr_perp' in g:g['lr_perp']=lr
        for model,optimizer,features,labels in [(a,oa,xx,yy),(b,ob,xp,yp)]:
            optimizer.zero_grad();loss=F.cross_entropy(model(features),labels);loss.backward()
            # Same initial routing basis for each coupled run.
            state=torch.get_rng_state();states=torch.cuda.get_rng_state_all()
            torch.manual_seed(4242);optimizer.step();torch.set_rng_state(state);torch.cuda.set_rng_state_all(states)
        rec={'epoch':ep+1,'train_loss':float(F.cross_entropy(a(xx),yy)),
             'parameter_distance':math.sqrt(sum(float((p-q).square().sum()) for p,q in zip(a.parameters(),b.parameters())))}
        if ep%25==0 or ep==999:
            with torch.no_grad():
                la=F.cross_entropy(a(xt),yt,reduction='none').clamp(max=10)/10
                lb=F.cross_entropy(b(xt),yt,reduction='none').clamp(max=10)/10
                rec.update(test_accuracy=float((a(xt).argmax(1)==yt).float().mean()),
                           mean_bounded_loss_sensitivity=float((la-lb).abs().mean()),
                           max_bounded_loss_sensitivity=float((la-lb).abs().max()))
                st=oa.state.get(a.weight,{})
                if 'U' in st:
                    g=a.weight.grad;u=st['U'];lam,v=torch.linalg.eigh(g.T@g)
                    th=torch.linalg.svdvals(u.T@v[:,-16:]).clamp(0,1).acos()
                    rec.update(full_batch_oracle_gap=float(lam[-16]-lam[-17]),
                               full_batch_oracle_radius=float(th.norm()),
                               captured_energy=float((g@u).square().sum()/g.square().sum()))
        records.append(rec)
    out[method]={'epochs':1000,'records':records,'wall_s':time.time()-start,'n_train':2048,'n_test':1000}
    (HERE/'cifar_stability_results.json').write_text(json.dumps({'seed':42,'protocol':'frozen random ResNet-18 features; neighboring sets replace one image; full-batch linear classifier; fixed hyperparameters', 'results':out},indent=2))
    print(method,records[-1],flush=True)
