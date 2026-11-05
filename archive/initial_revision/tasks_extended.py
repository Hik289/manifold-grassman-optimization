"""Representative models for every dataset/task named in proposal section 6.

No epoch truncation. Public data are cached on the original HPC, never replaced
by synthetic samples. Manufactured PDEs and generated operator data are labelled.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

ROOT = Path(os.environ.get('GR_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data'))
EXTRA = ROOT / 'codex_extension_20261101'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def download(url, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, 'wb') as f:
            import shutil
            shutil.copyfileobj(response, f)
        tmp.replace(dst)
    return dst


def loader(ds, shuffle=False, batch=256):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=False, generator=torch.Generator().manual_seed(42))


def split_indices(n, fraction=.1):
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    nv = max(1, int(fraction*n))
    return idx[nv:], idx[:nv]


class Task:
    def __init__(self, name, family, model, train, loss, evaluate, meta, epochs,
                 higher=False, tasks=None):
        self.name, self.family = name, family
        self.model = model.to(DEV)
        self.train, self.loss, self.evaluate = train, loss, evaluate
        self.meta, self.epochs, self.higher, self.tasks = meta, epochs, higher, tasks


def classification_eval(model, dl):
    model.eval()
    total = correct = 0
    loss = 0.
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            out = model(x)
            correct += (out.argmax(1)==y).sum().item()
            loss += F.cross_entropy(out, y, reduction='sum').item()
            total += len(y)
    return {'score':correct/total, 'loss':loss/total, 'n':total}


def vision(name):
    import torchvision as tv
    from torchvision import transforms as T
    import timm
    family, data = name.split('_',1)
    size = 32 if data in ('cifar10','cifar100') else 64
    # ImageNet-100 is explicitly evaluated at 64px in this extension.
    tftr = T.Compose([T.Resize((size,size)), T.RandomCrop(size,padding=4),
                      T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
    tfte = T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
    if data in ('cifar10','cifar100'):
        cls = tv.datasets.CIFAR10 if data=='cifar10' else tv.datasets.CIFAR100
        tr = cls(str(ROOT),train=True,download=True,transform=tftr)
        clean = cls(str(ROOT),train=True,download=False,transform=tfte)
        te = cls(str(ROOT),train=False,download=False,transform=tfte)
        nc = 10 if data=='cifar10' else 100
    elif data=='tinyimagenet':
        base = EXTRA/'tiny-imagenet-200'
        if not base.exists():
            import zipfile
            p=download('http://cs231n.stanford.edu/tiny-imagenet-200.zip',EXTRA/'tiny-imagenet-200.zip')
            with zipfile.ZipFile(p) as z: z.extractall(EXTRA)
        tr=tv.datasets.ImageFolder(base/'train',tftr)
        clean=tv.datasets.ImageFolder(base/'train',tfte)
        class TinyVal(torch.utils.data.Dataset):
            def __init__(self):
                self.rows=[]
                for line in (base/'val/val_annotations.txt').read_text().splitlines():
                    f,c,*_=line.split('\t'); self.rows.append((f,tr.class_to_idx[c]))
            def __len__(self): return len(self.rows)
            def __getitem__(self,i):
                from PIL import Image
                f,y=self.rows[i]
                return tfte(Image.open(base/'val/images'/f).convert('RGB')),y
        te=TinyVal(); nc=200
    elif data=='imagenet100':
        from datasets import load_dataset
        raw=load_dataset('clane9/imagenet-100',cache_dir=str(EXTRA/'hf'))
        class HFImages(torch.utils.data.Dataset):
            def __init__(self,ds,transform): self.ds,self.transform=ds,transform
            def __len__(self): return len(self.ds)
            def __getitem__(self,i):
                row=self.ds[i]
                return self.transform(row['image'].convert('RGB')),int(row['label'])
        tr=HFImages(raw['train'],tftr); clean=HFImages(raw['train'],tfte)
        te=HFImages(raw['validation'],tfte); nc=100
    else: raise ValueError(data)
    it,iv=split_indices(len(tr))
    train=loader(Subset(tr,it.tolist()),True,128)
    val=loader(Subset(clean,iv.tolist()),False,512)
    test=loader(te,False,512)
    if family=='cnn':
        model=tv.models.resnet18(num_classes=nc)
        model.conv1=nn.Conv2d(3,64,3,1,1,bias=False); model.maxpool=nn.Identity()
        architecture='ResNet-18'
    else:
        model=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=nc,
                                img_size=size,patch_size=4 if size==32 else 8)
        architecture='ViT-Tiny'
    return Task(name,family,model,train,
                lambda m,b:F.cross_entropy(m(b[0].to(DEV)),b[1].to(DEV)),
                lambda m,s:classification_eval(m,val if s=='val' else test),
                {'dataset':data,'model':architecture,'resolution':size,'n_train':len(it),
                 'n_val':len(iv),'n_test':len(te),'split_seed':42},1000,True)


def graph(name):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
    import tasks as legacy
    ds=name.split('_',1)[1]
    canonical={'cora':'Cora','citeseer':'CiteSeer','pubmed':'PubMed','arxiv':'ogbn-arxiv'}[ds]
    legacy.DATA_ROOT=str(ROOT)
    model,loss,ev,train,meta=legacy.build_graph(canonical)
    def evaluate(m,s):
        r=ev(m)
        return {'score':r['val' if s=='val' else 'test']}
    return Task(name,'graph',model,[None],loss,evaluate,
                {'dataset':canonical,'model':'two-layer GraphSAGE','hidden':128,
                 'split':'public Planetoid / official OGB'},1000,True)


def regression(name):
    d=int(name.split('_')[-1]); m=32; n=2048
    g=torch.Generator().manual_seed(42)
    q=torch.linalg.qr(torch.randn(d,d,generator=g)).Q
    low='lowrank' in name
    vals=torch.cat([torch.linspace(3,1,8),torch.full((d-8,),.05 if low else .5)])
    w=torch.randn(m,d,generator=g)/math.sqrt(d)
    xx=torch.randn(n+512,d,generator=g) @ (q*vals.sqrt()).T
    yy=xx@w.T + .01*torch.randn(n+512,m,generator=g)
    tr=TensorDataset(xx[:n],yy[:n]); va=TensorDataset(xx[n:n+256],yy[n:n+256]); te=TensorDataset(xx[n+256:],yy[n+256:])
    model=nn.Linear(d,m,bias=False)
    def evaluate(model,s):
        x,y=(va if s=='val' else te).tensors
        with torch.no_grad():
            err=model(x.to(DEV))-y.to(DEV)
            return {'score':err.square().mean().item(),'relative_l2':(err.norm()/y.to(DEV).norm()).item()}
    return Task(name,'matreg',model,loader(tr,True,2048),
                lambda m,b:F.mse_loss(m(b[0].to(DEV)),b[1].to(DEV)),evaluate,
                {'dataset':'generated low-rank-covariance regression' if low else 'generated matrix regression',
                 'd':d,'m':m,'n_train':n,'n_val':256,'n_test':256,'noise_std':.01},2000)


def mlp(name='mlp_mnist'):
    import torchvision as tv
    from torchvision import transforms as T
    tf=T.Compose([T.ToTensor(),T.Normalize((.1307,),(.3081,))])
    tr=tv.datasets.MNIST(str(ROOT),train=True,download=True,transform=tf)
    te=tv.datasets.MNIST(str(ROOT),train=False,download=True,transform=tf)
    it,iv=split_indices(len(tr))
    model=nn.Sequential(nn.Flatten(),nn.Linear(784,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,10))
    val=loader(Subset(tr,iv.tolist()),False,512); test=loader(te,False,512)
    return Task(name,'mlp',model,loader(Subset(tr,it.tolist()),True),
                lambda m,b:F.cross_entropy(m(b[0].to(DEV)),b[1].to(DEV)),
                lambda m,s:classification_eval(m,val if s=='val' else test),
                {'dataset':'MNIST','model':'512-512-256 MLP','n_train':len(it),'n_val':len(iv),'n_test':len(te)},2000,True)


class PINN(nn.Module):
    def __init__(self,inp):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inp,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),
                               nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,x):return self.net(x)


def pinn(name):
    eq=name.split('_',1)[1]; dim=1 if eq=='poisson' else 2
    model=PINN(dim)
    def exact(x):
        return torch.sin(math.pi*x[:,0:1]) if dim==1 else -torch.exp(-x[:,1:2])*torch.sin(math.pi*x[:,0:1])
    def residual(model,x):
        x=x.detach().clone().requires_grad_(True)
        u=model(x)
        du=torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        dxx=torch.autograd.grad(du[:,0].sum(),x,create_graph=True)[0][:,0:1]
        if dim==1:return -dxx-math.pi**2*exact(x)
        # Manufactured forced Burgers, nu=.01/pi, exact u=-e^-t sin(pi*x).
        ue=exact(x); ux=-math.pi*torch.exp(-x[:,1:2])*torch.cos(math.pi*x[:,0:1])
        forcing=-ue+ue*ux+(.01/math.pi)*math.pi**2*ue
        return du[:,1:2]+u*du[:,0:1]-(.01/math.pi)*dxx-forcing
    gen=torch.Generator().manual_seed(42)
    val=torch.rand(1024,dim,generator=gen).to(DEV)
    test=torch.rand(4096,dim,generator=gen).to(DEV)
    def loss(model,_):
        x=torch.rand(1024,dim,device=DEV)
        pde=residual(model,x).square().mean()
        b=torch.rand(256,dim,device=DEV); b[:,0]=torch.randint(0,2,(256,),device=DEV).float()
        bc=(model(b)-exact(b)).square().mean()
        if dim==2:
            ic=torch.rand(256,dim,device=DEV);ic[:,1]=0
            bc=bc+(model(ic)-exact(ic)).square().mean()
        return pde+bc
    def evaluate(model,s):
        x=val if s=='val' else test
        with torch.no_grad():
            y=exact(x);err=model(x)-y
            score=(err.norm()/y.norm()).item()
        pde=residual(model,x).detach().square().mean().sqrt().item()
        return {'score':score,'pde_residual_l2':pde}
    return Task(name,'sciml',model,[None],loss,evaluate,
                {'dataset':'manufactured '+eq,'exact_solution':'sin(pi*x)' if dim==1 else '-exp(-t)*sin(pi*x)',
                 'boundary_conditions':'Dirichlet; IC additionally for Burgers','collocation_per_step':1024,
                 'n_val':1024,'n_test':4096},2000)


def darcy_data():
    cache=EXTRA/'darcy32_seed42.npz'
    if cache.exists():return np.load(cache)
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    n=32; rng=np.random.default_rng(42); h=1/(n+1)
    x=np.arange(1,n+1)*h; X,Y=np.meshgrid(x,x,indexing='ij')
    aa=[]; uu=[]
    for _ in range(384):
        field=np.zeros_like(X)
        for k in range(1,5):
            for l in range(1,5):field+=rng.normal()*np.cos(k*np.pi*X)*np.cos(l*np.pi*Y)/(k*k+l*l)
        a=np.exp(field*.7)
        # Harmonic face conductivities, zero Dirichlet boundary values.
        pad=np.pad(a,1,mode='edge')
        ap=2*a*pad[2:,1:-1]/(a+pad[2:,1:-1]); am=2*a*pad[:-2,1:-1]/(a+pad[:-2,1:-1])
        bp=2*a*pad[1:-1,2:]/(a+pad[1:-1,2:]); bm=2*a*pad[1:-1,:-2]/(a+pad[1:-1,:-2])
        main=(ap+am+bp+bm).ravel()/h**2
        upper1=-bp.ravel()[:-1]/h**2;upper1[np.arange(n-1,n*n-1,n)]=0
        lower1=-bm.ravel()[1:]/h**2;lower1[np.arange(n-1,n*n-1,n)]=0
        A=diags([main,upper1,lower1,-ap.ravel()[:-n]/h**2,-am.ravel()[n:]/h**2],[0,1,-1,n,-n],format='csr')
        u=spsolve(A,np.ones(n*n)).reshape(n,n)
        aa.append(a);uu.append(u)
    EXTRA.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,a=np.array(aa,dtype='float32'),u=np.array(uu,dtype='float32'),xy=np.stack([X,Y],-1).astype('float32'))
    return np.load(cache)


class DeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch=nn.Sequential(nn.Linear(64,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,64))
        self.trunk=nn.Sequential(nn.Linear(2,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,64))
        self.bias=nn.Parameter(torch.zeros(()))
        x=torch.arange(1,33)/33
        self.register_buffer('xy',torch.stack(torch.meshgrid(x,x,indexing='ij'),-1).reshape(-1,2))
    def forward(self,a):return self.branch(a[:,::4,::4].reshape(len(a),-1))@self.trunk(self.xy).T/math.sqrt(64)+self.bias


class SpectralConv(nn.Module):
    def __init__(self,width=32,modes=12):
        super().__init__();self.modes=modes
        self.weight=nn.Parameter(torch.randn(width,width,modes,2)*.02)
    def forward(self,x):
        xf=torch.fft.rfft(x);out=torch.zeros_like(xf)
        out[:,:,:self.modes]=torch.einsum('bim,iom->bom',xf[:,:,:self.modes],torch.view_as_complex(self.weight.contiguous()))
        return torch.fft.irfft(out,n=x.shape[-1])


class FNO1D(nn.Module):
    def __init__(self):
        super().__init__();self.lift=nn.Linear(2,32)
        self.spectral=nn.ModuleList([SpectralConv() for _ in range(4)])
        self.local=nn.ModuleList([nn.Conv1d(32,32,1) for _ in range(4)])
        self.head=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):
        h=self.lift(x).transpose(1,2)
        for s,l in zip(self.spectral,self.local):h=F.gelu(s(h)+l(h))
        return self.head(h.transpose(1,2)).squeeze(-1)


def burgers_operator_data():
    cache=EXTRA/'burgers_operator64_seed42.npz'
    if cache.exists():return np.load(cache)
    rng=np.random.default_rng(42);x=np.arange(64)/64
    u=np.zeros((384,64))
    for k in range(1,9):u+=(rng.normal(size=(384,1))*np.sin(2*np.pi*k*x)+rng.normal(size=(384,1))*np.cos(2*np.pi*k*x))/(k*k)
    initial=u.copy();wave=2*np.pi*np.fft.fftfreq(64,d=1/64);mask=np.abs(np.fft.fftfreq(64)*64)<=64/3
    def rhs(v):
        vh=np.fft.fft(v);adv=np.fft.fft(v*np.fft.ifft(1j*wave*vh).real)*mask
        return np.fft.ifft(-adv-.01*wave**2*vh).real
    for _ in range(200):
        dt=.0005;k1=rhs(u);k2=rhs(u+dt*k1/2);k3=rhs(u+dt*k2/2);k4=rhs(u+dt*k3)
        u+=dt*(k1+2*k2+2*k3+k4)/6
    EXTRA.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,x=np.stack([initial,np.broadcast_to(x,initial.shape)],-1).astype('float32'),y=u.astype('float32'))
    return np.load(cache)


def operator_task(name):
    if name=='sciml_darcy':
        d=darcy_data();xx=torch.tensor(d['a']);yy=torch.tensor(d['u']).flatten(1)
        model=DeepONet();meta={'dataset':'generated variable-coefficient Darcy','model':'DeepONet','grid':32,'forcing':1,'boundary':'zero Dirichlet'}
    else:
        d=burgers_operator_data();xx=torch.tensor(d['x']);yy=torch.tensor(d['y'])
        model=FNO1D();meta={'dataset':'generated periodic Burgers operator','model':'FNO1D','grid':64,'viscosity':.01,'final_time':.1,'solver':'dealiased pseudospectral RK4, dt=.0005'}
    def evaluate(model,s):
        sl=slice(256,320) if s=='val' else slice(320,None)
        with torch.no_grad():
            y=yy[sl].to(DEV);pred=model(xx[sl].to(DEV));err=pred-y
            return {'score':(err.flatten(1).norm(dim=1)/y.flatten(1).norm(dim=1)).mean().item(),'mse':err.square().mean().item()}
    meta.update(n_train=256,n_val=64,n_test=64)
    return Task(name,'sciml',model,loader(TensorDataset(xx[:256],yy[:256]),True,32),
                lambda m,b:F.mse_loss(m(b[0].to(DEV)),b[1].to(DEV)),evaluate,meta,2000)


def recommendation(name):
    ds=name.split('_',1)[1]
    if ds=='movielens':
        import pandas as pd
        a=pd.read_csv(ROOT/'ml-1m/ratings.dat',sep='::',engine='python',header=None,names=['u','i','r','t'])
        a=a[a.r>=4].sort_values(['u','t']); source=str((ROOT/'ml-1m/ratings.dat').resolve())
        by={str(u):list(map(str,x.i)) for u,x in a.groupby('u')}
    else:
        folder='amazon-book' if ds=='amazonbooks' else 'yelp2018'
        url=f'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/{folder}/train.txt'
        p=download(url,EXTRA/(folder+'_train.txt'));source=url
        by={}
        for line in p.read_text().splitlines():
            z=line.split()
            if len(z)>=6:by[z[0]]=list(dict.fromkeys(z[1:]))
    # Deterministic 2,000-user subset, no test-driven subset selection.
    users=sorted(by,key=lambda z:hashlib.sha256(('42:'+z).encode()).hexdigest())[:2000]
    users=[u for u in users if len(by[u])>=5]
    items=sorted({i for u in users for i in by[u]});imap={i:k for k,i in enumerate(items)}
    pairs=[]; vp=[];tp=[];seen=[]
    for u,key in enumerate(users):
        seq=[imap[i] for i in by[key]];pairs.extend((u,i) for i in seq[:-2]);vp.append(seq[-2]);tp.append(seq[-1]);seen.append(seq[:-2])
    pairs=torch.tensor(pairs,dtype=torch.long)
    class MF(nn.Module):
        def __init__(self):
            super().__init__();self.users=nn.Embedding(len(users),64);self.items=nn.Embedding(len(items),64)
            nn.init.normal_(self.users.weight,std=.1);nn.init.normal_(self.items.weight,std=.1)
        def forward(self,u,i):return (self.users(u)*self.items(i)).sum(-1)
    model=MF()
    def loss(m,b):
        b=b[0].to(DEV);neg=torch.randint(len(items),(len(b),),device=DEV)
        # Reject observed train positives so BPR does not label them negative.
        # Sorted sparse pair lookup is shared by every optimizer.
        key=b[:,0]*len(items)+neg
        loc=torch.searchsorted(train_keys,key).clamp_max(len(train_keys)-1)
        mask=train_keys[loc]==key
        while mask.any():
            neg[mask]=torch.randint(len(items),(int(mask.sum()),),device=DEV)
            key=b[:,0]*len(items)+neg
            loc=torch.searchsorted(train_keys,key).clamp_max(len(train_keys)-1)
            mask=train_keys[loc]==key
        pos=m(b[:,0],b[:,1]);ns=m(b[:,0],neg)
        return F.softplus(ns-pos).mean()+1e-4*(m.users(b[:,0]).square().mean()+m.items(b[:,1]).square().mean()+m.items(neg).square().mean())
    train_keys=(pairs[:,0]*len(items)+pairs[:,1]).sort().values.to(DEV)
    def evaluate(m,s):
        m.eval();hits=[];nd=[]
        with torch.no_grad():
            for start in range(0,len(users),128):
                score=m.users.weight[start:start+128]@m.items.weight.T
                for j,u in enumerate(range(start,min(start+128,len(users)))):
                    score[j,seen[u]]=-torch.inf
                    if s=='test':score[j,vp[u]]=-torch.inf
                top=score.topk(min(20,len(items)),dim=1).indices.cpu()
                for j,u in enumerate(range(start,min(start+128,len(users)))):
                    target=vp[u] if s=='val' else tp[u];found=(top[j]==target).nonzero().flatten()
                    hits.append(float(len(found)>0));nd.append(1/math.log2(int(found[0])+2) if len(found) else 0.)
        return {'score':float(np.mean(hits)),'recall@20':float(np.mean(hits)),'ndcg@20':float(np.mean(nd))}
    return Task(name,'recsys',model,loader(TensorDataset(pairs),True,4096),loss,evaluate,
                {'dataset':ds,'model':'MF-BPR, latent64','source':source,'n_users':len(users),'n_items':len(items),
                 'n_train':len(pairs),'split':'last two input interactions per user held out; author-preprocessed order for Amazon/Yelp',
                 'subset':'first 2000 users by SHA256(42:user_id)','metric':'all-catalog leave-one-out Recall@20'},1000,True)


def continual(name):
    import torchvision as tv
    from torchvision import transforms as T
    split='splitcifar' in name
    tasks=[];vals=[];tests=[]
    if split:
        raw=tv.datasets.CIFAR100(str(ROOT),train=True,download=True)
        te=tv.datasets.CIFAR100(str(ROOT),train=False,download=False)
        tf=T.Compose([T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
        raw.transform=tf;te.transform=tf
        for k in range(5):
            ids=[i for i,y in enumerate(raw.targets) if 20*k<=y<20*(k+1)]
            it,iv=split_indices(len(ids));tasks.append(loader(Subset(raw,[ids[i] for i in it]),True,128))
            vals.append(loader(Subset(raw,[ids[i] for i in iv]),False,512))
            tests.append(loader(Subset(te,[i for i,y in enumerate(te.targets) if 20*k<=y<20*(k+1)]),False,512))
        model=tv.models.resnet18(num_classes=100);model.conv1=nn.Conv2d(3,64,3,1,1,bias=False);model.maxpool=nn.Identity()
    else:
        for angle in [0,30,60,90]:
            tf=T.Compose([T.Lambda(lambda x,a=angle:T.functional.rotate(x,a)),T.ToTensor(),T.Normalize((.1307,),(.3081,))])
            tr=tv.datasets.MNIST(str(ROOT),train=True,download=True,transform=tf)
            te=tv.datasets.MNIST(str(ROOT),train=False,download=False,transform=tf)
            it,iv=split_indices(len(tr));tasks.append(loader(Subset(tr,it.tolist()),True))
            vals.append(loader(Subset(tr,iv.tolist()),False,512));tests.append(loader(te,False,512))
        model=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,10))
    def evaluate(m,s):
        scores=[classification_eval(m,dl)['score'] for dl in (vals if s=='val' else tests)]
        return {'score':float(np.mean(scores)),'per_task':scores}
    return Task(name,'cnn' if split else 'continual',model,tasks[0],
                lambda m,b:F.cross_entropy(m(b[0].to(DEV)),b[1].to(DEV)),evaluate,
                {'dataset':'Split CIFAR-100' if split else 'Rotated MNIST','n_tasks':len(tasks),
                 'protocol':'single head, no task id at inference, no replay; 1000 total epochs split equally'},1000,True,tasks)


TASKS=(['matreg_'+str(d) for d in (128,256,512)]+
       ['lowrank_'+str(d) for d in (128,256,512)]+['mlp_mnist']+
       ['cnn_'+x for x in ('cifar10','cifar100','tinyimagenet','imagenet100')]+
       ['vit_'+x for x in ('cifar100','tinyimagenet','imagenet100')]+
       ['graph_'+x for x in ('cora','citeseer','pubmed','arxiv')]+
       ['sciml_'+x for x in ('poisson','burgers','darcy','fno')]+
       ['recsys_'+x for x in ('movielens','amazonbooks','yelp')]+
       ['continual_rotmnist','continual_splitcifar100'])


def build(name):
    if name.startswith(('matreg_','lowrank_')):return regression(name)
    if name.startswith(('cnn_','vit_')):return vision(name)
    if name.startswith('graph_'):return graph(name)
    if name.startswith('mlp_'):return mlp(name)
    if name in ('sciml_poisson','sciml_burgers'):return pinn(name)
    if name.startswith('sciml_'):return operator_task(name)
    if name.startswith('recsys_'):return recommendation(name)
    if name.startswith('continual_'):return continual(name)
    raise ValueError(name)
