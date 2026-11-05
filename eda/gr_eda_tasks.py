from pathlib import Path
"""
GrassRoute EDA — task-family runners.
Each runner trains a small-but-representative model briefly, hooks the
README-specified routed parameter matrices, and records C=G^T G spectrum +
subspace drift over training windows. Output: one JSON per task under eda_out/.

Usage:
  python gr_eda_tasks.py <family> --out <dir> [--device cuda:0]
families: matreg mlp cnn vit graph sciml recsys continual routing_inventory
"""
import os, sys, json, math, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gr_eda_core import CovTracker, RANKS, random_subspace_drift_null

SEED = 42

# torchvision's default MNIST mirror (yann.lecun.com) is dead -> use working S3 mirror
try:
    from torchvision.datasets import MNIST as _MNIST
    _MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
except Exception:
    pass

def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def flat2d(p):
    # parameter/grad tensor -> (m, d) with d = product of non-first dims (right subspace = input side)
    g = p
    if g.dim() == 1: return None
    if g.dim() == 2: return g
    return g.reshape(g.shape[0], -1)   # conv (out,in,kh,kw)->(out,in*kh*kw)

def is_routable(name, p, min_d=8, max_d=6000):
    if p.dim() < 2: return False
    d = int(np.prod(p.shape[1:]))
    return (min_d <= d <= max_d)

# ---------------- generic trainer with cov tracking ----------------
def run_training(model, batches_fn, routed, n_steps, snap_every, device,
                 loss_fn, tag, opt=None, drift_rank=16):
    """routed: dict name->param. Snapshots every snap_every steps."""
    model.to(device).train()
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trackers = {n: CovTracker(n, int(np.prod(p.shape[1:])), drift_rank=min(drift_rank, int(np.prod(p.shape[1:]))))
                for n, p in routed.items()}
    loss_hist = []
    step = 0
    t0 = time.time()
    while step < n_steps:
        for xb, yb in batches_fn():
            if step >= n_steps: break
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            for n, p in routed.items():
                if p.grad is not None:
                    g = flat2d(p.grad)
                    if g is not None: trackers[n].add(g)
            opt.step()
            loss_hist.append(float(loss.item()))
            step += 1
            if step % snap_every == 0:
                for tr in trackers.values(): tr.snapshot(step)
    for tr in trackers.values():
        if tr.cnt > 0: tr.snapshot(step)
    return {
        "tag": tag, "n_steps": step, "snap_every": snap_every,
        "wall_sec": round(time.time()-t0, 1),
        "loss_first": loss_hist[0] if loss_hist else None,
        "loss_last": float(np.mean(loss_hist[-20:])) if loss_hist else None,
        "routed_layers": {n: {"shape": list(p.shape), "d_right": int(np.prod(p.shape[1:]))}
                          for n, p in routed.items()},
        "trackers": {n: tr.summary() for n, tr in trackers.items()},
        "null_drift": {n: random_subspace_drift_null(tr.d, tr.drift_rank)
                       for n, tr in trackers.items()},
    }

# ============================================================
# 1. MATRIX REGRESSION / LOW-RANK REGRESSION
# ============================================================
def task_matreg(device, out):
    set_seed()
    res = {}
    for d, r_true in [(128, 8), (256, 16), (512, 16)]:
        m = 64; n = 4096
        U = torch.linalg.qr(torch.randn(d, r_true))[0]
        Wstar = (torch.randn(m, r_true) @ U.T)         # (m,d) rank r_true
        X = torch.randn(n, d)
        Y = X @ Wstar.T + 0.01*torch.randn(n, m)
        ds = torch.utils.data.TensorDataset(X, Y)
        W = nn.Parameter(torch.randn(m, d)*0.01)
        model = nn.Module(); model.W = W
        model.forward = lambda x: x @ model.W.T
        def batches():
            dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
            for xb, yb in dl: yield xb, yb
        routed = {"W": W}
        opt = torch.optim.AdamW([W], lr=3e-3)
        r = run_training(model, batches, routed, n_steps=1500, snap_every=100,
                         device=device, loss_fn=lambda o,y: F.mse_loss(o,y),
                         tag=f"matreg_d{d}_r{r_true}", opt=opt, drift_rank=min(32,r_true*2))
        r["d"] = d; r["r_true"] = r_true
        res[f"d{d}_rtrue{r_true}"] = r
    save(out, "matreg", res)

# ============================================================
# 2. DEEP MLP (real-ish: FashionMNIST-like synthetic classification)
# ============================================================
def task_mlp(device, out):
    set_seed()
    from torchvision import datasets, transforms
    root = os.environ.get("GR_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    tr = datasets.MNIST(root, train=True, download=True,
                        transform=transforms.ToTensor())
    X = tr.data.float().reshape(-1,784)/255.0; Y = tr.targets
    ds = torch.utils.data.TensorDataset(X, Y)
    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.f1=nn.Linear(784,512); s.f2=nn.Linear(512,512); s.f3=nn.Linear(512,256); s.head=nn.Linear(256,10)
        def forward(s,x):
            x=F.relu(s.f1(x)); x=F.relu(s.f2(x)); x=F.relu(s.f3(x)); return s.head(x)
    model = MLP()
    routed = {"f1.W":model.f1.weight,"f2.W":model.f2.weight,"f3.W":model.f3.weight,"head.W":model.head.weight}
    def batches():
        dl=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True)
        for xb,yb in dl: yield xb,yb
    r = run_training(model, batches, routed, n_steps=2000, snap_every=100,
                     device=device, loss_fn=F.cross_entropy, tag="mlp_mnist", drift_rank=16)
    save(out, "mlp", {"mnist_mlp": r})

# ============================================================
# 3. CNN — CIFAR-10 ResNet-18
# ============================================================
def task_cnn(device, out):
    set_seed()
    from torchvision import datasets, transforms, models
    root = os.environ.get("GR_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    tf = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=4)
    net = models.resnet18(num_classes=10)
    net.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); net.maxpool = nn.Identity()
    # routed: fc (classifier), a 1x1 conv (downsample), and selected 3x3 conv kernels
    routed = {}
    routed["fc"] = net.fc.weight                                    # (10,512) d=512
    routed["layer4.0.conv1"] = net.layer4[0].conv1.weight          # (512,256,3,3) d=2304
    routed["layer3.0.conv1"] = net.layer3[0].conv1.weight          # (256,128,3,3) d=1152
    routed["layer2.0.downsample.0"] = net.layer2[0].downsample[0].weight  # 1x1 (128,64,1,1) d=64
    def batches():
        for xb,yb in dl: yield xb,yb
    r = run_training(net, batches, routed, n_steps=1200, snap_every=100,
                     device=device, loss_fn=F.cross_entropy, tag="cnn_cifar10_resnet18", drift_rank=16)
    save(out, "cnn", {"cifar10_resnet18": r})

# ============================================================
# 4. ViT — small ViT on CIFAR-10 (timm)
# ============================================================
def task_vit(device, out):
    set_seed()
    import timm
    from torchvision import datasets, transforms
    root = os.environ.get("GR_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    tf = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=4)
    net = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                            num_classes=10, img_size=32, patch_size=4)
    # routed: q,k,v (qkv fused), proj (o), mlp fc1/fc2 in block 0 and a mid block
    routed = {}
    for bi in [0, 5]:
        blk = net.blocks[bi]
        routed[f"blk{bi}.qkv"] = blk.attn.qkv.weight   # (3*dim, dim)
        routed[f"blk{bi}.proj"] = blk.attn.proj.weight # (dim, dim)
        routed[f"blk{bi}.mlp.fc1"] = blk.mlp.fc1.weight # (hid, dim)
        routed[f"blk{bi}.mlp.fc2"] = blk.mlp.fc2.weight # (dim, hid)
    def batches():
        for xb,yb in dl: yield xb,yb
    r = run_training(net, batches, routed, n_steps=1000, snap_every=100,
                     device=device, loss_fn=F.cross_entropy, tag="vit_tiny_cifar10", drift_rank=16)
    save(out, "vit", {"cifar10_vit_tiny": r})

# ============================================================
# 5. GRAPH — Planetoid (Cora/Citeseer/PubMed) + ogbn-arxiv GraphSAGE
# ============================================================
def task_graph(device, out):
    set_seed()
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv, SAGEConv
    root = str(Path(os.environ.get("GR_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "graph")
    res = {}
    for name in ["Cora","Citeseer","PubMed"]:
        ds = Planetoid(root, name); data = ds[0].to(device)
        class GCN(nn.Module):
            def __init__(s,ic,hc,oc):
                super().__init__(); s.c1=GCNConv(ic,hc); s.c2=GCNConv(hc,oc)
            def forward(s,d):
                x=F.relu(s.c1(d.x,d.edge_index)); return s.c2(x,d.edge_index)
        model=GCN(ds.num_features,128,ds.num_classes).to(device)
        # GCNConv weight lives in .lin.weight (out,in)
        routed={"c1.lin":model.c1.lin.weight,"c2.lin":model.c2.lin.weight}
        opt=torch.optim.AdamW(model.parameters(),lr=1e-2,weight_decay=5e-4)
        trackers={n:__import__("gr_eda_core").CovTracker(n,int(np.prod(p.shape[1:])),
                   drift_rank=min(16,int(np.prod(p.shape[1:])))) for n,p in routed.items()}
        losshist=[]; t0=time.time(); NS=800; SN=50
        for step in range(1,NS+1):
            model.train(); opt.zero_grad(set_to_none=True)
            outp=model(data); loss=F.cross_entropy(outp[data.train_mask],data.y[data.train_mask])
            loss.backward()
            for n,p in routed.items():
                if p.grad is not None:
                    g=flat2d(p.grad);
                    if g is not None: trackers[n].add(g)
            opt.step(); losshist.append(float(loss.item()))
            if step%SN==0:
                for tr in trackers.values(): tr.snapshot(step)
        res[name]={"tag":f"graph_{name}_GCN","n_steps":NS,"snap_every":SN,
                   "wall_sec":round(time.time()-t0,1),
                   "loss_first":losshist[0],"loss_last":float(np.mean(losshist[-20:])),
                   "num_features":ds.num_features,"num_classes":ds.num_classes,
                   "num_nodes":int(data.num_nodes),"num_edges":int(data.edge_index.shape[1]),
                   "routed_layers":{n:{"shape":list(p.shape),"d_right":int(np.prod(p.shape[1:]))} for n,p in routed.items()},
                   "trackers":{n:tr.summary() for n,tr in trackers.items()},
                   "null_drift":{n:random_subspace_drift_null(tr.d,tr.drift_rank) for n,tr in trackers.items()}}
    # ogbn-arxiv GraphSAGE (full-batch, few steps)
    try:
        # ogb 1.3.6 uses torch.load w/o weights_only=False -> incompatible with torch>=2.6 default
        import torch.serialization as _ts
        try:
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage
            _ts.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        except Exception:
            pass
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        from ogb.nodeproppred import PygNodePropPredDataset
        ds=PygNodePropPredDataset("ogbn-arxiv",root=str(Path(os.environ.get("GR_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "ogb"))
        data=ds[0].to(device); split=ds.get_idx_split()
        tr_idx=split["train"].to(device); y=data.y.squeeze().to(device)
        class SAGE(nn.Module):
            def __init__(s,ic,hc,oc):
                super().__init__(); s.c1=SAGEConv(ic,hc); s.c2=SAGEConv(hc,oc)
            def forward(s,d):
                x=F.relu(s.c1(d.x,d.edge_index)); return s.c2(x,d.edge_index)
        model=SAGE(ds.num_features,256,int(ds.num_classes)).to(device)
        # SAGEConv: lin_l, lin_r weights
        routed={}
        for cn,c in [("c1",model.c1),("c2",model.c2)]:
            for ln in ["lin_l","lin_r"]:
                w=getattr(c,ln,None)
                if w is not None and hasattr(w,"weight"): routed[f"{cn}.{ln}"]=w.weight
        opt=torch.optim.AdamW(model.parameters(),lr=1e-2)
        trackers={n:__import__("gr_eda_core").CovTracker(n,int(np.prod(p.shape[1:])),
                   drift_rank=min(16,int(np.prod(p.shape[1:])))) for n,p in routed.items()}
        losshist=[]; t0=time.time(); NS=300; SN=25
        for step in range(1,NS+1):
            model.train(); opt.zero_grad(set_to_none=True)
            outp=model(data); loss=F.cross_entropy(outp[tr_idx],y[tr_idx]); loss.backward()
            for n,p in routed.items():
                if p.grad is not None:
                    g=flat2d(p.grad)
                    if g is not None: trackers[n].add(g)
            opt.step(); losshist.append(float(loss.item()))
            if step%SN==0:
                for tr in trackers.values(): tr.snapshot(step)
        res["ogbn-arxiv"]={"tag":"graph_ogbn-arxiv_GraphSAGE","n_steps":NS,"snap_every":SN,
                   "wall_sec":round(time.time()-t0,1),
                   "loss_first":losshist[0],"loss_last":float(np.mean(losshist[-20:])),
                   "num_features":ds.num_features,"num_classes":int(ds.num_classes),
                   "num_nodes":int(data.num_nodes),"num_edges":int(data.edge_index.shape[1]),
                   "routed_layers":{n:{"shape":list(p.shape),"d_right":int(np.prod(p.shape[1:]))} for n,p in routed.items()},
                   "trackers":{n:tr.summary() for n,tr in trackers.items()},
                   "null_drift":{n:random_subspace_drift_null(tr.d,tr.drift_rank) for n,tr in trackers.items()}}
    except Exception as e:
        res["ogbn-arxiv"]={"error":f"{type(e).__name__}: {e}"}
    save(out,"graph",res)

# ============================================================
# 6. SciML — PINN for 1D Burgers
# ============================================================
def task_sciml(device, out):
    set_seed()
    nu=0.01/math.pi
    class PINN(nn.Module):
        def __init__(s,w=128):
            super().__init__()
            s.l1=nn.Linear(2,w); s.l2=nn.Linear(w,w); s.l3=nn.Linear(w,w); s.l4=nn.Linear(w,w); s.out=nn.Linear(w,1)
        def forward(s,xt):
            h=torch.tanh(s.l1(xt)); h=torch.tanh(s.l2(h)); h=torch.tanh(s.l3(h)); h=torch.tanh(s.l4(h)); return s.out(h)
    model=PINN().to(device)
    routed={n:getattr(model,n).weight for n in ["l1","l2","l3","l4","out"] if getattr(model,n).weight.dim()==2 and int(np.prod(getattr(model,n).weight.shape[1:]))>=8}
    # actually l1 d=2 too small -> drop
    routed={k:v for k,v in routed.items() if int(np.prod(v.shape[1:]))>=8}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    trackers={n:__import__("gr_eda_core").CovTracker(n,int(np.prod(p.shape[1:])),
               drift_rank=min(16,int(np.prod(p.shape[1:])))) for n,p in routed.items()}
    losshist=[]; t0=time.time(); NS=2000; SN=100
    for step in range(1,NS+1):
        xf=torch.rand(2048,1,device=device)*2-1; tf=torch.rand(2048,1,device=device)
        xf.requires_grad_(True); tf.requires_grad_(True)
        u=model(torch.cat([xf,tf],1))
        ux=torch.autograd.grad(u,xf,torch.ones_like(u),create_graph=True)[0]
        ut=torch.autograd.grad(u,tf,torch.ones_like(u),create_graph=True)[0]
        uxx=torch.autograd.grad(ux,xf,torch.ones_like(ux),create_graph=True)[0]
        f=ut+u*ux-nu*uxx
        # IC: u(x,0)=-sin(pi x); BC u(+-1,t)=0
        x0=torch.rand(256,1,device=device)*2-1; t0_=torch.zeros_like(x0)
        u0=model(torch.cat([x0,t0_],1)); ic=u0+torch.sin(math.pi*x0)
        tb=torch.rand(256,1,device=device); xb=torch.cat([torch.ones(128,1),-torch.ones(128,1)]).to(device)
        ub=model(torch.cat([xb,tb],1))
        loss=(f**2).mean()+(ic**2).mean()+(ub**2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        for n,p in routed.items():
            if p.grad is not None:
                g=flat2d(p.grad)
                if g is not None: trackers[n].add(g)
        opt.step(); losshist.append(float(loss.item()))
        if step%SN==0:
            for tr in trackers.values(): tr.snapshot(step)
    save(out,"sciml",{"burgers_pinn":{"tag":"sciml_burgers_pinn","n_steps":NS,"snap_every":SN,
        "wall_sec":round(time.time()-t0,1),"loss_first":losshist[0],"loss_last":float(np.mean(losshist[-20:])),
        "routed_layers":{n:{"shape":list(p.shape),"d_right":int(np.prod(p.shape[1:]))} for n,p in routed.items()},
        "trackers":{n:tr.summary() for n,tr in trackers.items()},
        "null_drift":{n:random_subspace_drift_null(tr.d,tr.drift_rank) for n,tr in trackers.items()}}})

# ============================================================
# 7. RECSYS — MovieLens-1M matrix factorization
# ============================================================
def task_recsys(device, out):
    set_seed()
    import urllib.request, zipfile, io, pandas as pd
    root=str(Path(os.environ.get("GR_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "ml-1m")
    os.makedirs(root,exist_ok=True)
    dat=os.path.join(root,"ratings.dat")
    if not os.path.exists(dat):
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url,timeout=60).read()))
        z.extract("ml-1m/ratings.dat",path=os.path.dirname(root))
    df=pd.read_csv(dat,sep="::",engine="python",names=["u","i","r","ts"])
    df["u"]=(df["u"]-1).astype("int64"); df["i"]=df["i"].astype("category").cat.codes.astype("int64")
    nu=int(df["u"].max()+1); ni=int(df["i"].max()+1)
    U=torch.tensor(df["u"].values,dtype=torch.long); I=torch.tensor(df["i"].values,dtype=torch.long); Rt=torch.tensor(df["r"].values,dtype=torch.float32)
    ds=torch.utils.data.TensorDataset(U,I,Rt)
    K=64
    class MF(nn.Module):
        def __init__(s):
            super().__init__(); s.uemb=nn.Embedding(nu,K); s.iemb=nn.Embedding(ni,K)
            s.ub=nn.Embedding(nu,1); s.ib=nn.Embedding(ni,1); s.g=nn.Parameter(torch.tensor(3.5))
        def forward(s,u,i):
            return (s.uemb(u)*s.iemb(i)).sum(1)+s.ub(u).squeeze(1)+s.ib(i).squeeze(1)+s.g
    model=MF().to(device)
    # routed: embedding matrices (item emb is the big matrix ni x K -> right subspace K)
    routed={"uemb":model.uemb.weight,"iemb":model.iemb.weight}
    opt=torch.optim.AdamW(model.parameters(),lr=5e-3,weight_decay=1e-5)
    trackers={n:__import__("gr_eda_core").CovTracker(n,int(np.prod(p.shape[1:])),
               drift_rank=min(16,int(np.prod(p.shape[1:])))) for n,p in routed.items()}
    dl=torch.utils.data.DataLoader(ds,batch_size=4096,shuffle=True)
    losshist=[]; t0=time.time(); NS=1000; SN=50; step=0
    while step<NS:
        for u,i,r in dl:
            if step>=NS: break
            u=u.to(device);i=i.to(device);r=r.to(device)
            opt.zero_grad(set_to_none=True); pred=model(u,i); loss=F.mse_loss(pred,r); loss.backward()
            for n,p in routed.items():
                if p.grad is not None:
                    g=p.grad.coalesce().to_dense() if p.grad.is_sparse else p.grad
                    g=flat2d(g)
                    if g is not None: trackers[n].add(g)
            opt.step(); losshist.append(float(loss.item())); step+=1
            if step%SN==0:
                for tr in trackers.values(): tr.snapshot(step)
    save(out,"recsys",{"movielens1m_mf":{"tag":"recsys_ml1m_mf","n_steps":NS,"snap_every":SN,
        "wall_sec":round(time.time()-t0,1),"loss_first":losshist[0],"loss_last":float(np.mean(losshist[-20:])),
        "n_users":int(nu),"n_items":int(ni),"n_ratings":len(df),"K":K,
        "routed_layers":{n:{"shape":list(p.shape),"d_right":int(np.prod(p.shape[1:]))} for n,p in routed.items()},
        "trackers":{n:tr.summary() for n,tr in trackers.items()},
        "null_drift":{n:random_subspace_drift_null(tr.d,tr.drift_rank) for n,tr in trackers.items()}}})

# ============================================================
# 8. CONTINUAL — Rotated MNIST MLP (drift across tasks)
# ============================================================
def task_continual(device, out):
    set_seed()
    from torchvision import datasets, transforms
    root=os.environ.get("GR_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    tr=datasets.MNIST(root,train=True,download=True,transform=transforms.ToTensor())
    X=tr.data.float().reshape(-1,1,28,28)/255.0; Y=tr.targets
    class MLP(nn.Module):
        def __init__(s):
            super().__init__(); s.f1=nn.Linear(784,256); s.f2=nn.Linear(256,256); s.head=nn.Linear(256,10)
        def forward(s,x):
            x=x.reshape(x.shape[0],-1); x=F.relu(s.f1(x)); x=F.relu(s.f2(x)); return s.head(x)
    model=MLP().to(device)
    routed={"f1":model.f1.weight,"f2":model.f2.weight,"head":model.head.weight}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    trackers={n:__import__("gr_eda_core").CovTracker(n,int(np.prod(p.shape[1:])),
               drift_rank=min(16,int(np.prod(p.shape[1:])))) for n,p in routed.items()}
    angles=[0,30,60,90]; losshist=[]; t0=time.time(); SN=50; step=0
    task_boundaries=[]
    for ti,ang in enumerate(angles):
        Xr=torch.rot90(X,k=ang//90,dims=[2,3]) if ang in (0,90) else \
           transforms.functional.rotate(X,ang)
        ds=torch.utils.data.TensorDataset(Xr,Y)
        dl=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True)
        steps_this=300; s0=step
        for xb,yb in dl:
            if step-s0>=steps_this: break
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(xb),yb); loss.backward()
            for n,p in routed.items():
                if p.grad is not None:
                    g=flat2d(p.grad)
                    if g is not None: trackers[n].add(g)
            opt.step(); losshist.append(float(loss.item())); step+=1
            if step%SN==0:
                for tr in trackers.values(): tr.snapshot(step)
        task_boundaries.append({"task":ti,"angle":ang,"step_end":step})
    save(out,"continual",{"rotmnist_mlp":{"tag":"continual_rotmnist_mlp","n_steps":step,"snap_every":SN,
        "wall_sec":round(time.time()-t0,1),"loss_first":losshist[0],"loss_last":float(np.mean(losshist[-20:])),
        "task_boundaries":task_boundaries,
        "routed_layers":{n:{"shape":list(p.shape),"d_right":int(np.prod(p.shape[1:]))} for n,p in routed.items()},
        "trackers":{n:tr.summary() for n,tr in trackers.items()},
        "null_drift":{n:random_subspace_drift_null(tr.d,tr.drift_rank) for n,tr in trackers.items()}}})

def save(out, name, obj):
    os.makedirs(out, exist_ok=True)
    p=os.path.join(out,f"{name}.json")
    with open(p,"w") as f: json.dump(obj,f)
    print(f"[saved] {p}", flush=True)

DISPATCH={"matreg":task_matreg,"mlp":task_mlp,"cnn":task_cnn,"vit":task_vit,
          "graph":task_graph,"sciml":task_sciml,"recsys":task_recsys,"continual":task_continual}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("family"); ap.add_argument("--out",required=True); ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args()
    dev=torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"[start] {a.family} on {dev}", flush=True)
    DISPATCH[a.family](dev, a.out)
    print(f"[done] {a.family}", flush=True)
