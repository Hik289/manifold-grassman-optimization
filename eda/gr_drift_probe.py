from pathlib import Path
"""
GrassRoute EDA — rigorous drift-vs-noise probe (multi-rank).

Key question (H0 premise P2): does the PRINCIPAL gradient subspace drift SLOWLY?
Confound: with tiny eigengap at a fixed r, the empirical top-r subspace is noisily
estimated, so consecutive-window "drift" is inflated by estimation noise.

Split-half decomposition per window W (steps):
  C_a = cov over first half of window ; C_b = cov over second half.
  within_noise(r)   = d_Gr( topr(C_a), topr(C_b) )        # SAME subspace, 2x est. noise
  C_full = cov over full window (stored).
  consec_drift(r)   = d_Gr( topr(C_prev_full), topr(C_full) )  # true drift + 2x noise
  to_final(r)       = d_Gr( topr(C_full), topr(C_final_full) ) # convergence to a fixed ref
True slow-drift is supported when consec_drift ≈ within_noise (drift ≲ noise)
and/or to_final decreases & stabilizes. Reported at r in {1,2,4,8,16}.
"""
import os, sys, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gr_eda_core import principal_angles

RANKS=[1,2,4,8,16]
def topr(C,r):
    w,V=np.linalg.eigh(C); return V[:,::-1][:,:r]
def gd(U1,U2):
    th=principal_angles(U1,U2); return float(np.sqrt(np.sum(th**2))), float(th.max())

class SplitTracker:
    def __init__(s,name,d):
        s.name=name; s.d=d; s.a=np.zeros((d,d)); s.b=np.zeros((d,d)); s.na=0; s.nb=0
        s.snaps=[]; s.prev_full=None; s.fulls=[]  # store full covs to compute to_final
    def add(s,G,half):
        g=G.detach().float(); C=(g.T@g).cpu().numpy().astype(np.float64)
        if half==0: s.a+=C; s.na+=1
        else: s.b+=C; s.nb+=1
    def snap(s,step):
        if s.na==0 or s.nb==0: return
        Ca=s.a/s.na; Cb=s.b/s.nb; Cf=(s.a+s.b)/(s.na+s.nb)
        rec={"step":int(step),"within_noise":{},"consec_drift":{}}
        for r in RANKS:
            if r<s.d:
                Ua=topr(Ca,r); Ub=topr(Cb,r)
                rec["within_noise"][str(r)]=gd(Ua,Ub)[0]
                if s.prev_full is not None:
                    Up=topr(s.prev_full,r); Uc=topr(Cf,r)
                    rec["consec_drift"][str(r)]=gd(Up,Uc)[0]
        s.prev_full=Cf; s.fulls.append((step,Cf)); s.snaps.append(rec)
        s.a[:]=0;s.b[:]=0;s.na=0;s.nb=0
    def finalize(s):
        # to_final: distance of each stored full cov's top-r to the FINAL full cov top-r
        if not s.fulls: return
        _,Cfin=s.fulls[-1]
        Ufin={r:topr(Cfin,r) for r in RANKS if r<s.d}
        for (step,Cf),rec in zip(s.fulls,s.snaps):
            rec["to_final"]={}
            for r in RANKS:
                if r<s.d: rec["to_final"][str(r)]=gd(topr(Cf,r),Ufin[r])[0]
    def out(s):
        s.finalize()
        return {"layer":s.name,"d":s.d,"ranks":RANKS,"snaps":s.snaps}

def train_probe(model,routed,batches,n_steps,snap_every,device,loss_fn,tag,opt=None):
    model.to(device).train()
    if opt is None: opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    trk={n:SplitTracker(n,int(np.prod(p.shape[1:]))) for n,p in routed.items()}
    step=0; half_pt=snap_every//2; t0=time.time(); lh=[]
    while step<n_steps:
        for xb,yb in batches():
            if step>=n_steps: break
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward()
            half=0 if (step%snap_every)<half_pt else 1
            for n,p in routed.items():
                if p.grad is not None:
                    g=p.grad if p.grad.dim()==2 else p.grad.reshape(p.grad.shape[0],-1)
                    trk[n].add(g,half)
            opt.step(); lh.append(float(loss.item())); step+=1
            if step%snap_every==0:
                for t in trk.values(): t.snap(step)
    return {"tag":tag,"n_steps":step,"snap_every":snap_every,"wall_sec":round(time.time()-t0,1),
            "loss_first":lh[0],"loss_last":float(np.mean(lh[-20:])),
            "trackers":{n:t.out() for n,t in trk.items()}}

def main(out,device):
    res={}
    # 1) matreg d256 r16 (planted low-rank, fast)
    torch.manual_seed(42); np.random.seed(42)
    d=256; rt=16; m=64; n=4096
    U=torch.linalg.qr(torch.randn(d,rt))[0]; Wstar=torch.randn(m,rt)@U.T
    X=torch.randn(n,d); Y=X@Wstar.T+0.01*torch.randn(n,m)
    ds=torch.utils.data.TensorDataset(X,Y)
    W=nn.Parameter(torch.randn(m,d)*0.01); model=nn.Module(); model.W=W
    model.forward=lambda x:x@model.W.T
    def b1():
        for xb,yb in torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True): yield xb,yb
    res["matreg_d256_r16"]=train_probe(model,{"W":W},b1,1200,100,device,
        lambda o,y:F.mse_loss(o,y),"matreg_d256_r16",opt=torch.optim.AdamW([W],lr=3e-3))
    res["matreg_d256_r16"]["r_true"]=rt
    # 2) short MLP on MNIST
    from torchvision import datasets, transforms
    from torchvision.datasets import MNIST
    MNIST.mirrors=["https://ossci-datasets.s3.amazonaws.com/mnist/"]
    root=os.environ.get("GR_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    tr=datasets.MNIST(root,train=True,download=False,transform=transforms.ToTensor())
    XX=tr.data.float().reshape(-1,784)/255.0; YY=tr.targets
    dsm=torch.utils.data.TensorDataset(XX,YY)
    class MLP(nn.Module):
        def __init__(s):
            super().__init__(); s.f1=nn.Linear(784,512); s.f2=nn.Linear(512,256); s.head=nn.Linear(256,10)
        def forward(s,x): return s.head(F.relu(s.f2(F.relu(s.f1(x)))))
    mm=MLP()
    def b2():
        for xb,yb in torch.utils.data.DataLoader(dsm,batch_size=128,shuffle=True): yield xb,yb
    res["mlp_mnist"]=train_probe(mm,{"f1":mm.f1.weight,"f2":mm.f2.weight,"head":mm.head.weight},
        b2,800,100,device,F.cross_entropy,"mlp_mnist")
    # 3) short PINN (rank-1-ish, extreme concentration)
    torch.manual_seed(42)
    nu=0.01/math.pi
    class PINN(nn.Module):
        def __init__(s,w=128):
            super().__init__(); s.l1=nn.Linear(2,w); s.l2=nn.Linear(w,w); s.l3=nn.Linear(w,w); s.out=nn.Linear(w,1)
        def forward(s,xt): return s.out(torch.tanh(s.l3(torch.tanh(s.l2(torch.tanh(s.l1(xt)))))))
    pn=PINN().to(device); optp=torch.optim.AdamW(pn.parameters(),lr=1e-3)
    routed={"l2":pn.l2.weight,"l3":pn.l3.weight,"out":pn.out.weight}
    trk={n:SplitTracker(n,int(np.prod(p.shape[1:]))) for n,p in routed.items()}
    t0=time.time(); lh=[]; NS=800; SN=100
    for step in range(1,NS+1):
        xf=(torch.rand(2048,1,device=device)*2-1).requires_grad_(True)
        tf=torch.rand(2048,1,device=device).requires_grad_(True)
        u=pn(torch.cat([xf,tf],1))
        ux=torch.autograd.grad(u,xf,torch.ones_like(u),create_graph=True)[0]
        ut=torch.autograd.grad(u,tf,torch.ones_like(u),create_graph=True)[0]
        uxx=torch.autograd.grad(ux,xf,torch.ones_like(ux),create_graph=True)[0]
        f=ut+u*ux-nu*uxx
        x0=torch.rand(256,1,device=device)*2-1
        ic=pn(torch.cat([x0,torch.zeros_like(x0)],1))+torch.sin(math.pi*x0)
        loss=(f**2).mean()+(ic**2).mean()
        optp.zero_grad(set_to_none=True); loss.backward()
        half=0 if (step%SN)<SN//2 else 1
        for n,p in routed.items():
            if p.grad is not None: trk[n].add(p.grad,half)
        optp.step(); lh.append(float(loss.item()))
        if step%SN==0:
            for t in trk.values(): t.snap(step)
    res["pinn_burgers"]={"tag":"pinn_burgers","n_steps":NS,"snap_every":SN,"wall_sec":round(time.time()-t0,1),
        "loss_first":lh[0],"loss_last":float(np.mean(lh[-20:])),"trackers":{n:t.out() for n,t in trk.items()}}
    os.makedirs(out,exist_ok=True)
    json.dump(res,open(os.path.join(out,"drift_probe.json"),"w"))
    print("[saved] drift_probe.json",flush=True)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args(); dev=torch.device(a.device if torch.cuda.is_available() else "cpu")
    main(a.out,dev)
