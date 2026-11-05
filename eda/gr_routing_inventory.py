"""
GrassRoute EDA — routing inventory (goal 1).
Instantiate each task-family model and enumerate every parameter matrix, marking
which are routable (dim>=2, right-dim d in [min,max]) and mapping to README sec 6
routed-layer spec. Confirms 'which W matrices get routed' matches the algorithm spec.
"""
import os, sys, json
import numpy as np, torch, torch.nn as nn
import torch.serialization as _ts

MIN_D, MAX_D = 8, 6000   # routing sensible range (r<<d, and avoid tiny/huge)

def enumerate_model(model, spec_note=""):
    rows=[]
    for name,p in model.named_parameters():
        if p.dim()<2:
            rows.append({"name":name,"shape":list(p.shape),"routable":False,"reason":"vector/scalar"})
            continue
        m=p.shape[0]; d=int(np.prod(p.shape[1:]))
        routable = (MIN_D<=d<=MAX_D)
        rows.append({"name":name,"shape":list(p.shape),"m":m,"d_right":d,
                     "routable":bool(routable),
                     "reason":("" if routable else f"d={d} out of [{MIN_D},{MAX_D}]")})
    return {"spec_note":spec_note,"n_params_total":len(rows),
            "n_routable":sum(r["routable"] for r in rows),"params":rows}

def main(out):
    inv={}
    from torchvision import models
    import timm
    from torch_geometric.nn import GCNConv, SAGEConv

    # CNN: ResNet-18 (README: classifier, 1x1 conv, selected flattened conv kernels)
    net=models.resnet18(num_classes=10); net.conv1=nn.Conv2d(3,64,3,1,1,bias=False); net.maxpool=nn.Identity()
    inv["cnn_resnet18"]=enumerate_model(net,"README6.2: fc(classifier), 1x1 conv (downsample), selected 3x3 conv kernels flattened (out,in*kh*kw)")

    # ViT-Tiny (README: q,k,v,o, MLP matrices)
    vit=timm.create_model("vit_tiny_patch16_224",pretrained=False,num_classes=10,img_size=32,patch_size=4)
    inv["vit_tiny"]=enumerate_model(vit,"README6.3: qkv(fused q,k,v), attn.proj(o), mlp.fc1/fc2")

    # Graph GCN
    class GCN(nn.Module):
        def __init__(s): super().__init__(); s.c1=GCNConv(1433,128); s.c2=GCNConv(128,7)
        def forward(s,x): return x
    inv["graph_gcn"]=enumerate_model(GCN(),"README6.4: GCNConv weight = .lin.weight (out,in)")
    class SAGE(nn.Module):
        def __init__(s): super().__init__(); s.c1=SAGEConv(128,256); s.c2=SAGEConv(256,40)
        def forward(s,x): return x
    inv["graph_graphsage"]=enumerate_model(SAGE(),"README6.4: SAGEConv lin_l/lin_r weights")

    # PINN (SciML)
    class PINN(nn.Module):
        def __init__(s,w=128):
            super().__init__(); s.l1=nn.Linear(2,w); s.l2=nn.Linear(w,w); s.l3=nn.Linear(w,w); s.l4=nn.Linear(w,w); s.out=nn.Linear(w,1)
        def forward(s,x): return x
    inv["sciml_pinn"]=enumerate_model(PINN(),"README6.5: all hidden Linear weights; note l1 (d=2) & out (d=128,m=1) edge cases")

    # RecSys MF
    class MF(nn.Module):
        def __init__(s,nu=6040,ni=3706,K=64):
            super().__init__(); s.uemb=nn.Embedding(nu,K); s.iemb=nn.Embedding(ni,K); s.ub=nn.Embedding(nu,1); s.ib=nn.Embedding(ni,1)
        def forward(s,x): return x
    inv["recsys_mf"]=enumerate_model(MF(),"README6.6: embedding matrices (nu x K, ni x K) -> right subspace = K=64")

    # MLP / continual
    class MLP(nn.Module):
        def __init__(s): super().__init__(); s.f1=nn.Linear(784,512); s.f2=nn.Linear(512,512); s.f3=nn.Linear(512,256); s.head=nn.Linear(256,10)
        def forward(s,x): return x
    inv["mlp_deep"]=enumerate_model(MLP(),"README6.1: all Linear weights")

    # Matrix regression
    class MR(nn.Module):
        def __init__(s,m=64,d=256): super().__init__(); s.W=nn.Parameter(torch.zeros(m,d))
        def forward(s,x): return x
    inv["matreg"]=enumerate_model(MR(),"README6.1: single weight matrix W (m x d)")

    os.makedirs(out,exist_ok=True)
    json.dump(inv,open(os.path.join(out,"routing_inventory.json"),"w"),indent=1)
    # print compact
    for fam,d in inv.items():
        print(f"\n### {fam}  ({d['n_routable']}/{d['n_params_total']} routable)  [{d['spec_note']}]")
        for r in d["params"]:
            if r["routable"]:
                print(f"   ROUTE {r['name']:34s} {str(r['shape']):22s} m={r['m']} d={r['d_right']}")
        # show a few non-routable notable
        nonr=[r for r in d["params"] if not r["routable"] and r["shape"] and len(r["shape"])>=2]
        for r in nonr[:4]:
            print(f"   skip  {r['name']:34s} {str(r['shape']):22s} ({r['reason']})")
    print("\n[saved] routing_inventory.json")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True)
    a=ap.parse_args(); main(a.out)
