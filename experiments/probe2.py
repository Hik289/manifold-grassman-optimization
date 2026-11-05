"""隔离参数更新: oracle U=U_star 冻结, 扫 lr / 残差 / V重置 组合。"""
import torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grassroute import GrassRoute
from anchor1_matreg import make_data
SEED=42; DEV="cuda"
d,m,r_true=256,64,16
X,Y,W_star,U_star=make_data(d=d,m=m,r_true=r_true)

def oracle(lr, use_residual, reset_v, epochs=2000, decay=True):
    torch.manual_seed(SEED)
    W=torch.nn.Parameter(0.01*torch.randn(m,d,device=DEV))
    opt=GrassRoute([W],lr=lr,lr_perp=lr,grassmann_lr=0.0,rank=r_true,mu=0.9,
                   use_residual=use_residual,reset_v=reset_v)
    opt.zero_grad(); ((X@W.T-Y).pow(2).mean()).backward()
    st=opt.state[W]; st["U"]=U_star.clone()
    st["M"]=torch.zeros(m,r_true,device=DEV); st["V"]=torch.zeros(m,r_true,device=DEV)
    st["H"]=torch.zeros(d,r_true,device=DEV); st["step"]=0; st["v_step"]=0; st["sub_ctr"]=0; st["diag"]={}
    best=1e30
    for ep in range(epochs):
        f=(0.01+0.99*0.5*(1+math.cos(math.pi*ep/epochs))) if decay else 1.0
        opt.param_groups[0]["lr"]=lr*f; opt.param_groups[0]["lr_perp"]=lr*f
        opt.zero_grad(); loss=(X@W.T-Y).pow(2).mean(); loss.backward(); opt.step()
        best=min(best,loss.item())
    return loss.item(), best

for lr in [3e-3, 1e-3, 3e-4]:
    for res in [True, False]:
        for rv in [True, False]:
            fl,bl=oracle(lr,res,rv)
            print(f"lr={lr:.0e} residual={res} reset_v={rv}: final={fl:.4e} best={bl:.4e}")
