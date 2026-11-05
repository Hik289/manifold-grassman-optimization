"""真实 anchor_1 (U 在线更新) 下确认修复: reset_v / rho-decay 组合。"""
import torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grassroute import GrassRoute
from anchor1_matreg import make_data, principal_angles_R
SEED=42; DEV="cuda"
d,m,r_true=256,64,16
X,Y,W_star,U_star=make_data(d=d,m=m,r_true=r_true)

def run(reset_v, rho0, rho_decay, epochs=2000):
    torch.manual_seed(SEED)
    W=torch.nn.Parameter(0.01*torch.randn(m,d,device=DEV))
    opt=GrassRoute([W],lr=3e-3,lr_perp=3e-3,grassmann_lr=rho0,rank=r_true,mu=0.9,reset_v=reset_v)
    best=1e30
    for ep in range(epochs):
        f=0.01+0.99*0.5*(1+math.cos(math.pi*ep/epochs))
        opt.param_groups[0]["lr"]=3e-3*f; opt.param_groups[0]["lr_perp"]=3e-3*f
        opt.param_groups[0]["grassmann_lr"]=rho0*(f if rho_decay else 1.0)
        opt.zero_grad(); loss=(X@W.T-Y).pow(2).mean(); loss.backward(); opt.step()
        best=min(best,loss.item())
    U=opt.state[W]["U"]
    R=principal_angles_R(U,U_star)/(math.pi/2)
    return loss.item(),best,R

for rv in [True,False]:
    for rd in [False,True]:
        fl,bl,R=run(rv,0.1,rd)
        print(f"reset_v={rv} rho_decay={rd}: final={fl:.4e} best={bl:.4e} finalR/halfpi={R:.4f}")
