"""ISOLATE/TRACE 探针: 定位 anchor_1 发散根因。
1. 直接测: 训练中 C=G^T G 的 top-16 特征子空间与植入 U_star 的夹角
2. Oracle: 冻结 U=U_star, 只跑 Adam-on-Z + residual, 看是否收敛
3. 对照: GrassRoute 学到的 U 与 C-top16 / U_star 三者关系
"""
import torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grassroute import GrassRoute
from anchor1_matreg import make_data, principal_angles_R
SEED=42; DEV="cuda" if torch.cuda.is_available() else "cpu"

d,m,r_true=256,64,16
X,Y,W_star,U_star=make_data(d=d,m=m,r_true=r_true)

def C_top_angle(W):
    G=(2.0/X.shape[0])*((X@W.T-Y).T@X)   # (m,d)
    C=G.T@G
    w,V=torch.linalg.eigh(C)
    Utop=V[:,-r_true:]                    # top-16
    R=principal_angles_R(Utop,U_star)
    return R/(math.pi/2), G.norm().item()

# 在随机 W 和一个"部分训练"W 上测 C-top16 vs U_star
torch.manual_seed(SEED)
W0=0.01*torch.randn(m,d,device=DEV)
r0,gn0=C_top_angle(W0)
print(f"[TRACE] 随机W: C-top16 vs U_star  R/halfpi={r0:.4f}  ||G||={gn0:.3e}")

# oracle: 冻结 U=U_star, 只 Adam-on-Z + residual (routed=True 但不更新子空间)
torch.manual_seed(SEED)
W=torch.nn.Parameter(0.01*torch.randn(m,d,device=DEV))
opt=GrassRoute([W],lr=3e-3,grassmann_lr=0.0,rank=r_true,mu=0.9)  # rho=0 -> U 不动
# 强制 U=U_star
opt.step  # noop
# 手动注入初始 state
import torch as T
_=W.sum(); (W.sum()).backward(); opt.zero_grad()  # 触发 grad 存在
# 先跑一步初始化 state
opt.zero_grad(); ((X@W.T-Y).pow(2).mean()).backward();
# init state then override U
p=W
# manual init
from grassroute import _qf
st=opt.state[p]
st["U"]=U_star.clone(); st["M"]=T.zeros(m,r_true,device=DEV); st["V"]=T.zeros(m,r_true,device=DEV)
st["H"]=T.zeros(d,r_true,device=DEV); st["step"]=0; st["v_step"]=0; st["sub_ctr"]=0; st["diag"]={}
for ep in range(2000):
    cos=0.5*(1+math.cos(math.pi*ep/2000)); f=0.01+0.99*cos
    opt.param_groups[0]["lr"]=3e-3*f; opt.param_groups[0]["lr_perp"]=3e-3*f
    opt.zero_grad(); loss=(X@W.T-Y).pow(2).mean(); loss.backward(); opt.step()
    # 保持 U 冻结在 U_star (rho=0 已保证, 但 reset_v 会动 V; U 不变)
print(f"[ISOLATE] oracle U=U_star 冻结: final_loss={loss.item():.4e}")
rf,gnf=C_top_angle(W.data)
print(f"[TRACE] oracle训练后: C-top16 vs U_star R/halfpi={rf:.4f} ||G||={gnf:.3e}")
