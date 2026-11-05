"""
6 条 baseline 的统一封装 (README §7 + 人类指令加 B6 SubTrack++):
  B1 SGD-M       : 全参数 SGD + momentum (无子空间)
  B2 AdamW       : 全参数 AdamW (无子空间, 性能上界)
  B3 Fixed-random: 固定随机子空间, Adam-in-sub + SGD-residual (不更新 U)
  B4 Periodic-SVD: 周期 SVD 重估子空间, Adam-in-sub + SGD-residual
  B5 GrassRoute  : 在线 Grassmann (grassroute.py)
  B6 SubTrack++  : Grassmann 测地线 rank-1 步 + Projection-aware Adam + Recovery Scaling
                   (仅在 7 传统任务族对比, 绝不 LLM —— 人类明确指令)

B3/B4/B6 复用 GrassRoute 的路由骨架 (routing + 双分支), 只替换子空间更新规则:
  - B3: 不更新 (sub_every=inf 等价, 这里用 freeze_subspace)
  - B4: 每 k 步用 torch.linalg.svd(C) 取 top-r 替换 U
  - B6: Grassmann 测地线步 (exact Exp via SVD) + recovery scaling on residual

所有 baseline 暴露相同接口: .step(), .collect_diagnostics(), .state_memory_bytes()
以便训练脚本统一逐 epoch 记录 §5.3 的 20 项。
"""
import torch
import math
from grassroute import GrassRoute, _qf, _polar_orthogonal


def build_optimizer(baseline_id, params, rank=16, lr=3e-3, grassmann_lr=0.1,
                    weight_decay=0.0, svd_every=25, **kw):
    """工厂函数: 按 baseline_id 返回配置好的 optimizer。
    params 可以是 list[Tensor] (单参数矩阵测试) 或 param_groups。"""
    bid = baseline_id.upper()
    if bid in ("B1", "SGD", "SGD-M"):
        return _WrapTorch(torch.optim.SGD(params, lr=lr, momentum=kw.get("momentum", 0.9),
                                          weight_decay=weight_decay), kind="B1")
    if bid in ("B2", "ADAMW"):
        return _WrapTorch(torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                            betas=(kw.get("beta1", 0.9), kw.get("beta2", 0.999))),
                          kind="B2")
    if bid in ("B3", "FIXED", "FIXED-RANDOM"):
        return GrassRoute(params, lr=lr, grassmann_lr=0.0, rank=rank,
                          weight_decay=weight_decay, **_gr_kw(kw))
    if bid in ("B4", "PERIODIC-SVD", "SVD"):
        return PeriodicSVDRouter(params, lr=lr, rank=rank, weight_decay=weight_decay,
                                 svd_every=svd_every, **_gr_kw(kw))
    if bid in ("B5", "GRASSROUTE"):
        return GrassRoute(params, lr=lr, grassmann_lr=grassmann_lr, rank=rank,
                          weight_decay=weight_decay, **_gr_kw(kw))
    if bid in ("B6", "SUBTRACK", "SUBTRACK++"):
        return SubTrackPP(params, lr=lr, grassmann_lr=grassmann_lr, rank=rank,
                          weight_decay=weight_decay, **_gr_kw(kw))
    raise ValueError(f"unknown baseline {baseline_id}")


def _gr_kw(kw):
    """从通用 kw 中过滤出 GrassRoute 接受的参数。"""
    allowed = {"lr_perp", "mu", "beta1", "beta2", "eps_adam", "eps_res",
               "s_min", "s_max", "fallback", "momentum", "sub_every",
               "grassmann_momentum", "use_residual", "residual_adam", "main_sgd",
               "procrustes", "reset_v", "reset_mode", "v_decay", "reset_v_thresh",
               "normalize_residual", "normalize_cov", "routed"}
    return {k: v for k, v in kw.items() if k in allowed}


class _WrapTorch:
    """把原生 torch optimizer 包装成统一诊断接口 (B1/B2 无子空间, 诊断量为空/内存=全 state)。"""
    def __init__(self, opt, kind):
        self.opt = opt
        self.kind = kind
        self.param_groups = opt.param_groups
        self.state = opt.state

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        return self.opt.step(closure)

    @torch.no_grad()
    def collect_diagnostics(self):
        # B1/B2 无子空间路由; 仍记录梯度范数 (由训练脚本从 param.grad 直接算更准, 这里返回空)
        return []

    @torch.no_grad()
    def state_memory_bytes(self):
        total = 0
        for st in self.opt.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        return total


class PeriodicSVDRouter(GrassRoute):
    """B4: 周期 SVD 路由。继承 GrassRoute 的路由/双分支骨架,
    但子空间不走 Grassmann gradient, 而是每 svd_every 步用 top-r SVD(C) 重估。"""
    def __init__(self, params, svd_every=25, **kw):
        kw.setdefault("grassmann_lr", 0.0)
        super().__init__(params, **kw)
        for g in self.param_groups:
            g["svd_every"] = svd_every

    @torch.no_grad()
    def _step_routed(self, group, p):
        st = self.state[p]
        m = p.shape[0]
        d = int(torch.tensor(p.shape[1:]).prod().item())
        r = min(group["rank"], d)
        dev = p.device
        if "U" not in st:
            self._init_routed_state(st, p, r, dev, p.dtype)
            st["svd_ctr"] = 0
        st["step"] += 1
        st["v_step"] += 1
        t = st["step"]; tv = st["v_step"]
        U = st["U"]; M, V = st["M"], st["V"]
        b1, b2 = group["beta1"], group["beta2"]
        eps_a, eps_r = group["eps_adam"], group["eps_res"]
        G = p.grad.reshape(m, d).float()
        W = p.data.reshape(m, d)
        Z = G @ U; G_par = Z @ U.T; G_perp = G - G_par
        M.mul_(b1).add_(Z, alpha=1 - b1)
        V.mul_(b2).addcmul_(Z, Z, value=1 - b2)
        M_hat = M / (1 - b1 ** t); V_hat = V / (1 - b2 ** tv)
        D_par = M_hat / (V_hat.sqrt() + eps_a)
        gnorm = G.norm(); gpnorm = G_perp.norm()
        scale = (gnorm / (gpnorm + eps_r)).clamp(group["s_min"], group["s_max"])
        D_perp = scale * G_perp
        eta = group["lr"]; wd = group["weight_decay"]
        if wd != 0:
            W.mul_(1 - eta * wd)
        W.add_(D_par @ U.T, alpha=-eta)
        W.add_(D_perp, alpha=-group["lr_perp"])
        # 周期 SVD 重估 (稳健: PINN 等近 rank-1 任务 C 高度退化, eigh 可能不收敛)
        st["svd_ctr"] += 1
        if st["svd_ctr"] % group["svd_every"] == 0:
            C = G.T @ G
            try:
                _, Vec = torch.linalg.eigh(C)
                U_new = Vec[:, -r:].contiguous()
            except Exception:
                # 回退: C 加 jitter 后重试 eigh (PINN 近rank-1 时 C 退化); 保证输出 d×r
                try:
                    jit = 1e-6 * C.diagonal().abs().mean().clamp_min(1e-12)
                    _, Vec = torch.linalg.eigh(C + jit * torch.eye(d, device=C.device))
                    U_new = Vec[:, -r:].contiguous()
                except Exception:
                    U_new = _qf(C @ U)             # 一步幂迭代兜底, 保证 d×r
            # 保证列数=r (SVD/退化可能给出 <r 列)
            if U_new.shape[1] < r:
                U_new = _qf(torch.cat([U_new, U[:, U_new.shape[1]:]], dim=1))
            Rp = _polar_orthogonal(U_new.T @ U)
            M.copy_(M @ Rp.T)
            st["U"] = U_new
        # 诊断
        with torch.no_grad():
            dd = st["diag"]
            dd["total_grad_norm"] = float(gnorm.item())
            dd["routed_grad_norm"] = float(G_par.norm().item())
            dd["residual_grad_norm"] = float(gpnorm.item())
            dd["captured_energy"] = float((Z.norm() ** 2 / (gnorm ** 2 + 1e-30)).item())
            dd["residual_scale"] = float(scale.item())
            dd["rank"] = r; dd["d"] = d; dd["m"] = m


class SubTrackPP(GrassRoute):
    """B6: SubTrack++ (Rajabi et al. 2025) 简化实现 (非 LLM, 仅传统任务对比):
    - Grassmann 测地线步 (exact Exp map via SVD of tangent) 更新子空间
    - Projection-aware Adam (同 GrassRoute 主分支)
    - Recovery Scaling: 残差按 ||G||/||G_par|| 缩放 (而非 GrassRoute 的双分支归一化 SGD)
    与 GrassRoute 区别: (a) 测地线 Exp vs QR retraction; (b) recovery scaling vs 无状态残差 SGD。
    """
    @torch.no_grad()
    def _step_routed(self, group, p):
        st = self.state[p]
        m = p.shape[0]
        d = int(torch.tensor(p.shape[1:]).prod().item())
        r = min(group["rank"], d)
        dev = p.device
        if "U" not in st:
            self._init_routed_state(st, p, r, dev, p.dtype)
        st["step"] += 1; st["v_step"] += 1
        t = st["step"]; tv = st["v_step"]
        U = st["U"]; M, V, H = st["M"], st["V"], st["H"]
        b1, b2 = group["beta1"], group["beta2"]
        eps_a, eps_r = group["eps_adam"], group["eps_res"]
        G = p.grad.reshape(m, d).float()
        W = p.data.reshape(m, d)
        Z = G @ U; G_par = Z @ U.T; G_perp = G - G_par
        M.mul_(b1).add_(Z, alpha=1 - b1)
        V.mul_(b2).addcmul_(Z, Z, value=1 - b2)
        M_hat = M / (1 - b1 ** t); V_hat = V / (1 - b2 ** tv)
        D_par = M_hat / (V_hat.sqrt() + eps_a)
        # Recovery Scaling: 用 ||G||/||G_par|| 补偿投影丢失的能量 (SubTrack++ 核心)
        gnorm = G.norm(); gparnorm = G_par.norm()
        rec = (gnorm / (gparnorm + eps_r)).clamp(group["s_min"], group["s_max"])
        eta = group["lr"]; wd = group["weight_decay"]
        if wd != 0:
            W.mul_(1 - eta * wd)
        W.add_((D_par @ U.T), alpha=-eta * float(rec))
        # 子空间: Grassmann 测地线 Exp map (rank-r), 用协方差归一 ascent 方向
        GtZ = G.T @ Z
        if group["normalize_cov"]:
            GtZ = GtZ / (gnorm ** 2 + eps_r)
        Q = GtZ - U @ (U.T @ GtZ)
        H_new = group["mu"] * (H - U @ (U.T @ H)) + (1 - group["mu"]) * Q
        H_new = H_new - U @ (U.T @ H_new)
        rho = group["grassmann_lr"]
        # exact Exp_U(rho H) via thin SVD of tangent
        Qm, S, Pt = torch.linalg.svd(rho * H_new, full_matrices=False)
        cosS = torch.diag(torch.cos(S)); sinS = torch.diag(torch.sin(S))
        U_new = U @ Pt.T @ cosS @ Pt + Qm @ sinS @ Pt
        U_new = _qf(U_new)
        Rp = _polar_orthogonal(U_new.T @ U)
        M.copy_(M @ Rp.T)
        st["U"] = U_new; st["H"] = H_new
        with torch.no_grad():
            dd = st["diag"]
            dd["total_grad_norm"] = float(gnorm.item())
            dd["routed_grad_norm"] = float(gparnorm.item())
            dd["residual_grad_norm"] = float(G_perp.norm().item())
            dd["captured_energy"] = float((Z.norm() ** 2 / (gnorm ** 2 + 1e-30)).item())
            dd["recovery_scale"] = float(rec)
            dd["grassmann_momentum_norm"] = float(H_new.norm().item())
            dd["rank"] = r; dd["d"] = d; dd["m"] = m
