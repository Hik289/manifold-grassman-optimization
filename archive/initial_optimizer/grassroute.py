"""
GrassRoute: Online Gradient Routing on the Grassmann Manifold.

实现 README §3 的主算法（严格按 §3.2/§3.3 公式），作为 torch.optim.Optimizer。

对每个被路由的参数矩阵 W ∈ R^{m×d}（右维 d 上取子空间 U ∈ Gr(r,d)）:
  Z = G U                         # 低维坐标 (m×r)
  G_par = Z U^T ; G_perp = G-G_par
  低维 Adam 作用于 Z -> D_par     # (m×r)
  残差归一化 SGD 作用于 G_perp    # (m×d)
  W <- (1-ηλ)W - η_par D_par U^T - η_perp D_perp
  子空间: C=G^T G, Q=(I-UU^T)CU (= G^T Z - U(U^T(G^T Z)))
          H=Π_U[μ Π_U(H) + (1-μ)Q] ; U_new=qf(U+ρH)
          R=Polar(U_new^T U) ; M<-M R^T ; V<-0 ; U<-U_new

非路由参数走 fallback（AdamW 或 SGD-M），用于整模型训练时的其余参数。

路由约定: 参数 p 的自然形状被 reshape 成 (m, -1)，右维 d=prod(shape[1:])。
  Linear (out,in): d=in ; Conv (out,in,kh,kw): d=in*kh*kw（EDA §2 约定一致）。
FP32 子空间运算（README §3.5）。
"""
import torch
import math


def _polar_orthogonal(A):
    """返回 A 的极分解正交因子: A = Q P, Q = U_s V_s^T (via SVD)。A: (r×r)。"""
    U_s, _, Vh = torch.linalg.svd(A, full_matrices=False)
    return U_s @ Vh


def _qf(A):
    """thin QR 的正交因子（列正交，且符号规范化使对角为正，保证唯一性/稳定）。"""
    Q, R = torch.linalg.qr(A)
    # 符号规范化: 使 R 对角非负（qf 的标准定义，保证 retraction 连续）
    diag_sign = torch.sign(torch.diagonal(R))
    diag_sign[diag_sign == 0] = 1.0
    return Q * diag_sign.unsqueeze(0)


class GrassRoute(torch.optim.Optimizer):
    def __init__(self, params,
                 lr=1e-3,                # η_par (主子空间参数 LR)
                 lr_perp=None,           # η_perp (残差 LR); 默认 = lr
                 grassmann_lr=1e-3,      # ρ (子空间步长)
                 rank=16,
                 mu=0.9,                 # Grassmann momentum
                 beta1=0.9, beta2=0.999,
                 eps_adam=1e-8, eps_res=1e-12,
                 s_min=0.0, s_max=10.0,  # 残差 scale clip
                 weight_decay=0.0,
                 routed=True,            # 该 group 是否走 Grassmann 路由
                 fallback="adamw",       # 非路由参数: "adamw" | "sgd"
                 momentum=0.9,           # SGD-M fallback 动量
                 sub_every=1,            # 子空间更新频率 (A9=5, A10=25); 1=每步
                 grassmann_momentum=True,  # A3: 关闭 Grassmann momentum
                 use_residual=True,      # A4: 关闭残差分支
                 residual_adam=False,    # A5: 残差用 Adam 而非 SGD
                 main_sgd=False,         # A6: 主分支用 SGD 而非 Adam
                 procrustes=True,        # A7: 关闭 Procrustes 对齐
                 reset_v=None,           # 向后兼容: True->hard, False->none (优先用 reset_mode)
                 reset_mode="procrustes",  # 二阶矩处理: hard(A8,§3.5硬reset) / none / decay / procrustes
                 v_decay=0.5,            # reset_mode=decay 的衰减因子 (保留部分曲率新鲜度)
                 reset_v_thresh=0.0,     # hard 模式: 仅子空间旋转超此阈值才 reset (0=每步, 忠实原spec)
                 normalize_residual=True,  # A12: 关闭残差归一化
                 normalize_cov=True):    # 子空间 ascent 方向按 tr(C)=||G||_F^2 归一 (步长稳定, 不改方向)
        if lr_perp is None:
            lr_perp = lr
        # 向后兼容: reset_v 显式给定则覆盖 reset_mode
        if reset_v is True:
            reset_mode = "hard"
        elif reset_v is False:
            reset_mode = "none"
        defaults = dict(lr=lr, lr_perp=lr_perp, grassmann_lr=grassmann_lr,
                        reset_mode=reset_mode, v_decay=v_decay,
                        rank=rank, mu=mu, beta1=beta1, beta2=beta2,
                        eps_adam=eps_adam, eps_res=eps_res, s_min=s_min, s_max=s_max,
                        weight_decay=weight_decay, routed=routed,
                        fallback=fallback, momentum=momentum,
                        sub_every=sub_every, grassmann_momentum=grassmann_momentum,
                        use_residual=use_residual, residual_adam=residual_adam,
                        main_sgd=main_sgd, procrustes=procrustes,
                        reset_v_thresh=reset_v_thresh,
                        normalize_residual=normalize_residual,
                        normalize_cov=normalize_cov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _init_routed_state(self, state, p, r, dev, dtype):
        m = p.shape[0]
        d = int(torch.tensor(p.shape[1:]).prod().item()) if p.dim() > 1 else p.shape[0]
        # U0 随机正交初始化 (d×r)，seed 由外部 torch.manual_seed 控制
        A = torch.randn(d, r, device=dev, dtype=torch.float32)
        U = _qf(A)
        state["U"] = U
        state["M"] = torch.zeros(m, r, device=dev, dtype=torch.float32)
        state["V"] = torch.zeros(m, r, device=dev, dtype=torch.float32)
        state["H"] = torch.zeros(d, r, device=dev, dtype=torch.float32)
        state["step"] = 0
        state["v_step"] = 0        # V 的局部年龄 (距上次 reset 的步数); 主算法每步 reset -> 恒为1
        state["sub_ctr"] = 0       # 子空间更新计数器 (支持 A9/A10 每 k 步更新)
        # 诊断量（供训练脚本逐 epoch 读取；每步覆盖为最近一次）
        state["diag"] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            routed = group["routed"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if routed and p.dim() >= 2:
                    self._step_routed(group, p)
                else:
                    self._step_fallback(group, p)
        return loss

    @torch.no_grad()
    def _step_routed(self, group, p):
        st = self.state[p]
        m = p.shape[0]
        d = int(torch.tensor(p.shape[1:]).prod().item())
        r = min(group["rank"], d, m if m >= 1 else d)  # r<=d 必须; 允许 r<=m 不强制
        r = min(group["rank"], d)
        dev = p.device
        if "U" not in st:
            self._init_routed_state(st, p, r, dev, p.dtype)
        st["step"] += 1
        st["v_step"] += 1
        t = st["step"]
        tv = st["v_step"]                            # V 的局部年龄 (距上次 reset)

        U = st["U"]                                  # (d,r) fp32
        M, V, H = st["M"], st["V"], st["H"]
        b1, b2 = group["beta1"], group["beta2"]
        eps_a, eps_r = group["eps_adam"], group["eps_res"]

        G = p.grad.reshape(m, d).float()             # (m,d) fp32
        W = p.data.reshape(m, d)

        # ---- Step 1: 路由梯度 ----
        Z = G @ U                                    # (m,r)
        G_par = Z @ U.T                              # (m,d)
        G_perp = G - G_par                           # (m,d)

        # ---- Step 2: 低维 Adam (主分支) ----
        M.mul_(b1).add_(Z, alpha=1 - b1)
        V.mul_(b2).addcmul_(Z, Z, value=1 - b2)
        if group["main_sgd"]:                        # A6: 主分支用 SGD (动量已在 M 里)
            D_par = M / (1 - b1 ** t)
        else:
            M_hat = M / (1 - b1 ** t)
            # V 的 bias correction 用局部年龄 tv (主算法每步 reset -> tv=1)
            V_hat = V / (1 - b2 ** tv)
            D_par = M_hat / (V_hat.sqrt() + eps_a)   # (m,r)

        # ---- Step 3: 残差分支 ----
        gnorm = G.norm()
        gpnorm = G_perp.norm()
        if not group["use_residual"]:                # A4: 无残差分支
            D_perp = torch.zeros_like(G_perp)
            scale = torch.tensor(0.0, device=dev)
        elif group["residual_adam"]:                 # A5: 残差用 Adam
            if "M_perp" not in st:
                st["M_perp"] = torch.zeros_like(G_perp)
                st["V_perp"] = torch.zeros_like(G_perp)
                st["perp_step"] = 0
            st["perp_step"] += 1
            tp = st["perp_step"]
            st["M_perp"].mul_(b1).add_(G_perp, alpha=1 - b1)
            st["V_perp"].mul_(b2).addcmul_(G_perp, G_perp, value=1 - b2)
            mph = st["M_perp"] / (1 - b1 ** tp)
            vph = st["V_perp"] / (1 - b2 ** tp)
            D_perp = mph / (vph.sqrt() + eps_a)
            scale = torch.tensor(1.0, device=dev)
        elif group["normalize_residual"]:            # 主算法: 范数匹配归一化 SGD
            scale = (gnorm / (gpnorm + eps_r)).clamp(group["s_min"], group["s_max"])
            D_perp = scale * G_perp
        else:                                        # A12: 无归一化, 裸残差
            scale = torch.tensor(1.0, device=dev)
            D_perp = G_perp

        # ---- Step 4: 更新参数 ----
        eta_par = group["lr"]
        eta_perp = group["lr_perp"]
        wd = group["weight_decay"]
        if wd != 0:
            W.mul_(1 - eta_par * wd)
        W.add_((D_par @ U.T), alpha=-eta_par)
        if group["use_residual"]:
            W.add_(D_perp, alpha=-eta_perp)

        # ---- Step 5: 更新 Grassmann 子空间 (每 sub_every 步) ----
        st["sub_ctr"] += 1
        do_sub = (st["sub_ctr"] % group["sub_every"] == 0)
        GtZ = G.T @ Z                                # (d,r) = C U
        if group["normalize_cov"]:                   # Ĉ=C/||G||_F^2: 归一步长, 特征向量不变
            GtZ = GtZ / (gnorm ** 2 + eps_r)
        Q = GtZ - U @ (U.T @ GtZ)                    # tangent (I-UU^T)ĈU
        if do_sub:
            if group["grassmann_momentum"]:          # H = Π_U[ μ Π_U(H) + (1-μ) Q ]
                H_proj = H - U @ (U.T @ H)
                H_new = group["mu"] * H_proj + (1 - group["mu"]) * Q
                H_new = H_new - U @ (U.T @ H_new)
            else:                                    # A3: 无 Grassmann momentum
                H_new = Q
            rho = group["grassmann_lr"]
            U_new = _qf(U + rho * H_new)             # QR retraction
            # 子空间实际旋转量 (sin 主角) = ||U_new - U(U^T U_new)||
            rot = float((U_new - U @ (U.T @ U_new)).norm().item())
            Rp = _polar_orthogonal(U_new.T @ U)      # Procrustes 旋转 (r×r)
            if group["procrustes"]:                  # 一阶矩对齐
                M.copy_(M @ Rp.T)
            # 二阶矩处理: 4 种模式 (Director 裁决三变体对比 + A8 硬 reset)
            mode = group["reset_mode"]
            if mode == "hard":                       # A8 / §3.5: 硬 reset (仅旋转超阈值)
                if rot > group["reset_v_thresh"]:
                    V.zero_(); st["v_step"] = 0
                    st["diag"]["reset_count"] = st["diag"].get("reset_count", 0) + 1
            elif mode == "decay":                    # §Risks: 指数衰减 reset (保留部分曲率新鲜度)
                V.mul_(group["v_decay"])
            elif mode == "procrustes":               # V 随坐标系旋转 (非清零): V <- (V @ R.T^2 的正部近似)
                # V 是 elementwise E[Z²] (m×r, 非负). 旋转坐标系需先取 sqrt(能量幅度)再旋转再平方.
                Vroot = V.sqrt()                     # ~|Z| 的 EMA 幅度
                Vrot = Vroot @ Rp.T                  # 随子空间旋转到新坐标
                V.copy_(Vrot * Vrot)                 # 回到二阶矩 (非负保证)
            # mode == "none": 完全不动 V (= 旧 reset_v=False)
            st["U"] = U_new
            st["H"] = H_new
        else:
            H_new = H
            U_new = U

        # ---- 诊断量（供 §5.3 逐 epoch 记录）----
        with torch.no_grad():
            captured = float((Z.norm() ** 2 / (gnorm ** 2 + 1e-30)).item())
            # eigengap: 用 C 的谱 (小 d 才算; 大 d 用 Z 的 gram 近似 top 谱)
            diag = st["diag"]
            diag["total_grad_norm"] = float(gnorm.item())
            diag["routed_grad_norm"] = float(G_par.norm().item())
            diag["residual_grad_norm"] = float(gpnorm.item())
            diag["captured_energy"] = captured
            diag["grassmann_grad_norm"] = float(Q.norm().item())
            diag["grassmann_momentum_norm"] = float(H_new.norm().item())
            diag["residual_scale"] = float(scale.item())
            diag["retraction_error"] = float((U_new.T @ U_new - torch.eye(r, device=dev)).norm().item())
            # tangency: H_new 相对当前 U_t 是否切向 (算法不变量, U 更新前的量)
            diag["tangency_error"] = float((U.T @ H_new).norm().item())
            diag["rank"] = r
            diag["d"] = d
            diag["m"] = m
            # eigengap / rank_tail / curvature_proxy (廉价: 用 Z 的 r×r gram 近似主子空间谱)
            # Z^T Z (r×r) 的特征值 ~ C 在 U 张成子空间上的能量分布
            try:
                ZtZ = (Z.T @ Z)
                evals = torch.linalg.eigvalsh(ZtZ).clamp_min(0)   # 升序, r 个
                evals_desc = evals.flip(0)
                lam1 = float(evals_desc[0].item()) + 1e-30
                # eigengap: (λ_{r-1}-λ_r)/λ1 在子空间内最后一个 gap (代理 Δ_r)
                if r >= 2:
                    diag["eigengap"] = float(((evals_desc[-2] - evals_desc[-1]) / lam1).item())
                else:
                    diag["eigengap"] = 0.0
                # rank_tail: 残差能量占总能量 (被路由到 SGD 的谱尾 τ_r 代理)
                diag["rank_tail"] = float((gpnorm ** 2 / (gnorm ** 2 + 1e-30)).item())
                # curvature_proxy: ||H||·ρ 的量级 (Grassmann 步长, 关联曲率修正项)
                diag["curvature_proxy"] = float((group["grassmann_lr"] * H_new.norm()).item())
            except Exception:
                pass

    @torch.no_grad()
    def _step_fallback(self, group, p):
        """非路由参数: AdamW 或 SGD-M。"""
        st = self.state[p]
        g = p.grad
        wd = group["weight_decay"]
        if group["fallback"] == "sgd":
            if "buf" not in st:
                st["buf"] = torch.zeros_like(p)
            buf = st["buf"]
            if wd != 0:
                g = g.add(p, alpha=wd)
            buf.mul_(group["momentum"]).add_(g)
            p.add_(buf, alpha=-group["lr"])
        else:  # adamw
            if "step" not in st:
                st["step"] = 0
                st["m"] = torch.zeros_like(p)
                st["v"] = torch.zeros_like(p)
            st["step"] += 1
            t = st["step"]
            b1, b2 = group["beta1"], group["beta2"]
            if wd != 0:
                p.mul_(1 - group["lr"] * wd)
            st["m"].mul_(b1).add_(g, alpha=1 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            mh = st["m"] / (1 - b1 ** t)
            vh = st["v"] / (1 - b2 ** t)
            p.addcdiv_(mh, vh.sqrt() + group["eps_adam"], value=-group["lr"])

    # ---- 工具: 收集所有路由层诊断（训练脚本调用）----
    @torch.no_grad()
    def collect_diagnostics(self):
        out = []
        for group in self.param_groups:
            if not group["routed"]:
                continue
            for p in group["params"]:
                st = self.state.get(p, {})
                if "diag" in st and st["diag"]:
                    out.append(dict(st["diag"]))
        return out

    @torch.no_grad()
    def state_memory_bytes(self):
        """自适应状态内存（bytes）: 路由层 M,V,H,U + 残差Adam的 M_perp,V_perp; fallback adamw 层 m,v。
        注: A5(residual_adam=True) 的 M_perp/V_perp 是 O(m·d) 残差自适应状态, 牺牲 G3 内存优势。"""
        total = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                for k in ("M", "V", "H", "U", "M_perp", "V_perp", "m", "v", "buf"):
                    if k in st and torch.is_tensor(st[k]):
                        total += st[k].numel() * st[k].element_size()
        return total
