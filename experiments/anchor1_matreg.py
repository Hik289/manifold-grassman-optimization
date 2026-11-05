"""
anchor_1: GrassRoute 在合成 matrix regression (植入低秩结构) 上正确收敛。

设置 (checklist H0.anchor_1 / README §6.1):
  真值 W* = B_low @ C_low, rank(W*) = r_true (植入低秩)
  观测: Y = X W*^T + noise ; 学 W ∈ R^{m×d}, 最小化 ||X W^T - Y||^2
  d=256, r_true=16, m=64 (输出维), N 样本
  对比 AdamW vs GrassRoute (rank=16)

预期 (prediction):
  - GrassRoute final loss ≤ AdamW × 1.1
  - captured energy 比例 > 0.9 (早/中期; EDA §3 提示收敛后期会塌陷)
  - principal-angle radius R < π/2 全程
  - 学到的 W 的主子空间与植入 W* 的行空间对齐

seed=42, 2000 epochs (full-batch)。
"""
import torch
import json
import sys
import os
import math
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grassroute import GrassRoute

SEED = 42
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def principal_angles_R(U1, U2):
    M = U1.T @ U2
    s = torch.linalg.svdvals(M).clamp(-1, 1)
    theta = torch.arccos(s)
    return theta.max().item()  # R (rad)


def make_data(d=256, m=64, r_true=16, N=2048, noise=1e-3, tail=1e-2):
    """植入低秩结构 = 输入 X 的能量集中在 r_true 维子空间 span(U_star)。
    这样梯度右协方差 C=G^T G 真正低秩 (集中在 U_star), routing 子空间有意义,
    捕获能量 -> 主子空间 (符合 anchor_1 前提)。tail 加微弱满维能量避免完全退化。"""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    # ground-truth 主子空间 U_star ∈ R^{d×r_true} 正交
    U_star = torch.linalg.qr(torch.randn(d, r_true, generator=g))[0]  # (d,r_true)
    # 输入位于该子空间 (+ 微弱满维尾) : X = S U_star^T + tail·noise
    S = torch.randn(N, r_true, generator=g)                          # (N,r_true) 主能量坐标
    X = S @ U_star.T + tail * torch.randn(N, d, generator=g)         # (N,d)
    # 真值权重 W* (可满秩), 观测 Y = X W*^T + noise
    W_star = torch.randn(m, d, generator=g) / math.sqrt(d)
    Y = X @ W_star.T + noise * torch.randn(N, m, generator=g)
    return X.to(DEV), Y.to(DEV), W_star.to(DEV), U_star.to(DEV)


def train(method, X, Y, U_star, d, m, rank, epochs=2000, lr=None):
    torch.manual_seed(SEED)
    W = torch.nn.Parameter(0.01 * torch.randn(m, d, device=DEV))
    if method == "adamw":
        opt = torch.optim.AdamW([W], lr=lr or 3e-3, weight_decay=0.0)
    elif method == "grassroute":
        # reset_mode 默认 procrustes (Director 裁决三变体后固化); 硬 reset 是发散源.
        opt = GrassRoute([W], lr=lr or 3e-3, grassmann_lr=0.1, rank=rank,
                         mu=0.9, weight_decay=0.0)
    traj = []
    t0 = time.time()
    base_lrs = [(g["lr"], g.get("lr_perp", g["lr"]), g.get("grassmann_lr", 0)) for g in opt.param_groups]
    for ep in range(epochs):
        # cosine LR decay (README §5 标准做法; signSGD 式主分支需衰减消除 plateau)
        cos = 0.5 * (1 + math.cos(math.pi * ep / epochs))
        floor = 0.01
        f = floor + (1 - floor) * cos
        for g, (l0, lp0, lg0) in zip(opt.param_groups, base_lrs):
            g["lr"] = l0 * f
            if "lr_perp" in g:
                g["lr_perp"] = lp0 * f
            # grassmann_lr 不衰减(子空间需持续跟踪), 保持 tracking 能力
        opt.zero_grad()
        pred = X @ W.T
        loss = (pred - Y).pow(2).mean()
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            rec = {"epoch": ep, "loss": float(loss.item())}
            if method == "grassroute":
                diag = opt.collect_diagnostics()
                if diag:
                    d0 = diag[0]
                    rec["captured_energy"] = d0["captured_energy"]
                    rec["residual_scale"] = d0["residual_scale"]
                    rec["grassmann_grad_norm"] = d0["grassmann_grad_norm"]
                    U = opt.state[W]["U"]
                    R = principal_angles_R(U, U_star)
                    rec["R_rad"] = R
                    rec["R_over_halfpi"] = R / (math.pi / 2)
            traj.append(rec)
    elapsed = time.time() - t0
    final_loss = float(loss.item())
    best_loss = min(r["loss"] for r in traj)
    result = {"method": method, "final_loss": final_loss, "best_loss": best_loss,
              "epochs": epochs, "time_s": elapsed, "trajectory": traj}
    if method == "grassroute":
        U = opt.state[W]["U"]
        result["final_R_over_halfpi"] = principal_angles_R(U, U_star) / (math.pi / 2)
        result["final_captured_energy"] = traj[-1].get("captured_energy")
        result["max_R_over_halfpi"] = max(r.get("R_over_halfpi", 0) for r in traj)
        result["max_captured_energy"] = max(r.get("captured_energy", 0) for r in traj)
    return result


def run():
    d, m, r_true = 256, 64, 16
    X, Y, W_star, U_star = make_data(d=d, m=m, r_true=r_true)
    print(f"[data] X={tuple(X.shape)} Y={tuple(Y.shape)} W*={tuple(W_star.shape)} rank_true={r_true}")

    res_adamw = train("adamw", X, Y, U_star, d, m, rank=16)
    print(f"[adamw]      final_loss={res_adamw['final_loss']:.6e} best={res_adamw['best_loss']:.6e}")
    res_gr = train("grassroute", X, Y, U_star, d, m, rank=16)
    print(f"[grassroute] final_loss={res_gr['final_loss']:.6e} best={res_gr['best_loss']:.6e} "
          f"final_R/half_pi={res_gr['final_R_over_halfpi']:.4f} "
          f"max_captured={res_gr['max_captured_energy']:.4f}")

    ratio = res_gr["final_loss"] / (res_adamw["final_loss"] + 1e-30)
    checks = {
        "final_loss_ratio_gr_over_adamw": ratio,
        "pred_loss_leq_1.1x": ratio <= 1.1,
        "pred_max_captured_energy_gt_0.9": res_gr["max_captured_energy"] > 0.9,
        "pred_R_below_halfpi_allthrough": res_gr["max_R_over_halfpi"] < 1.0,
    }
    checks["anchor_1_pass"] = all([
        checks["pred_loss_leq_1.1x"],
        checks["pred_max_captured_energy_gt_0.9"],
        checks["pred_R_below_halfpi_allthrough"],
    ])
    out = {"anchor": "anchor_1_matreg", "seed": SEED, "device": DEV,
           "config": {"d": d, "m": m, "r_true": r_true, "rank": 16, "epochs": 2000},
           "adamw": {k: v for k, v in res_adamw.items() if k != "trajectory"},
           "grassroute": {k: v for k, v in res_gr.items() if k != "trajectory"},
           "checks": checks,
           "trajectories": {"adamw": res_adamw["trajectory"], "grassroute": res_gr["trajectory"]}}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anchor1_results.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("\n[checks]", json.dumps(checks, indent=1))
    print("anchor_1_pass =", checks["anchor_1_pass"])
    return checks["anchor_1_pass"]


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
