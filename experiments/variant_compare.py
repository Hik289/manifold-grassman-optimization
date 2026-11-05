"""
二阶矩处理三变体对比 (Director 裁决 1)。
在 anchor_1 matreg (真实在线 U 更新) 上对比 4 种 reset_mode:
  hard        : §3.5 硬 reset (V<-0 每步) = ablation A8 = 发散源
  none        : 完全不 reset = 旧 reset_v=False
  decay       : §Risks 指数衰减 reset (v_decay=0.5), 保留部分曲率新鲜度
  procrustes  : V 随子空间旋转 (非清零), 最保留"曲率新鲜度"意图
选最稳且保留避免陈旧曲率意图的作默认。
"""
import torch, math, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grassroute import GrassRoute
from anchor1_matreg import make_data, principal_angles_R
SEED = 42; DEV = "cuda" if torch.cuda.is_available() else "cpu"
d, m, r_true = 256, 64, 16
X, Y, W_star, U_star = make_data(d=d, m=m, r_true=r_true)


def run(mode, epochs=2000):
    torch.manual_seed(SEED)
    W = torch.nn.Parameter(0.01 * torch.randn(m, d, device=DEV))
    kw = {"reset_mode": mode}
    if mode == "hard":
        kw["reset_v_thresh"] = 0.0   # 每步硬 reset (忠实 §3.5)
    opt = GrassRoute([W], lr=3e-3, lr_perp=3e-3, grassmann_lr=0.1, rank=r_true, mu=0.9, **kw)
    best = 1e30; losses = []
    for ep in range(epochs):
        f = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * ep / epochs))
        opt.param_groups[0]["lr"] = 3e-3 * f
        opt.param_groups[0]["lr_perp"] = 3e-3 * f
        opt.zero_grad()
        loss = (X @ W.T - Y).pow(2).mean()
        loss.backward(); opt.step()
        best = min(best, loss.item())
        if ep % 100 == 0:
            losses.append((ep, loss.item()))
    U = opt.state[W]["U"]
    R = principal_angles_R(U, U_star) / (math.pi / 2)
    # 后期稳定性: 最后 200 epoch loss 的最大/最小比 (发散->大)
    return {"mode": mode, "final_loss": loss.item(), "best_loss": best,
            "final_R_over_halfpi": R, "loss_curve_every100": losses}


if __name__ == "__main__":
    # AdamW 参考
    torch.manual_seed(SEED)
    Wa = torch.nn.Parameter(0.01 * torch.randn(m, d, device=DEV))
    oa = torch.optim.AdamW([Wa], lr=3e-3)
    for ep in range(2000):
        f = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * ep / 2000))
        oa.param_groups[0]["lr"] = 3e-3 * f
        oa.zero_grad(); la = (X @ Wa.T - Y).pow(2).mean(); la.backward(); oa.step()
    adamw_final = la.item()

    results = {"adamw_final_loss": adamw_final, "variants": []}
    print(f"{'mode':<12} {'final_loss':>12} {'best_loss':>12} {'R/halfpi':>10} {'vs_adamw':>10}")
    print(f"{'AdamW':<12} {adamw_final:>12.4e} {adamw_final:>12.4e} {'—':>10} {'1.0x':>10}")
    for mode in ["hard", "none", "decay", "procrustes"]:
        r = run(mode)
        results["variants"].append(r)
        ratio = r["final_loss"] / (adamw_final + 1e-30)
        print(f"{mode:<12} {r['final_loss']:>12.4e} {r['best_loss']:>12.4e} "
              f"{r['final_R_over_halfpi']:>10.4f} {ratio:>9.1f}x")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "variant_compare.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\n[saved] variant_compare.json")
