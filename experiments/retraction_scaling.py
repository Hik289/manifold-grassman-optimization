"""
Per-step 子空间更新成本 scaling: GrassRoute (QR retraction, O(mdr)) vs SubTrack++ (exact Exp-map, per-step SVD)。
支撑核心设计论点 (README §3.5 / checklist 1.1): 为什么用 QR retraction 而非 exact Exp。

在一系列矩阵维度 d 上, 隔离测量单步"子空间更新"操作的 wall-clock:
- GrassRoute 子空间步: Q=(I-UU^T)C U ; H momentum ; U_new=qf(U+ρH) (QR)  → O(mdr) + O(dr²) QR
- SubTrack++ 子空间步: 同 Q/H, 但 U_new = Exp_U(ρH) via thin SVD of (ρH)  → 每步 SVD O(dr²)... 实为 svd(d×r)
  ⚠️ SubTrack++ 论文的 exact geodesic 需要 tangent 的 SVD; 大 d 时 svd 比 qr 昂贵得多 + 数值退化

固定 m, r; 扫 d ∈ {64, 128, 192, 256, 512, 768, 1152, 2304, 4608} (覆盖 RecSys→ViT→CNN conv)。
每个 d 重复 N 次取中位数, GPU warmup。输出 per-step μs + 比值 + scaling。
"""
import torch
import time
import json
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def qf(A):
    Q, R = torch.linalg.qr(A)
    s = torch.sign(torch.diagonal(R)); s[s == 0] = 1
    return Q * s.unsqueeze(0)


def polar(A):
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh


@torch.no_grad()
def grassroute_step(G, U, H, rho, mu):
    """GrassRoute 子空间更新 (QR retraction)。"""
    Z = G @ U                          # (m,r)
    GtZ = G.T @ Z                      # (d,r) = C U
    gnorm2 = (G * G).sum() + 1e-12
    GtZ = GtZ / gnorm2
    Q = GtZ - U @ (U.T @ GtZ)
    Hn = mu * (H - U @ (U.T @ H)) + (1 - mu) * Q
    Hn = Hn - U @ (U.T @ Hn)
    U_new = qf(U + rho * Hn)           # QR retraction O(dr^2)
    Rp = polar(U_new.T @ U)            # Procrustes r×r
    return U_new, Hn


@torch.no_grad()
def subtrack_step(G, U, H, rho, mu):
    """SubTrack++ 子空间更新 (exact Exp-map via thin SVD of tangent)。"""
    Z = G @ U
    GtZ = G.T @ Z
    gnorm2 = (G * G).sum() + 1e-12
    GtZ = GtZ / gnorm2
    Q = GtZ - U @ (U.T @ GtZ)
    Hn = mu * (H - U @ (U.T @ H)) + (1 - mu) * Q
    Hn = Hn - U @ (U.T @ Hn)
    # exact Exp_U(rho Hn) via thin SVD (SubTrack++ 几何步)
    Qm, S, Pt = torch.linalg.svd(rho * Hn, full_matrices=False)   # 每步 SVD
    cosS = torch.diag(torch.cos(S)); sinS = torch.diag(torch.sin(S))
    U_new = U @ Pt.T @ cosS @ Pt + Qm @ sinS @ Pt
    U_new = qf(U_new)                  # 重正交
    Rp = polar(U_new.T @ U)
    return U_new, Hn


def bench(fn, m, d, r, iters=50, warmup=10):
    torch.manual_seed(0)
    G = torch.randn(m, d, device=DEV)
    U = qf(torch.randn(d, r, device=DEV))
    H = torch.zeros(d, r, device=DEV)
    for _ in range(warmup):
        U, H = fn(G, U, H, 0.1, 0.9)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        U, H = fn(G, U, H, 0.1, 0.9)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6   # μs/step


def run():
    m, r = 256, 16
    ds = [64, 128, 192, 256, 512, 768, 1152, 2304, 4608]
    res = {"m": m, "r": r, "d_values": ds, "grassroute_us": [], "subtrack_us": [], "ratio": []}
    print(f"per-step 子空间更新成本 (m={m}, r={r}), μs/step:")
    print(f"{'d':>6} {'GrassRoute(QR)':>16} {'SubTrack++(SVD)':>18} {'ratio':>8}")
    for d in ds:
        gr = bench(grassroute_step, m, d, r)
        st = bench(subtrack_step, m, d, r)
        res["grassroute_us"].append(gr)
        res["subtrack_us"].append(st)
        res["ratio"].append(st / gr)
        print(f"{d:>6} {gr:>16.1f} {st:>18.1f} {st/gr:>7.2f}x", flush=True)
    # 也测一个真实 ViT full-model 单步 (48 路由层) 的两方对比
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "retraction_scaling.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n[saved] {out}")
    # scaling 拟合 (log-log slope)
    import math
    def slope(xs, ys):
        lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
        n = len(lx); mx = sum(lx)/n; my = sum(ly)/n
        return sum((lx[i]-mx)*(ly[i]-my) for i in range(n))/sum((lx[i]-mx)**2 for i in range(n))
    print(f"GrassRoute scaling: O(d^{slope(ds, res['grassroute_us']):.2f})")
    print(f"SubTrack++ scaling: O(d^{slope(ds, res['subtrack_us']):.2f})")


if __name__ == "__main__":
    run()
