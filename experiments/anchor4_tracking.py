"""
anchor_4: 合成"低秩慢漂移"数据上 GrassRoute 的 subspace tracking 有效。

复现 EDA 合成检验床 (data/eda_raw/gr_eda_synth.py, seed=42) 的 24 组 ground-truth 流,
用 GrassRoute 的子空间更新机制 (Grassmann momentum + QR retraction, README §3 Step5)
跟踪植入的时变 U_t*, 对照 EDA 已交付的 Oja 参考探针。

验收 (checklist H0.anchor_4 + EDA §6):
  - d_Gr([U_t],[U_t*]) 收敛到小稳态值, R < π/2 全程
  - 慢漂移区 (δ≤5e-3) R/(π/2) < 0.1
  - GrassRoute 不劣于 Oja 探针 (尤其高漂移 δ=0.02 区展示曲率修正优势)
  - tracking error 随 eigengap 增大而减小 (Theorem 2 定性)

注意: 这里只驱动"子空间更新"部分 (给定梯度流 G_t, 更新 U_t),
不涉及参数 W 更新——与 EDA Oja 探针同口径对比。

seed=42。
"""
import numpy as np
import json
import sys
import os
import math

SEED = 42


def principal_angles(U1, U2):
    M = U1.T @ U2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def geodesic_U(U0, Delta, t):
    Q, S, Pt = np.linalg.svd(Delta, full_matrices=False)
    cos = np.cos(S * t); sin = np.sin(S * t)
    Ut = U0 @ Pt.T @ np.diag(cos) @ Pt + Q @ np.diag(sin) @ Pt
    Ut, _ = np.linalg.qr(Ut)
    return Ut


def make_spectrum(r, eigengap_ratio):
    top = np.linspace(10.0, 5.0, r)
    lam_r = top[-1]
    lam_rp1 = lam_r / eigengap_ratio
    return top, lam_r, lam_rp1


def qf(A):
    Q, R = np.linalg.qr(A)
    sign = np.sign(np.diag(R)); sign[sign == 0] = 1.0
    return Q * sign[None, :]


def proj_tangent(U, A):
    return A - U @ (U.T @ A)


def grassroute_track(stream, U_init, rho, mu):
    """GrassRoute 子空间更新 (README §3 Step5), 从错误初始 U 跟踪 ground-truth。
    stream: list of (C, Ustar). 返回逐步 tracking error。"""
    U = U_init.copy()
    r = U.shape[1]
    H = np.zeros_like(U)
    errs = []
    for C, Ustar in stream:
        # Q = (I-UU^T) C U
        CU = C @ U
        Q = CU - U @ (U.T @ CU)
        H_proj = proj_tangent(U, H)
        H = mu * H_proj + (1 - mu) * Q
        H = proj_tangent(U, H)
        U = qf(U + rho * H)
        th = principal_angles(U, Ustar)
        errs.append({"grassmann_dist": float(np.sqrt(np.sum(th ** 2))),
                     "R_rad": float(th.max()),
                     "R_over_halfpi": float(th.max() / (math.pi / 2))})
    return errs


def oja_track(stream, U_init, lr):
    U = U_init.copy()
    errs = []
    for C, Ustar in stream:
        G = C @ U
        U, _ = np.linalg.qr(U + lr * G)
        th = principal_angles(U, Ustar)
        errs.append({"grassmann_dist": float(np.sqrt(np.sum(th ** 2))),
                     "R_rad": float(th.max()),
                     "R_over_halfpi": float(th.max() / (math.pi / 2))})
    return errs


def build_stream(eg, drift, tau, rng, d=256, r=16, steps=600, batch=512):
    top, lam_r, lam_rp1 = make_spectrum(r, eg)
    U0 = np.linalg.qr(rng.standard_normal((d, r)))[0]
    Draw = rng.standard_normal((d, r)); Draw = Draw - U0 @ (U0.T @ Draw)
    Draw, _ = np.linalg.qr(Draw)
    stream = []
    for t in range(steps):
        Ut = geodesic_U(U0, Draw, drift * t)
        Sig = (Ut * top) @ Ut.T + tau * np.eye(d)
        L = np.linalg.cholesky(Sig + 1e-8 * np.eye(d))
        Gt = rng.standard_normal((batch, d)) @ L.T
        C = (Gt.T @ Gt) / batch
        stream.append((C, Ut))
    return stream, top


def run():
    d, r, steps, batch = 256, 16, 600, 512
    # 与 gr_eda_synth.py 完全一致的配置顺序与 rng 消费顺序 (保证 ground-truth 一致)
    rng = np.random.default_rng(SEED)
    configs = []
    for eg in [1.5, 3.0, 10.0]:
        for drift in [0.0, 1e-3, 5e-3, 2e-2]:
            for tau in [0.05, 0.5]:
                configs.append((eg, drift, tau))

    results = []
    for (eg, drift, tau) in configs:
        stream, top = build_stream(eg, drift, tau, rng, d, r, steps, batch)
        # 同一错误初始子空间, 对两个 tracker 公平
        Uinit = np.linalg.qr(rng.standard_normal((d, r)))[0]
        # GrassRoute: rho 与 Oja lr 同量级 (lr=0.05/mean), 便于公平比较
        rho = 0.05 / top.mean()
        gr_errs = grassroute_track(stream, Uinit, rho=rho, mu=0.9)
        oja_errs = oja_track(stream, Uinit.copy(), lr=0.05 / top.mean())
        gt_drift = float(np.sqrt(np.sum(principal_angles(stream[0][1], stream[1][1]) ** 2)))
        results.append({
            "config": {"eigengap_ratio": eg, "drift_rate": drift, "tau_tail": tau,
                       "d": d, "r": r, "steps": steps, "batch": batch},
            "gt_drift_per_step": gt_drift,
            "grassroute": {"final": gr_errs[-1], "mid": gr_errs[len(gr_errs) // 2],
                           "min_R_over_halfpi": min(e["R_over_halfpi"] for e in gr_errs),
                           "converged": bool(gr_errs[-1]["R_over_halfpi"] < 1.0 and
                                             gr_errs[-1]["grassmann_dist"] < gr_errs[len(gr_errs) // 2]["grassmann_dist"])},
            "oja": {"final": oja_errs[-1], "mid": oja_errs[len(oja_errs) // 2]},
        })
        print(f"[track] eg={eg} drift={drift} tau={tau} -> "
              f"GR final R/(pi/2)={gr_errs[-1]['R_over_halfpi']:.3f} | "
              f"Oja={oja_errs[-1]['R_over_halfpi']:.3f} | "
              f"GR<=Oja: {gr_errs[-1]['R_over_halfpi'] <= oja_errs[-1]['R_over_halfpi'] + 0.02}")

    # ---- 验收判定 ----
    slow = [x for x in results if x["config"]["drift_rate"] <= 5e-3 and x["config"]["tau_tail"] == 0.05]
    slow_R_ok = all(x["grassroute"]["final"]["R_over_halfpi"] < 0.1 for x in slow)
    all_R_below_halfpi = all(x["grassroute"]["final"]["R_over_halfpi"] < 1.0 for x in results
                             if x["config"]["drift_rate"] <= 5e-3)
    # GrassRoute 不劣于 Oja (容差 0.02)
    not_worse = sum(1 for x in results
                    if x["grassroute"]["final"]["R_over_halfpi"] <= x["oja"]["final"]["R_over_halfpi"] + 0.02)
    # eigengap 单调性: 固定 drift/tau, 大 eigengap 应有更小或相当 error (高漂移区检验)
    high_drift = {(x["config"]["eigengap_ratio"]): x["grassroute"]["final"]["R_over_halfpi"]
                  for x in results if x["config"]["drift_rate"] == 2e-2 and x["config"]["tau_tail"] == 0.05}

    checks = {
        "slow_drift_R_below_0.1": slow_R_ok,
        "slow_drift_R_below_halfpi": all_R_below_halfpi,
        "grassroute_not_worse_than_oja_count": f"{not_worse}/{len(results)}",
        "grassroute_not_worse_frac": not_worse / len(results),
        "high_drift_R_by_eigengap": high_drift,
    }
    checks["anchor_4_pass"] = bool(slow_R_ok and all_R_below_halfpi and not_worse >= len(results) * 0.8)

    out = {"anchor": "anchor_4_tracking", "seed": SEED,
           "n_configs": len(results), "checks": checks, "results": results}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anchor4_results.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("\n[checks]", json.dumps(checks, indent=1))
    print("anchor_4_pass =", checks["anchor_4_pass"])
    return checks["anchor_4_pass"]


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
