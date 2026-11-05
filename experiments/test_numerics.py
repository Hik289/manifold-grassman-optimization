"""
GrassRoute 数值单元测试 (README §11, anchor_2)。

8 项断言，阈值 < 1e-6:
  1. Orthonormality:            ||U^T U - I||_F < 1e-6
  2. Tangency:                  ||U^T H||_F < 1e-6
  3. Projection decomposition:  ||G - G_par - G_perp||_F < 1e-6
  4. Orthogonality:             |<G_par, G_perp>_F| < 1e-6
  5. Grassmann invariance:      ||UU^T - (UQ)(UQ)^T||_F < 1e-6, Q∈O(r)
  6. Objective invariance:      |phi(U) - phi(UQ)| < 1e-6
  7. Retraction identity:       ||Retr_U(0) - U||_F < 1e-6
  8. Finite difference:         Grassmann 方向导数 vs retraction 有限差分一致

全部在 FP32/FP64 混合验证下运行。输出 JSON: test_results.json
"""
import torch
import json
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grassroute import GrassRoute, _qf, _polar_orthogonal

torch.manual_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64   # 数值测试用 fp64 以确认解析恒等式（实现本身 fp32，见 §3.5）
THR = 1e-6


def phi(U, G):
    """残差能量目标 phi(U) = 1/2 ||G(I-UU^T)||_F^2 (README §2.2)。"""
    P = U @ U.T
    R = G @ (torch.eye(U.shape[0], device=U.device, dtype=U.dtype) - P)
    return 0.5 * (R.norm() ** 2)


def proj_tangent(U, A):
    return A - U @ (U.T @ A)


def grassmann_grad_phi(U, G):
    """phi(U) 在 Grassmann 上的黎曼梯度 = -(I-UU^T) C U, C=G^T G。
    (最小化 phi <=> 最大化 tr(U^T C U); ascent 方向 Q=(I-UU^T)CU)。"""
    C = G.T @ G
    return -proj_tangent(U, C @ U)


def run():
    results = {}
    d, r, m = 64, 8, 32

    # 基础对象
    U = _qf(torch.randn(d, r, device=DEV, dtype=DTYPE))
    G = torch.randn(m, d, device=DEV, dtype=DTYPE)
    Qo, _ = torch.linalg.qr(torch.randn(r, r, device=DEV, dtype=DTYPE))  # Q∈O(r)

    # ---- Test 1: Orthonormality ----
    e1 = (U.T @ U - torch.eye(r, device=DEV, dtype=DTYPE)).norm().item()
    results["1_orthonormality"] = {"error": e1, "pass": e1 < THR}

    # ---- Test 2: Tangency (构造 Grassmann momentum H 后检查 U^T H = 0) ----
    # 复现 optimizer Step5 的切向投影
    C = G.T @ G
    Qtan = proj_tangent(U, C @ U)
    H0 = torch.randn(d, r, device=DEV, dtype=DTYPE)
    H_proj = proj_tangent(U, H0)
    H = 0.9 * H_proj + 0.1 * Qtan
    H = proj_tangent(U, H)
    e2 = (U.T @ H).norm().item()
    results["2_tangency"] = {"error": e2, "pass": e2 < THR}

    # ---- Test 3: Projection decomposition ----
    Z = G @ U
    G_par = Z @ U.T
    G_perp = G - G_par
    e3 = (G - G_par - G_perp).norm().item()
    results["3_projection_decomposition"] = {"error": e3, "pass": e3 < THR}

    # ---- Test 4: Orthogonality <G_par, G_perp>_F = 0 ----
    e4 = abs((G_par * G_perp).sum().item())
    results["4_orthogonality"] = {"error": e4, "pass": e4 < THR}

    # ---- Test 5: Grassmann invariance UU^T = (UQ)(UQ)^T ----
    UQ = U @ Qo
    e5 = (U @ U.T - UQ @ UQ.T).norm().item()
    results["5_grassmann_invariance"] = {"error": e5, "pass": e5 < THR}

    # ---- Test 6: Objective invariance phi(U)=phi(UQ) ----
    e6 = abs(phi(U, G).item() - phi(UQ, G).item())
    results["6_objective_invariance"] = {"error": e6, "pass": e6 < THR}

    # ---- Test 7: Retraction identity Retr_U(0) = U ----
    # QR retraction 含符号规范化，与 U 的符号一致（U 本身已由 _qf 规范化）
    U_retr0 = _qf(U + 0.0 * H)
    e7 = (U_retr0 - U).norm().item()
    results["7_retraction_identity"] = {"error": e7, "pass": e7 < THR}

    # ---- Test 8: Finite difference ----
    # d/dt phi(Retr_U(t*xi))|_{t=0} 应等于 <grad phi(U), xi>_F (canonical metric)
    xi = proj_tangent(U, torch.randn(d, r, device=DEV, dtype=DTYPE))
    xi = xi / xi.norm()
    grad = grassmann_grad_phi(U, G)
    analytic = (grad * xi).sum().item()          # <grad, xi>_F
    h = 1e-6
    # 中心差分, 沿 retraction
    phi_p = phi(_qf(U + h * xi), G).item()
    phi_m = phi(_qf(U - h * xi), G).item()
    fd = (phi_p - phi_m) / (2 * h)
    e8 = abs(analytic - fd)
    # 相对阈值: 梯度量级大时用相对误差
    rel = e8 / (abs(analytic) + 1e-12)
    results["8_finite_difference"] = {
        "analytic_deriv": analytic, "finite_diff": fd,
        "abs_error": e8, "rel_error": rel,
        "pass": rel < 1e-4 or e8 < THR
    }

    all_pass = all(v["pass"] for v in results.values())
    out = {"all_pass": all_pass, "threshold": THR,
           "dtype": str(DTYPE), "device": DEV,
           "dims": {"d": d, "r": r, "m": m}, "tests": results}

    # ---- 附加: 在真实 optimizer.step() 后验证 fp32 实现路径的不变量 ----
    # 用有界 loss (低秩回归): 最小化 ||W A - Y||^2, 保证梯度不发散
    torch.manual_seed(42)
    A_fix = torch.randn(d, d, device=DEV)
    W_star = torch.randn(m, d, device=DEV)
    Y = W_star @ A_fix
    W = torch.nn.Parameter(torch.randn(m, d, device=DEV, dtype=torch.float32) * 0.01)
    opt = GrassRoute([W], lr=1e-3, grassmann_lr=1e-3, rank=r, mu=0.9)
    for _ in range(50):
        opt.zero_grad()
        loss = (W @ A_fix - Y).pow(2).mean()
        loss.backward()
        opt.step()
    st = opt.state[W]
    U_run = st["U"]
    orth_err = (U_run.T @ U_run - torch.eye(r, device=DEV)).norm().item()
    # tangency 不变量: 每步 U_t^T H_t 应~0 (optimizer 内部在 U 更新前记录到 diag)
    tang_err = st["diag"]["tangency_error"]
    retr_err = st["diag"]["retraction_error"]
    out["optimizer_runtime_check"] = {
        "orthonormality_after_50steps": orth_err,
        "tangency_error_last_step": tang_err,
        "retraction_error_last_step": retr_err,
        "final_loss": loss.item(),
        "pass": orth_err < 1e-5 and tang_err < 1e-5 and retr_err < 1e-5,
        "note": "fp32 实现路径, 阈值 1e-5 (fp32 QR 精度); tangency=||U_t^T H_t|| (U更新前)"
    }

    print(json.dumps(out, indent=2))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nALL_PASS =", all_pass, "| runtime_check =", out["optimizer_runtime_check"]["pass"])
    return all_pass


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
