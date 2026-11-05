"""
GrassRoute EDA — core measurement library.
Measures the two H0 premises on real training gradients:
  (P1) gradient right-covariance C = G^T G spectrum is concentrated in a low-dim subspace
  (P2) the principal subspace drifts slowly over training.

Routing convention (README sec 1/3): for a parameter matrix W in R^{m x d},
G = dL/dW in R^{m x d}; the routed subspace is a RIGHT subspace U in Gr(r, d),
so the relevant covariance is C = G^T G in R^{d x d}. Conv kernels (out,in,kh,kw)
are flattened to (out, in*kh*kw). This matches Z_t = G_t U_t, U in Gr(r,d).
"""
import numpy as np, torch, json, math

RANKS = [4, 8, 16, 32]

@torch.no_grad()
def spectrum_metrics(C):
    # C: (d,d) symmetric PSD covariance (numpy float64)
    d = C.shape[0]
    w = np.linalg.eigvalsh(C)            # ascending
    lam = np.clip(w[::-1], 0, None)      # descending, nonneg
    E = float(lam.sum())
    out = {"d": int(d), "total_energy": E, "lambda_top64": lam[:64].tolist()}
    if E <= 0:
        return out, lam
    p = lam / E
    # effective ranks
    out["erank_participation"] = float((lam.sum()**2) / (np.sum(lam**2) + 1e-30))  # (Σλ)²/Σλ²
    pp = p[p > 0]
    out["erank_entropy"] = float(np.exp(-np.sum(pp * np.log(pp))))                 # exp(H)
    out["stable_rank"] = float(E / (lam[0] + 1e-30))                               # E/λ1
    # energy captured in top-r, eigengap(r), normalized eigengap
    er, gap, ngap = {}, {}, {}
    for r in RANKS:
        if r < d:
            er[r] = float(lam[:r].sum() / E)
            g = float(lam[r-1] - lam[r])
            gap[r] = g
            ngap[r] = float(g / (lam[0] + 1e-30))
    out["energy_topr"] = er
    out["eigengap_r"] = gap
    out["eigengap_r_norm"] = ngap
    return out, lam

@torch.no_grad()
def principal_angles(U1, U2):
    # U1,U2: (d,r) orthonormal columns. Returns angles (rad) ascending-cos => we return sorted asc angle.
    M = U1.T @ U2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)                  # r angles
    return theta

def topr_basis(C, r):
    w, V = np.linalg.eigh(C)              # ascending
    return V[:, ::-1][:, :r]              # top-r eigenvectors (d,r)

class CovTracker:
    """Accumulates C = mean_s G_s^T G_s over a window, snapshots spectrum + drift."""
    def __init__(self, name, d, drift_rank=16):
        self.name = name; self.d = d
        self.drift_rank = min(drift_rank, d)
        self.buf = np.zeros((d, d), dtype=np.float64); self.cnt = 0
        self.snaps = []          # list of metric dicts
        self.prev_U = None; self.prev_step = None

    def add(self, G):
        # G: torch tensor (m,d) or flattened (m,d)
        g = G.detach().float()
        C = (g.T @ g).cpu().numpy().astype(np.float64)
        self.buf += C; self.cnt += 1

    def snapshot(self, step):
        if self.cnt == 0: return
        C = self.buf / self.cnt
        m, lam = spectrum_metrics(C)
        m["step"] = int(step); m["window_steps"] = int(self.cnt)
        # drift
        Ur = topr_basis(C, self.drift_rank)
        if self.prev_U is not None:
            th = principal_angles(self.prev_U, Ur)
            gdist = float(np.sqrt(np.sum(th**2)))
            R = float(th.max())            # principal-angle radius
            dstep = max(1, step - self.prev_step)
            m["drift"] = {
                "rank": int(self.drift_rank),
                "grassmann_dist": gdist,
                "R_radius_rad": R,
                "R_over_halfpi": float(R / (math.pi/2)),
                "grassmann_dist_per_step": float(gdist / dstep),
                "mean_angle_rad": float(th.mean()),
                "steps_since_prev": int(dstep),
            }
        self.prev_U = Ur; self.prev_step = step
        self.snaps.append(m)
        # reset window buffer (measure LOCAL covariance per window)
        self.buf[:] = 0; self.cnt = 0

    def summary(self):
        return {"layer": self.name, "d": self.d, "drift_rank": self.drift_rank,
                "snapshots": self.snaps}

def random_subspace_drift_null(d, r, trials=20):
    """Expected Grassmann dist / R between two independent random r-subspaces in R^d."""
    gd, Rs = [], []
    for _ in range(trials):
        A = np.random.randn(d, r); B = np.random.randn(d, r)
        Qa = np.linalg.qr(A)[0]; Qb = np.linalg.qr(B)[0]
        th = principal_angles(Qa, Qb)
        gd.append(np.sqrt(np.sum(th**2))); Rs.append(th.max())
    return {"rank": r, "d": d, "null_grassmann_dist_mean": float(np.mean(gd)),
            "null_R_mean": float(np.mean(Rs)), "half_pi": math.pi/2}
