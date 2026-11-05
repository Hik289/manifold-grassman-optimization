"""
GrassRoute EDA — synthetic ground-truth tracking bed (checklist H0.anchor_4 / goal 4).

Generates controllable low-rank, time-varying gradient covariance:
    Sigma_t = U_t* diag(spec) U_t*^T + tau * I_d
where:
  - U_t* in Gr(r,d) is the planted GROUND-TRUTH principal subspace,
  - spec has a controllable eigengap (lambda_r / lambda_{r+1} contrast),
  - U_t* rotates along a fixed geodesic at a controllable drift rate delta (rad/step),
  - tau controls the rank-tail (noise floor / spectrum concentration).

We emit, per (eigengap, drift, tau) config:
  - the planted spectrum + true eigengap,
  - synthetic gradient stream G_t ~ rows N(0, Sigma_t),
  - and a reference measurement of how a *sample* covariance recovers the planted subspace.

This is the ground-truth bed the tracking mechanism (GrassRoute U_t update) will be
validated against later. We save both the generator config AND a difficulty diagnostic:
how well a plain online-PCA/Oja tracker locks onto U_t* under each regime — establishing
the baseline "trackability" landscape (which regimes are easy/hard) so experiment design
can pick per-family rank and predict favorable/unfavorable settings.
"""
import os, sys, json, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gr_eda_core import principal_angles

def geodesic_U(U0, Delta, t):
    # move U0 along tangent Delta (d x r, U0^T Delta = 0) by arc t (radians via SVD of Delta)
    # Exp_{U0}(t*Delta): use thin SVD Delta = Q S P^T
    Q,S,Pt = np.linalg.svd(Delta, full_matrices=False)
    cos=np.cos(S*t); sin=np.sin(S*t)
    Ut = U0@Pt.T@np.diag(cos)@Pt + Q@np.diag(sin)@Pt
    # reorthonormalize
    Ut,_=np.linalg.qr(Ut)
    return Ut

def make_spectrum(r, d, eigengap_ratio, tau):
    # top-r eigenvalues (well above tail), tail = tau. eigengap_ratio = lambda_r/lambda_{r+1}
    top = np.linspace(10.0, 5.0, r)          # decaying signal eigenvalues
    lam_r = top[-1]
    lam_rp1 = lam_r / eigengap_ratio         # first tail eigenvalue
    return top, lam_r, lam_rp1

def oja_track(stream_C, U_init, lr):
    """Simple online-PCA (Oja on covariance) subspace tracker as a reference difficulty probe."""
    U = U_init.copy()
    errs=[]
    for C, Ustar in stream_C:
        # Oja update: U <- qf(U + lr * C U)
        G = C @ U
        U,_ = np.linalg.qr(U + lr*G)
        th = principal_angles(U, Ustar)
        errs.append({"grassmann_dist":float(np.sqrt(np.sum(th**2))),
                     "R_rad":float(th.max()),
                     "R_over_halfpi":float(th.max()/(math.pi/2))})
    return errs

def run(out, d=256, r=16, steps=600, batch=512):
    rng=np.random.default_rng(42)
    configs=[]
    for eg in [1.5, 3.0, 10.0]:              # eigengap ratio: small / medium / large
        for drift in [0.0, 1e-3, 5e-3, 2e-2]:# drift rate rad/step
            for tau in [0.05, 0.5]:          # rank-tail floor: concentrated / diffuse
                configs.append((eg,drift,tau))
    results=[]
    for (eg,drift,tau) in configs:
        top, lam_r, lam_rp1 = make_spectrum(r,d,eg,tau)
        # base subspace + fixed geodesic tangent direction
        U0 = np.linalg.qr(rng.standard_normal((d,r)))[0]
        Draw = rng.standard_normal((d,r)); Draw = Draw - U0@(U0.T@Draw)  # tangent
        Draw,_=np.linalg.qr(Draw); Draw = Draw * 1.0                      # unit-ish speeds
        stream=[]
        planted=[]
        for t in range(steps):
            Ut = geodesic_U(U0, Draw, drift*t)
            # true covariance: signal + isotropic tail
            Sig = (Ut*top) @ Ut.T + tau*np.eye(d)
            # sample gradient rows to form empirical C (this is what a real optimizer sees)
            L = np.linalg.cholesky(Sig + 1e-8*np.eye(d))
            Gt = rng.standard_normal((batch,d)) @ L.T
            C = (Gt.T@Gt)/batch
            stream.append((C, Ut))
            if t % 100 == 0:
                w=np.linalg.eigvalsh(C)[::-1]
                planted.append({"step":t,
                    "emp_eigengap_r":float(w[r-1]-w[r]),
                    "emp_energy_topr":float(w[:r].sum()/w.sum()),
                    "true_lambda_r":float(lam_r),"true_lambda_rp1":float(lam_rp1)})
        # difficulty probe: Oja tracker from a random (wrong) start
        Uinit=np.linalg.qr(rng.standard_normal((d,r)))[0]
        errs=oja_track(stream, Uinit, lr=0.05/ (top.mean()))
        final=errs[-1]; mid=errs[len(errs)//2]
        # drift ground-truth: dist between consecutive true subspaces
        gt_drift = float(np.sqrt(np.sum(principal_angles(stream[0][1],stream[1][1])**2))) if steps>1 else 0.0
        results.append({
            "config":{"eigengap_ratio":eg,"drift_rate":drift,"tau_tail":tau,"d":d,"r":r,
                      "steps":steps,"batch":batch},
            "true_eigengap":float(lam_r-lam_rp1),
            "gt_drift_per_step":gt_drift,
            "planted_spectrum_top":top.tolist(),
            "planted_checks":planted,
            "oja_probe":{"final":final,"mid":mid,
                         "converged": bool(final["R_over_halfpi"]<1.0 and final["grassmann_dist"]<mid["grassmann_dist"])},
        })
        print(f"[synth] eg={eg} drift={drift} tau={tau} -> Oja final R/(pi/2)={final['R_over_halfpi']:.3f} gd={final['grassmann_dist']:.3f}", flush=True)
    os.makedirs(out,exist_ok=True)
    p=os.path.join(out,"synth_tracking_bed.json")
    with open(p,"w") as f: json.dump({"description":"controllable low-rank time-varying covariance tracking bed; Oja probe = reference difficulty",
                                       "results":results},f)
    print(f"[saved] {p}",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True)
    a=ap.parse_args(); run(a.out)
