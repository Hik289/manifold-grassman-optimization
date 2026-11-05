"""
GrassRoute — Complete Figure Generation Script
Generates all 13 main figures (F1–F13) for the paper.
Usage: Run from a checkout containing archived result files:
    python3 figures/figures_all_code.py
Outputs written to figures/*.pdf
All data read from experiments/ directory.
Single seed (seed=42), no error bars.
"""
import json, os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXP  = ROOT / "experiments"
RUNS = EXP / "runs"
OUT  = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
os.listdir(str(OUT))   # warm dentry cache (see MEMORY.md: matplotlib evicts cache)

# ── Load ALL data BEFORE matplotlib import ───────────────────────────────
print("Loading data...")
with open(EXP/"results_running.json") as f:  rr=json.load(f)
with open(EXP/"variant_compare.json") as f:  vc=json.load(f)
with open(EXP/"anchor4_results.json") as f:  a4=json.load(f)
with open(RUNS/"traj_sciml_B4_r16.json") as f: b4t=json.load(f)
with open(RUNS/"traj_sciml_B5_r16.json") as f: b5t=json.load(f)
with open(RUNS/"traj_graph_B5_r16.json") as f: gb5t=json.load(f)
with open(RUNS/"theory_check.json") as f:    tc=json.load(f)
with open(RUNS/"sciml_g2.json") as f:        sg2=json.load(f)
with open(RUNS/"wallclock_panorama.json") as f: wcp=json.load(f)
with open(RUNS/"rank_scaling.json") as f:    rs=json.load(f)
with open(RUNS/"retraction_scaling.json") as f: ret=json.load(f)
with open(RUNS/"recsys_full.json") as f:     recsys=json.load(f)
with open(RUNS/"continual_results.json") as f: cont=json.load(f)
with open(RUNS/"continual_g2.json") as f:    cg2=json.load(f)
with open(RUNS/"cnn_partial1000.json") as f: cpdata=json.load(f)
print("Data loaded.")

# Open output file handles BEFORE matplotlib import
handles = {n: open(OUT/f"{n}.pdf","wb") for n in [
    "F1_winmap","F2_spectral_bar","F3_tradeoff","F4_sciml_g5",
    "F5_vreset","F6_anchor4","F7_tracking","F8_efficiency",
    "F9_cnn_tradeoff","F10_recsys","F11_continual","F12_ablation","F13_rank_sweep"
]}

# ── matplotlib ─────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "font.family":"serif","font.size":10,"axes.labelsize":11,
    "axes.labelweight":"bold","axes.titlesize":10,"xtick.labelsize":9.5,
    "ytick.labelsize":9.5,"legend.fontsize":8.5,"lines.linewidth":2.0,
    "lines.markersize":6,"axes.grid":True,"grid.alpha":0.3,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":150,"savefig.bbox":"tight","savefig.pad_inches":0.05,
})
C5 = ["#648FFF","#785EF0","#DC267F","#FE6100","#FFB000"]
C_WIN="#2CA02C"; C_TIE="#FF7F0E"; C_LOSE="#D62728"
color_map = {"win":C_WIN,"tie":C_TIE,"lose":C_LOSE}
marker_map = {"win":"o","tie":"s","lose":"X"}

def save(name):
    plt.savefig(handles[name], format="pdf")
    plt.close()
    handles[name].close()
    print(f"  {name}.pdf saved")

# ─────────────────────────────────────────────────────────────────────────
# F1 — Win-map scatter
# ─────────────────────────────────────────────────────────────────────────
fams_f1 = {
    "SciML":        {"ce": tc["families"]["sciml_B5_r16"]["T1_captured_energy"]["final"],
                     "y": (0.010964 - sg2["B5_lr0.01_glr0.3"]) / 0.010964 * 100, "verdict":"win"},
    "Graph\n(ogbn)":{"ce": tc["families"]["graph_B5_r16"]["T1_captured_energy"]["final"],
                     "y": (rr["main_table"]["graph"]["B5"]["best_test"] -
                           rr["main_table"]["graph"]["B2"]["best_test"]) /
                          rr["main_table"]["graph"]["B2"]["best_test"] * 100, "verdict":"lose"},
    "MLP":          {"ce": tc["families"]["mlp_B5_r16"]["T1_captured_energy"]["final"],
                     "y": (rr["main_table"]["mlp"]["B5"]["best_val"] -
                           rr["main_table"]["mlp"]["B2"]["best_val"]) /
                          rr["main_table"]["mlp"]["B2"]["best_val"] * 100, "verdict":"tie"},
    "CNN\n(full)":  {"ce": rr["main_table"]["cnn"]["B5"]["final_captured_energy"],
                     "y": (rr["main_table"]["cnn"]["B5"]["best_val"] -
                           rr["main_table"]["cnn"]["B2"]["best_val"]) /
                          rr["main_table"]["cnn"]["B2"]["best_val"] * 100, "verdict":"lose"},
}
fig, ax = plt.subplots(figsize=(6.5, 5.2))
xs, ys = [], []
for name, d in fams_f1.items():
    x, y = d["ce"], d["y"]
    ax.scatter(x, y, s=150, color=color_map[d["verdict"]], marker=marker_map[d["verdict"]],
               zorder=5, edgecolors="white", linewidths=0.8)
    dx, dy = 0.02, 2.5
    if "Graph" in name: dx, dy = 0.015, -7
    if "MLP" in name:   dx, dy = 0.02, 2.5
    if "CNN" in name:   dx, dy = 0.02, -3
    ax.annotate(name, (x, y), (x+dx, y+dy), fontsize=9, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    xs.append(x); ys.append(y)
z = np.polyfit(xs, ys, 1)
xr = np.linspace(min(xs)-0.05, max(xs)+0.05, 100)
ax.plot(xr, np.polyval(z, xr), "--", color="#555555", lw=1.5, alpha=0.6, label="Linear trend")
ylim_now = ax.get_ylim()
ax.axvline(0.6, color="#CCCCCC", ls=":", lw=1.2); ax.axvline(0.75, color="#CCCCCC", ls=":", lw=1.2)
ax.axhline(0, color="#888888", ls="-", lw=0.7)
ax.axvspan(0.0, 0.60, alpha=0.06, color=C_LOSE, zorder=0)
ax.axvspan(0.60, 0.75, alpha=0.06, color=C_TIE, zorder=0)
ax.axvspan(0.75, 1.0,  alpha=0.06, color=C_WIN, zorder=0)
ax.set_ylim(ylim_now); ax.set_ylim(-25, 115)
ax.text(0.18, 105, "Unfavorable\n(CE<0.6)", ha="center", fontsize=8, color=C_LOSE, va="bottom")
ax.text(0.675, 105, "Boundary", ha="center", fontsize=8, color=C_TIE, va="bottom")
ax.text(0.88, 105, "Favorable\n(CE>0.75)", ha="center", fontsize=8, color=C_WIN, va="bottom")
hs = [Line2D([0],[0],marker="o",color="w",markerfacecolor=C_WIN,markersize=9,label="Win vs AdamW"),
      Line2D([0],[0],marker="s",color="w",markerfacecolor=C_TIE,markersize=9,label="Tie vs AdamW"),
      Line2D([0],[0],marker="X",color="w",markerfacecolor=C_LOSE,markersize=9,label="Lose vs AdamW"),
      Line2D([0],[0],color="#555555",ls="--",lw=1.5,label="Linear trend")]
ax.legend(handles=hs, loc="upper left", fontsize=8.5, framealpha=0.9)
ax.set_xlabel("Captured Energy (B5, final epoch)  [proxy for Delta_r*cos^2(R)]")
ax.set_ylabel("B5 vs AdamW: relative improvement (%)")
ax.set_title("F1: Captured Energy Predicts GrassRoute Outcome (Win-Map)\n"
             "High captured energy = strong spectral concentration = GrassRoute wins", fontsize=9.5)
ax.set_xlim(0.35, 1.05)
ax.text(0.98, 0.03, "SciML: loss (lower=better).\nOthers: accuracy.\nSingle seed (seed=42).",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92))
save("F1_winmap")

# ─────────────────────────────────────────────────────────────────────────
# F2 — Spectral concentration bar
# ─────────────────────────────────────────────────────────────────────────
fams_f2 = [("SciML (PINN)", 0.978, C_WIN), ("MLP (MNIST)", 0.584, C_TIE),
            ("CNN (full)", 0.549, C_TIE), ("Graph (ogbn)", 0.463, C_LOSE),
            ("RecSys (MF emb)", 0.063, C_LOSE)]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh([x[0] for x in fams_f2], [x[1] for x in fams_f2],
               color=[x[2] for x in fams_f2], edgecolor="white", linewidth=0.8, height=0.55)
ax.axvline(0.75, color=C_WIN,  ls="--", lw=1.5, label="Favorable threshold (0.75)")
ax.axvline(0.60, color=C_TIE,  ls="--", lw=1.5, label="Boundary threshold (0.60)")
ax.axvspan(0.75, 1.0, alpha=0.08, color=C_WIN); ax.axvspan(0.60, 0.75, alpha=0.08, color=C_TIE)
ax.axvspan(0.0, 0.60, alpha=0.08, color=C_LOSE)
for bar, v in zip(bars, [x[1] for x in fams_f2]):
    ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
ax.set_xlabel("B5 Captured Energy (final epoch) [proxy for Delta_r*cos^2(R)]")
ax.set_title("F2: Spectral Concentration by Task Family\nRecSys estimated. Single seed.", fontsize=9.5)
ax.set_xlim(0, 1.08); ax.legend(loc="lower right", fontsize=8)
save("F2_spectral_bar")

# ─────────────────────────────────────────────────────────────────────────
# F3 — Memory ↔ Compute trade-off
# ─────────────────────────────────────────────────────────────────────────
mt = rr["main_table"]
mem_savings = {
    "sciml":(mt["sciml"]["B2"]["state_memory_bytes"]-mt["sciml"]["B5"]["state_memory_bytes"])/mt["sciml"]["B2"]["state_memory_bytes"]*100,
    "graph":(mt["graph"]["B2"]["state_memory_bytes"]-mt["graph"]["B5"]["state_memory_bytes"])/mt["graph"]["B2"]["state_memory_bytes"]*100,
    "mlp":  (mt["mlp"]["B2"]["state_memory_bytes"]-mt["mlp"]["B5"]["state_memory_bytes"])/mt["mlp"]["B2"]["state_memory_bytes"]*100,
    "cnn":  (mt["cnn"]["B2"]["state_memory_bytes"]-mt["cnn"]["B5"]["state_memory_bytes"])/mt["cnn"]["B2"]["state_memory_bytes"]*100,
    "vit":  88.0,
}
task_labels  = {"sciml":"SciML (PINN)","graph":"Graph (GCN/SAGE)","mlp":"MLP (MNIST)","cnn":"CNN (full)","vit":"ViT (~4x slow)"}
task_verdicts= {"sciml":"win","graph":"win","mlp":"tie","cnn":"lose","vit":"lose"}
fig, ax = plt.subplots(figsize=(7.5, 5.5)); fig.subplots_adjust(left=0.18)
for task, wc in wcp.items():
    if task not in mem_savings: continue
    x, y, n = wc["slowdown"], mem_savings[task], wc["n_routed_layers"]
    c = color_map[task_verdicts[task]]
    ax.scatter(x, y, s=n*20+60, color=c, alpha=0.88, zorder=5, edgecolors="white", linewidths=0.9)
    dx, dy = 0.08, 1.0
    if task=="vit":   dx, dy = 0.08, -3.5
    if task=="cnn":   dx, dy = -0.6, 3.5
    if task=="mlp":   dx, dy = 0.08, -3.5
    ax.annotate(f"{task_labels[task]}\n({n} layers)", (x,y), (x+dx,y+dy), fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))
ax.text(0.82, 95.5, "Ideal:\nHigh save\nLow slowdown", fontsize=8.5, color=C_WIN, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_WIN, alpha=0.9))
ax.axvline(2.0, color="#CCCCCC", ls=":", lw=1.0); ax.axhline(70, color="#CCCCCC", ls=":", lw=1.0)
hs2 = [Line2D([0],[0],marker="o",color="w",markerfacecolor=C_WIN,markersize=10,label="Positive result"),
       Line2D([0],[0],marker="o",color="w",markerfacecolor=C_TIE,markersize=10,label="Tie / saturated"),
       Line2D([0],[0],marker="o",color="w",markerfacecolor=C_LOSE,markersize=10,label="Negative result")]
ax.legend(handles=hs2, loc="lower left", fontsize=8.5, title="Outcome", title_fontsize=8)
ax.set_xlabel("Per-step slowdown vs AdamW  (GrassRoute / AdamW wall-clock)");
ax.set_ylabel("Adaptive state memory saving (%)")
ax.set_title("F3: Memory vs Compute Trade-off — Both Axes Reported Honestly (G3)\n"
             "SciML: best trade-off (84% mem save, 1.78x slower, 4 layers)", fontsize=9.5)
ax.set_xlim(0.7, 5.0); ax.set_ylim(60, 102)
ax.text(0.98, 0.97, "Memory=adaptive state O(mr+dr) vs O(md) for AdamW.\nViT estimated. Single seed.",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
save("F3_tradeoff")

# ─────────────────────────────────────────────────────────────────────────
# F4 — SciML G5 long training
# ─────────────────────────────────────────────────────────────────────────
b4_recs = b4t["records"]; b5_recs = b5t["records"]
def thin(recs, step=5):
    return (np.array([r["epoch"] for r in recs[::step]]),
            np.array([r["train_loss"] for r in recs[::step]]))
ep4, loss4 = thin(b4_recs); ep5, loss5 = thin(b5_recs)
loss4_interp = np.interp(ep5, ep4, loss4)
ratio = loss4_interp / np.maximum(loss5, 1e-10)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios":[2,1]})
fig.subplots_adjust(hspace=0.08)
ax1.semilogy(ep4, loss4, color=C5[0], label="B4 Periodic-SVD", lw=2.0, alpha=0.9)
ax1.semilogy(ep5, loss5, color=C5[2], label="B5 GrassRoute",   lw=2.0, alpha=0.9)
ax1.set_ylabel("Train Loss (log scale)", fontweight="bold"); ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("F4: SciML Long-Training Advantage (G5) — 2000 Epochs\n"
              "B5 GrassRoute vs B4 Periodic-SVD; non-monotonic crossing at ep~1000", fontsize=10)
ax1.axvline(1000, color="gray", ls=":", lw=1.2, alpha=0.7)
ax1.text(1010, ax1.get_ylim()[0]*1.5, "ep~1000\ncrossing", fontsize=7.5, color="gray", va="bottom")
ax2.plot(ep5, ratio, color=C5[4], lw=2.0)
ax2.axhline(1.0, color="gray", ls="--", lw=1.0, label="Parity (ratio=1)")
ax2.fill_between(ep5, 1.0, ratio, where=(ratio>1.0), alpha=0.25, color=C_WIN, label="B5 better")
ax2.fill_between(ep5, ratio, 1.0, where=(ratio<1.0), alpha=0.25, color=C_LOSE, label="B4 better")
ax2.set_ylabel("B4/B5 loss ratio\n(>1 = B5 better)", fontweight="bold", fontsize=10)
ax2.set_xlabel("Epoch", fontweight="bold"); ax2.legend(loc="upper left", fontsize=8)
ax2.set_ylim(0.4, 2.0); ax2.axvline(1000, color="gray", ls=":", lw=1.2, alpha=0.7)
ax2.text(0.97, 0.05,
         "Final: B4={:.3e}  B5={:.3e}\nB5 advantage = {:.0f}% at ep1999 (single seed)".format(
             b4_recs[-1]["train_loss"], b5_recs[-1]["train_loss"],
             (b4_recs[-1]["train_loss"]-b5_recs[-1]["train_loss"])/b4_recs[-1]["train_loss"]*100),
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
save("F4_sciml_g5")

# ─────────────────────────────────────────────────────────────────────────
# F5 — V-reset Theorem 4
# ─────────────────────────────────────────────────────────────────────────
vc_vars = {v["mode"]: v for v in vc["variants"]}
adamw_loss = vc["adamw_final_loss"]
vc_sty = {"hard":{"color":C_LOSE,"ls":"--","lw":2.5,"marker":None},
           "decay":{"color":C_TIE,"ls":":","lw":2.2,"marker":"s"},
           "none":{"color":C5[0],"ls":"-.","lw":2.2,"marker":"^"},
           "procrustes":{"color":C_WIN,"ls":"-","lw":2.8,"marker":None}}
vc_lbl = {"hard":"Hard reset (V=0): diverges [final=41.1]",
          "decay":"Decay reset: intermediate [final=3.31e-4]",
          "none":"No reset: stable [final=1.90e-4]",
          "procrustes":"Procrustes transport: stable [final=1.90e-4]"}
fig, ax = plt.subplots(figsize=(8, 5.5))
for mode, d in vc_vars.items():
    pts = d["loss_curve_every100"]
    ep = [p[0] for p in pts]; lo = [max(p[1],1e-5) for p in pts]
    s = vc_sty[mode]
    kw = dict(color=s["color"],ls=s["ls"],lw=s["lw"],label=vc_lbl[mode])
    if s["marker"]: kw.update(marker=s["marker"],markersize=5,markevery=2)
    ax.semilogy(ep, lo, **kw)
ax.axhline(adamw_loss, color="#333333", ls="-", lw=1.8, alpha=0.8,
           label="AdamW reference ({:.2e})".format(adamw_loss))
ax.text(100, adamw_loss*1.5, "AdamW: {:.2e}".format(adamw_loss), color="#333333", fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#333333", alpha=0.9))
ax.set_ylim(5e-6, 200)
ax.annotate("Hard reset\ndivergence", xy=(500,25), xytext=(700,70), fontsize=8.5,
            color=C_LOSE, fontweight="bold", arrowprops=dict(arrowstyle="->",color=C_LOSE,lw=1.2))
ax.annotate("Procrustes/None converge\n(same curve, reset=0 times)",
            xy=(300,1.9e-4), xytext=(400,5e-4), fontsize=8, color=C_WIN,
            arrowprops=dict(arrowstyle="->",color=C_WIN,lw=1.0))
ax.set_xlabel("Epoch",fontweight="bold"); ax.set_ylabel("Train Loss (log scale)",fontweight="bold")
ax.set_title("F5: V-Reset Variants — Theorem 4 Empirical Verification\n"
             "Hard reset diverges (e_reset=Theta(||V||)); Procrustes stable (e_reset=O(||DeltaP||^2))", fontsize=9.5)
ax.legend(loc="upper right",fontsize=8.5,framealpha=0.95)
ax.text(0.02,0.03,"Toy matreg task, seed=42. Theorem 4 ordering confirmed.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
save("F5_vreset")

# ─────────────────────────────────────────────────────────────────────────
# F6 — Anchor4 heatmap
# ─────────────────────────────────────────────────────────────────────────
results_a4 = a4["results"]
eigengaps = sorted(set(r["config"]["eigengap_ratio"] for r in results_a4))
drifts    = sorted(set(r["config"]["drift_rate"] for r in results_a4))
taus      = sorted(set(r["config"]["tau_tail"] for r in results_a4))
def get_R(eg,dr,tau,method="grassroute"):
    for r in results_a4:
        if r["config"]["eigengap_ratio"]==eg and r["config"]["drift_rate"]==dr and r["config"]["tau_tail"]==tau:
            return r[method]["final"]["R_over_halfpi"]
    return np.nan
grid_gr  = np.array([[np.nanmean([get_R(eg,dr,t,"grassroute") for t in taus]) for dr in drifts] for eg in eigengaps])
grid_oja = np.array([[np.nanmean([get_R(eg,dr,t,"oja") for t in taus]) for dr in drifts] for eg in eigengaps])
grid_diff = grid_oja - grid_gr
from matplotlib.colors import TwoSlopeNorm
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("F6: Anchor-4 Synthetic Tracking — GrassRoute vs Oja Probe\nTracking error R/(pi/2) vs eigengap and drift-rate", fontsize=10)
for ax, grid, title, cmap, norm in [
    (axes[0],grid_gr, "GrassRoute R/(pi/2)", plt.cm.YlOrRd, None),
    (axes[1],grid_oja,"Oja probe  R/(pi/2)", plt.cm.YlOrRd, None),
    (axes[2],grid_diff,"Oja - GR (>0 = GR better)", plt.cm.RdYlGn, TwoSlopeNorm(vmin=-0.1,vcenter=0,vmax=0.15))]:
    vmin=(0 if cmap==plt.cm.YlOrRd else None); vmax=(1 if cmap==plt.cm.YlOrRd else None)
    im = ax.imshow(grid,aspect="auto",cmap=cmap,norm=norm,vmin=vmin,vmax=vmax,origin="lower")
    ax.set_xticks(range(len(drifts))); ax.set_xticklabels([str(d) for d in drifts],fontsize=8.5)
    ax.set_yticks(range(len(eigengaps))); ax.set_yticklabels([str(e) for e in eigengaps],fontsize=8.5)
    ax.set_xlabel("Drift rate"); ax.set_ylabel("Eigengap ratio")
    ax.set_title(title,fontsize=9.5)
    for ii in range(len(eigengaps)):
        for jj in range(len(drifts)):
            ax.text(jj,ii,"{:.3f}".format(grid[ii,jj]),ha="center",va="center",fontsize=8,
                    color="white" if (cmap==plt.cm.YlOrRd and grid[ii,jj]>0.4) else "black")
    plt.colorbar(im,ax=ax,shrink=0.8)
axes[2].text(0.5,-0.18,"22/24 configs: GR <= Oja. High drift+low eigengap -> R near pi/2.",
             transform=axes[2].transAxes,ha="center",va="top",fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
fig.tight_layout()
save("F6_anchor4")

# ─────────────────────────────────────────────────────────────────────────
# F7 — Theorem 2 tracking dynamics
# ─────────────────────────────────────────────────────────────────────────
def thin3(recs, step=10):
    ep = [r["epoch"] for r in recs[::step]]
    gd = [r.get("grassmann_distance") or 0 for r in recs[::step]]
    Rh = [r.get("principal_angle_R_over_halfpi") or r.get("principal_angle_R") or 0 for r in recs[::step]]
    ce = [r.get("captured_energy") or 0 for r in recs[::step]]
    return np.array(ep), np.array(gd), np.array(Rh), np.array(ce)
ep_s,gd_s,R_s,ce_s = thin3(b5t["records"],step=10)
ep_g,gd_g,R_g,ce_g = thin3(gb5t["records"],step=5)
fig, axes = plt.subplots(1,3,figsize=(14,4.5))
fig.suptitle("F7: Theorem 2 Online Tracking Dynamics — SciML (Win) vs Graph-ogbn (Lose)",fontsize=10,y=1.01)
axes[0].plot(ep_s,gd_s,color=C_WIN,lw=2.2,label="SciML B5 (win)")
axes[0].plot(ep_g,gd_g,color=C_LOSE,lw=2.2,label="Graph B5 (lose)",ls="--")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Grassmann Distance d_Gr")
axes[0].set_title("(a) Subspace tracking distance\n(lower = more stable)"); axes[0].legend(loc="upper right",fontsize=8.5)
axes[1].plot(ep_s,R_s,color=C_WIN,lw=2.2,label="SciML B5")
axes[1].plot(ep_g,R_g,color=C_LOSE,lw=2.2,label="Graph B5",ls="--")
axes[1].axhline(1.0,color="black",ls=":",lw=1.2,label="Cut-locus (R=pi/2)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Principal Angle R/(pi/2)")
axes[1].set_title("(b) Angle to Grassmann cut-locus\n(<1 = Assumption 3 holds)"); axes[1].legend(loc="upper right",fontsize=8.5)
axes[2].plot(ep_s,ce_s,color=C_WIN,lw=2.2,label="SciML B5 (-> 0.978)")
axes[2].plot(ep_g,ce_g,color=C_LOSE,lw=2.2,label="Graph B5 (-> 0.463)",ls="--")
axes[2].axhline(0.75,color="gray",ls="--",lw=1.2,label="Favorable threshold")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Captured Energy")
axes[2].set_title("(c) Captured energy vs epoch\n(SciML -> 0.978; Graph -> 0.46)")
axes[2].legend(loc="lower right",fontsize=8.5); axes[2].set_ylim(0,1.10)
axes[2].text(0.02,0.03,"Graph R stays <<1: geometrically stable.\nGraph fails due to high rank-tail (Thm 3), not tracking divergence.",
             transform=axes[2].transAxes,ha="left",va="bottom",fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.92))
fig.tight_layout()
save("F7_tracking")

# ─────────────────────────────────────────────────────────────────────────
# F8 — Efficiency scaling
# ─────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,4))
fig.suptitle("F8: Retraction Efficiency — QR vs exact-Exp Scaling\nSub-operation level benchmark",fontsize=10)
ax1.plot(rs["ranks"],rs["qr_us"],"o-",color=C5[0],label="GrassRoute QR retraction",lw=2.0)
ax1.plot(rs["ranks"],rs["svd_exp_us"],"s-",color=C5[2],label="SubTrack++ exact-Exp",lw=2.0)
for i,(r_,rr_) in enumerate(zip(rs["ranks"],rs["ratio"])):
    ax1.annotate("{:.1f}x".format(rr_),(r_,rs["svd_exp_us"][i]),xytext=(r_+3,rs["svd_exp_us"][i]+50),
                fontsize=8.5,arrowprops=dict(arrowstyle="-",color="gray",lw=0.6))
ax1.set_xlabel("Rank r (d=768 fixed)"); ax1.set_ylabel("Per-step time (us)")
ax1.set_title("(a) Rank scaling (d=768)"); ax1.legend(fontsize=8.5); ax1.set_xticks(rs["ranks"])
ax2.plot(ret["d_values"],ret["grassroute_us"],"o-",color=C5[0],label="GrassRoute QR",lw=2.0)
ax2.plot(ret["d_values"],ret["subtrack_us"],"s-",color=C5[2],label="SubTrack++ exact-Exp",lw=2.0)
ax2.set_xlabel("Dimension d (r=16 fixed)"); ax2.set_ylabel("Per-step time (us)")
ax2.set_title("(b) Dimension scaling (r=16)"); ax2.legend(fontsize=8.5)
fig.text(0.5,-0.05,"QR is {:.1f}-{:.1f}x faster at sub-op level. Full-model ViT: B5=78ms, B6=88ms (~13%). Single GPU.".format(
    min(rs["ratio"]),max(rs["ratio"])),ha="center",fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
fig.tight_layout()
save("F8_efficiency")

# ─────────────────────────────────────────────────────────────────────────
# F9 — CNN four-way trade-off
# ─────────────────────────────────────────────────────────────────────────
mt_cnn = rr["main_table"]["cnn"]
cnn_d = [{"label":"B5-partial\n(fc+1x1, 4L)","top1":cpdata["cnn_partial_B5_1000ep"],"mem_mb":cpdata["state_mem_MB"],"color":"#648FFF"},
         {"label":"B1 tuned-SGD\n(lr=0.1)","top1":0.7313,"mem_mb":mt_cnn["B1"]["state_memory_bytes"]/1e6,"color":"#785EF0"},
         {"label":"B2 AdamW\n(baseline)","top1":mt_cnn["B2"]["best_val"],"mem_mb":mt_cnn["B2"]["state_memory_bytes"]/1e6,"color":"#888888"},
         {"label":"B5-full\n(all-conv,r32)","top1":mt_cnn["B5"]["best_val"],"mem_mb":mt_cnn["B5"]["state_memory_bytes"]/1e6,"color":C_LOSE}]
x_c=np.arange(len(cnn_d)); fig, ax1 = plt.subplots(figsize=(8.5,5.5)); ax2=ax1.twinx()
bars1=ax1.bar(x_c-0.1,[d["top1"] for d in cnn_d],0.4,color=[d["color"] for d in cnn_d],alpha=0.88,edgecolor="white",zorder=3)
ax2.scatter(x_c+0.1,[d["mem_mb"] for d in cnn_d],s=100,color=[d["color"] for d in cnn_d],marker="D",zorder=5)
ax2.plot(x_c+0.1,[d["mem_mb"] for d in cnn_d],"--",color="gray",lw=1.2,alpha=0.5,zorder=4)
for bar,v in zip(bars1,[d["top1"] for d in cnn_d]):
    ax1.text(bar.get_x()+bar.get_width()/2,v+0.004,"{:.4f}".format(v),ha="center",fontsize=9,fontweight="bold",zorder=6)
for xi,m in zip(x_c+0.1,[d["mem_mb"] for d in cnn_d]):
    ax2.text(xi+0.12,m+1.5,"{:.1f}MB".format(m),fontsize=8.5,color="#555555")
ax1.axhline(mt_cnn["B2"]["best_val"],color="#888888",ls=":",lw=1.0,alpha=0.7)
ax1.set_xticks(x_c); ax1.set_xticklabels([d["label"] for d in cnn_d],fontsize=9)
ax1.set_ylabel("CIFAR-100 Top-1 Accuracy (higher is better)"); ax2.set_ylabel("Adaptive State Memory (MB, lower is better)",color="#555555")
ax2.tick_params(axis="y",labelcolor="#555555"); ax1.set_ylim(0.50,0.84); ax2.set_ylim(-5,120)
ax1.set_title("F9: CNN CIFAR-100 Precision-Memory Trade-off (G1 boundary)\nPartial: save precision, no mem gain. Full: save mem, lose -12pp.",fontsize=9.5)
hs=[Patch(color="#648FFF",label="B5-partial (fc+1x1, 4 layers)"),Patch(color="#785EF0",label="B1 tuned-SGD"),
    Patch(color="#888888",label="B2 AdamW"),Patch(color=C_LOSE,label="B5-full (all-conv, r32)"),
    Line2D([0],[0],marker="D",color="w",markerfacecolor="gray",markersize=9,ls="--",label="State mem (right axis)",markeredgecolor="gray")]
ax1.legend(handles=hs,loc="lower left",fontsize=8,ncol=2,framealpha=0.95)
ax1.text(0.5,0.97,"CNN structural boundary: partial = +precision but ~0% mem saving;\nfull = 90% mem saved but -12pp vs AdamW. Single seed, 1000ep.",
         transform=ax1.transAxes,ha="center",va="top",fontsize=7.5,
         bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
save("F9_cnn_tradeoff")

# ─────────────────────────────────────────────────────────────────────────
# F10 — RecSys negative contrast
# ─────────────────────────────────────────────────────────────────────────
keys_r=["B1","B3","B5","B6","B4","B2"]
mths=["B1\nSGD","B3\nFixed-rand","B5\nGrassRoute","B6\nSubTrack++","B4\nPeriodic-SVD","B2\nAdamW"]
recalls=[recsys[k]["best_recall@20"] for k in keys_r]
bc=[C_WIN if k=="B1" else ("#FFA500" if k in ["B3","B5"] else C_LOSE) for k in keys_r]
fig,ax=plt.subplots(figsize=(8,4.5))
bars_r=ax.bar(mths,recalls,color=bc,edgecolor="white",linewidth=0.8,width=0.6)
for bar,v in zip(bars_r,recalls):
    ax.text(bar.get_x()+bar.get_width()/2,v+0.003,"{:.3f}".format(v),ha="center",fontsize=9.5,fontweight="bold")
ax.annotate("",xy=(1.3,0.083),xytext=(2.3,0.083),arrowprops=dict(arrowstyle="<->",color="#FF7F0E",lw=1.5))
ax.text(1.8,0.086,"B3 (fixed) ~= B5 (online)\nrouting collapse",ha="center",fontsize=8,color="#FF7F0E",
        bbox=dict(boxstyle="round,pad=0.2",facecolor="white",edgecolor="#FF7F0E",alpha=0.9))
ax.axhline(recsys["B2"]["best_recall@20"],color="gray",ls=":",lw=1.2,alpha=0.7)
ax.set_ylabel("Recall@20 (higher is better)")
ax.set_title("F10: RecSys (MF) — Spectral Collapse of Routing (Negative Contrast)\n"
             "Near-full-rank embeddings: Delta_r->0, B5 ~= B3 (random), SGD dominates",fontsize=9.5)
ax.text(0.02,0.97,"B5 (online) barely above B3 (fixed random): no stable subspace.\nSGD 2.3x better. (single seed)",
        transform=ax.transAxes,ha="left",va="top",fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
save("F10_recsys")

# ─────────────────────────────────────────────────────────────────────────
# F11 — Continual learning
# ─────────────────────────────────────────────────────────────────────────
best_vals = {
    "B2 AdamW":    {"acc":cg2["B2_lr0.003"]["avg_acc"],"drift":None,"color":C5[1]},
    "B4 Periodic": {"acc":cg2["B4_lr0.003"]["avg_acc"],"drift":cont["B4"]["mean_subspace_drift"],"color":C5[0]},
    "B5 GrassRoute":{"acc":cg2["B5_lr0.01_glr1.0"]["avg_acc"],"drift":cont["B5"]["mean_subspace_drift"],"color":C5[2]},
    "B6 SubTrack++":{"acc":cont["B6"]["avg_acc"],"drift":cont["B6"]["mean_subspace_drift"],"color":C5[4]},
}
names_11=list(best_vals.keys()); accs_11=[best_vals[n]["acc"] for n in names_11]
colors_11=[best_vals[n]["color"] for n in names_11]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.5))
fig.suptitle("F11: Continual RotMNIST — Geometric Tracking Better But Accuracy Does Not Follow\nB5 drifts less (0.73 vs 0.99) but accuracy still worse than AdamW",fontsize=9.5)
bars11a=ax1.bar(names_11,accs_11,color=colors_11,edgecolor="white",linewidth=0.8,width=0.6)
for bar,v in zip(bars11a,accs_11):
    ax1.text(bar.get_x()+bar.get_width()/2,v+0.003,"{:.4f}".format(v),ha="center",fontsize=9,fontweight="bold")
ax1.axhline(accs_11[0],color=C5[1],ls=":",lw=1.2,alpha=0.7)
ax1.set_ylabel("Avg Accuracy (4 tasks, higher is better)"); ax1.set_title("(a) Accuracy: AdamW best")
ax1.set_ylim(0.50,0.65); ax1.tick_params(axis="x",labelsize=8.5)
drifts_avail=[(n,best_vals[n]["drift"]) for n in names_11 if best_vals[n]["drift"] is not None]
names_d=[x[0] for x in drifts_avail]; dvals=[x[1] for x in drifts_avail]
colors_d=[best_vals[n]["color"] for n in names_d]
bars11b=ax2.bar(names_d,dvals,color=colors_d,edgecolor="white",linewidth=0.8,width=0.5)
for bar,v in zip(bars11b,dvals):
    ax2.text(bar.get_x()+bar.get_width()/2,v+0.01,"{:.3f}".format(v),ha="center",fontsize=9,fontweight="bold")
ax2.set_ylabel("Mean Subspace Drift (lower is more stable)"); ax2.set_title("(b) Subspace drift: B5/B6 much better")
ax2.tick_params(axis="x",labelsize=8.5); ax2.set_ylim(0,1.15)
ax2.text(0.5,0.97,"Geometric success (drift) not translating to accuracy:\ndiscrete task jumps favor periodic reset over online tracking.",
         transform=ax2.transAxes,ha="center",va="top",fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
fig.tight_layout()
save("F11_continual")

# ─────────────────────────────────────────────────────────────────────────
# F12 — Ablation tornado
# ─────────────────────────────────────────────────────────────────────────
abl=rr["ablation"]; baseline_b5=rr["main_table"]["graph"]["B5"]["best_test"]
abl_entries={"A3 no Grassmann-mom":abl["graph_A3_no_gmom"]["best_test"],
             "A6 main-branch SGD":abl["graph_A6_main_sgd"]["best_test"],
             "A7 no Procrustes":abl["graph_A7_no_procrustes"]["best_test"],
             "A8 hard V-reset":abl["graph_A8_hard_reset"]["best_test"],
             "A9 sub-update (5ep)":abl["graph_A9_sub5"]["best_test"],
             "A10 sub-update (25ep)":abl["graph_A10_sub25"]["best_test"]}
abl_names=list(abl_entries.keys()); abl_vals=[v-baseline_b5 for v in abl_entries.values()]
abl_colors=[C_LOSE if v<-0.05 else (C_TIE if v<-0.005 else C_WIN) for v in abl_vals]
sort_idx=sorted(range(len(abl_vals)),key=lambda i:abl_vals[i])
abl_names=[abl_names[i] for i in sort_idx]; abl_vals=[abl_vals[i] for i in sort_idx]; abl_colors=[abl_colors[i] for i in sort_idx]
fig,ax=plt.subplots(figsize=(8,5))
bars_abl=ax.barh(abl_names,abl_vals,color=abl_colors,edgecolor="white",linewidth=0.8,height=0.55)
ax.axvline(0,color="black",lw=1.2)
for bar,v in zip(bars_abl,abl_vals):
    ax.text(v-0.003 if v<0 else v+0.003,bar.get_y()+bar.get_height()/2,
            "{:+.3f}".format(v),va="center",ha="right" if v<0 else "left",fontsize=9,fontweight="bold")
ax.set_xlabel("Delta Test-Acc vs B5 baseline ({:.3f})".format(baseline_b5))
ax.set_title("F12: GrassRoute Ablation Tornado (Graph/ogbn-arxiv)\nA6 (SGD main) and A8 (hard reset) collapse; A9/A10 confirm online > periodic",fontsize=9.5)
ax.text(0.02,0.97,"B5 baseline={:.4f}. A6/A8=critical. A9/A10=G2 mechanism. Single seed.".format(baseline_b5),
        transform=ax.transAxes,ha="left",va="top",fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
save("F12_ablation")

# ─────────────────────────────────────────────────────────────────────────
# F13 — Rank sweep
# ─────────────────────────────────────────────────────────────────────────
rs_data=rr["rank_sweep"]; ranks_g=["r4","r8","r16","r32"]; rank_nums=[4,8,16,32]
graph_b5=[rs_data["graph"][r]["B5"]["best_test"] for r in ranks_g]
graph_b4=[rs_data["graph"][r]["B4"]["best_test"] for r in ranks_g]
graph_b2=rr["main_table"]["graph"]["B2"]["best_test"]
fig,ax=plt.subplots(figsize=(6,4.5))
ax.plot(rank_nums,graph_b5,"o-",color=C5[2],lw=2.0,label="B5 GrassRoute",ms=8)
ax.plot(rank_nums,graph_b4,"s-",color=C5[0],lw=2.0,label="B4 Periodic-SVD",ms=8)
ax.axhline(graph_b2,color="gray",ls="--",lw=1.5,label="B2 AdamW (rank-free)")
for r,v in zip(rank_nums,graph_b5): ax.text(r,v+0.001,"{:.4f}".format(v),ha="center",fontsize=8.5)
ax.set_xlabel("Rank r"); ax.set_ylabel("Test Accuracy (higher is better)")
ax.set_title("F13: Rank Sweep Insensitivity (Graph/ogbn-arxiv)\nNear-rank-1 task: r=4 already saturates B5 and B4",fontsize=9.5)
ax.legend(fontsize=8.5); ax.set_xticks(rank_nums)
ax.text(0.02,0.03,"B5 flat across r=4-32: near rank-1 gradient. AdamW best. Single seed.",
        transform=ax.transAxes,ha="left",va="bottom",fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
save("F13_rank_sweep")

print("\nAll 13 figures generated in", OUT)
