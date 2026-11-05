"""Draw the GrassRoute README intuition graphic as a high-resolution PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


NAVY = "#14233b"
MUTED = "#64748b"
BLUE = "#2563eb"
BLUE_BG = "#eaf2ff"
TEAL = "#0f8b79"
TEAL_BG = "#e9f8f4"
AMBER = "#bc6c17"
AMBER_BG = "#fff4df"
LINE = "#dce5ee"


def card(ax, x, y, w, h, face="white", edge=LINE, radius=.18):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.015,rounding_size={radius}",
                           facecolor=face, edgecolor=edge, linewidth=1.3)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=BLUE, width=2.2, size=15, curve=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=size, linewidth=width,
                                 color=color, connectionstyle=f"arc3,rad={curve}",
                                 shrinkA=0, shrinkB=0))


def label(ax, x, y, text, size=13, color=NAVY, weight="normal", **kwargs):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            va="center", **kwargs)


def main():
    out = Path(__file__).with_name("grassroute_intuition.png")
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "savefig.facecolor": "#f7f9fc"}):
        fig, ax = plt.subplots(figsize=(14.8, 6.1), dpi=220)
        fig.patch.set_facecolor("#f7f9fc")
        ax.set_facecolor("#f7f9fc")
        ax.set_xlim(0, 14.8)
        ax.set_ylim(0, 6.1)
        ax.axis("off")

        label(ax, .35, 5.73, "GrassRoute", 23, NAVY, "bold")
        label(ax, 3.40, 5.73, "Online adaptive updates in a learned subspace", 13.5, MUTED)
        ax.plot([.35, 14.45], [5.38, 5.38], color=LINE, linewidth=1)

        # Input and decomposition.
        card(ax, .35, 1.55, 3.45, 3.46)
        ax.add_patch(Circle((.72, 4.59), .15, facecolor=BLUE, edgecolor="none"))
        label(ax, .72, 4.59, "1", 10, "white", "bold", ha="center")
        label(ax, 1.02, 4.59, "Route the gradient", 15, NAVY, "bold")
        label(ax, .65, 3.84, r"$G_t \in \mathbb{R}^{m\times d}$", 19)
        label(ax, .65, 3.34, r"$U_t \in \mathrm{Gr}(r,d)$", 18)
        ax.plot([.65, 3.45], [2.98, 2.98], color=LINE, linewidth=1)
        label(ax, .65, 2.55, r"$Z_t = G_tU_t$", 17, BLUE, "bold")
        label(ax, .65, 2.04, r"$G_t^{\perp}=G_t-Z_tU_t^{\top}$", 16, TEAL)

        # Two paths and their update rules.
        arrow(ax, (3.8, 3.7), (4.22, 3.7), BLUE)
        arrow(ax, (3.8, 2.3), (4.22, 2.3), TEAL)
        card(ax, 4.25, 3.05, 5.72, 1.95, BLUE_BG, "#c6d8fa")
        label(ax, 4.58, 4.61, "LOW-RANK ADAPTIVE BRANCH", 10.5, BLUE, "bold")
        label(ax, 4.58, 4.03, r"$\mathrm{Adam}(Z_t)\;\rightarrow\; D_t^{\parallel}$", 20)
        label(ax, 4.58, 3.43, r"$\Delta W_{\parallel}=-\eta_{\parallel}D_t^{\parallel}U_t^{\top}$", 17)

        card(ax, 4.25, 1.35, 5.72, 1.45, TEAL_BG, "#b9e9db")
        label(ax, 4.58, 2.47, "ORTHOGONAL RESIDUAL BRANCH", 10.5, TEAL, "bold")
        label(ax, 4.58, 1.87, r"$\Delta W_{\perp}=-\eta_{\perp}s_tG_t^{\perp}$", 18)

        # Merge, weight update, and feedback.
        arrow(ax, (9.97, 3.91), (10.48, 3.47), BLUE)
        arrow(ax, (9.97, 2.08), (10.48, 2.73), TEAL)
        card(ax, 10.52, 1.55, 3.93, 3.46)
        ax.add_patch(Circle((10.89, 4.59), .15, facecolor=AMBER, edgecolor="none"))
        label(ax, 10.89, 4.59, "2", 10, "white", "bold", ha="center")
        label(ax, 11.19, 4.59, "Merge and track", 15, NAVY, "bold")
        label(ax, 10.84, 3.82, r"$W_{t+1}=W_t+\Delta W_{\parallel}$", 16)
        label(ax, 11.62, 3.35, r"$+\Delta W_{\perp}$", 16)
        ax.plot([10.84, 14.1], [3.0, 3.0], color=LINE, linewidth=1)
        label(ax, 10.84, 2.56, r"$Q_t=(I-U_tU_t^{\top})G_t^{\top}Z_t$", 14.2, AMBER)
        label(ax, 10.84, 2.05, r"$U_{t+1}=\mathrm{qf}(U_t+\rho H_t)$", 14.5)

        card(ax, .35, .35, 14.1, .67, AMBER_BG, "#f1d5a8", radius=.12)
        label(ax, .64, .69, "ONLINE GEOMETRY", 10, AMBER, "bold")
        label(ax, 3.0, .69,
              r"Track the changing gradient subspace on $\mathrm{Gr}(r,d)$; align stored moments by Procrustes rotation.",
              11.2, NAVY)
        arrow(ax, (12.4, 1.53), (12.4, 1.05), AMBER, 1.8, 12)
        arrow(ax, (2.1, 1.05), (2.1, 1.51), AMBER, 1.8, 12)

        fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
