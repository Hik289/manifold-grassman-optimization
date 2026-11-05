"""Draw the GrassRoute README method figure as a high-resolution PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#17263b"
MUTED = "#5e7084"
LINE = "#dce5ef"
BG = "#f7f9fc"
BLUE = "#2870c7"
BLUE_BG = "#edf5ff"
ORANGE = "#df7842"
ORANGE_BG = "#fff2e9"
TEAL = "#168b88"
TEAL_BG = "#eaf8f6"
GREEN = "#2c9972"
GREEN_BG = "#edf8f2"
PURPLE = "#8655b6"
PURPLE_BG = "#f5effc"


def box(ax, x, y, w, h, border, fill="white", radius=.14, lw=1.7):
    shape = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=fill, edgecolor=border, linewidth=lw,
    )
    ax.add_patch(shape)


def text(ax, x, y, value, size=16, color=INK, weight="normal", ha="left"):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center")


def arrow(ax, a, b, color=BLUE, width=2.3, scale=18):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale,
                                 linewidth=width, color=color, shrinkA=0, shrinkB=0))


def matrix(ax, x, y, rows, cols, color, label, dims, cell=.23):
    width, height = cols * cell, rows * cell
    left, bottom = x - width / 2, y - height / 2
    text(ax, x, bottom + height + .23, label, 16, ha="center")
    for i in range(rows):
        for j in range(cols):
            alpha = (.13, .23, .35, .47)[(3 * i + 2 * j) % 4]
            ax.add_patch(Rectangle((left + j * cell, bottom + (rows - 1 - i) * cell),
                                   cell, cell, facecolor=color, edgecolor=color,
                                   linewidth=.7, alpha=alpha))
    ax.add_patch(Rectangle((left, bottom), width, height, facecolor="none",
                           edgecolor=color, linewidth=1.55))
    text(ax, x, bottom - .25, dims, 12.5, MUTED, ha="center")


def header(ax, x, y, w, number, title, color):
    box(ax, x + .23, y + .23, .43, .40, color, color, radius=.18, lw=0)
    text(ax, x + .445, y + .43, str(number), 12, "white", "bold", "center")
    text(ax, x + .79, y + .43, title, 17, INK, "bold")
    ax.plot([x + .23, x + w - .23], [y + .23, y + .23], color=LINE, lw=1)


def draw():
    output = Path(__file__).with_name("grassroute_intuition.png")
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "savefig.facecolor": BG}):
        fig, ax = plt.subplots(figsize=(19.5, 8.5), dpi=190)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.set_xlim(0, 19.5)
        ax.set_ylim(0, 8.5)
        ax.axis("off")

        text(ax, .35, 8.16, "GrassRoute", 24, INK, "bold")
        text(ax, 3.43, 8.16, "Online routing of gradients through a learned Grassmann subspace",
             15, MUTED)
        ax.plot([.35, 19.15], [7.86, 7.86], color=LINE, lw=1.1)

        top_y, top_h, top_w = 4.35, 3.17, 4.40
        xs = [.35, 5.15, 9.95, 14.75]
        for k, (x, color, title) in enumerate(zip(
            xs, [ORANGE, TEAL, BLUE, PURPLE],
            ["Gradient + basis", "Orthogonal routing",
             "Adaptive coordinates", "Lift + residual step"],
        ), 1):
            box(ax, x, top_y, top_w, top_h, color)
            header(ax, x, top_y + top_h - .79, top_w, k, title, color)
        for x in [4.75, 9.55, 14.35]:
            arrow(ax, (x, 5.91), (x + .39, 5.91), BLUE, 2.6, 19)

        matrix(ax, 1.45, 5.95, 4, 3, ORANGE, r"$G_t$", r"$m\times d$")
        matrix(ax, 3.25, 5.95, 4, 2, BLUE, r"$U_t$", r"$d\times r$")
        box(ax, .59, 4.48, 3.92, .55, ORANGE, ORANGE_BG, radius=.09, lw=.9)
        text(ax, 2.55, 4.755, r"$G_t=\nabla_W\ell_t,\quad U_t^{\top}U_t=I_r$",
             15, INK, ha="center")

        matrix(ax, 6.22, 5.95, 4, 2, TEAL, r"$Z_t=G_tU_t$", r"$m\times r$")
        matrix(ax, 8.17, 5.95, 4, 3, GREEN, r"$G_t^{\perp}$", r"$m\times d$")
        box(ax, 5.39, 4.48, 3.92, .55, TEAL, TEAL_BG, radius=.09, lw=.9)
        text(ax, 7.35, 4.755, r"$G_t^{\perp}=G_t-Z_tU_t^{\top}$",
             15.5, INK, ha="center")

        matrix(ax, 11.02, 5.95, 3, 2, TEAL, r"$M_t$", r"$m\times r$")
        matrix(ax, 12.40, 5.95, 3, 2, TEAL, r"$V_t$", r"$m\times r$")
        matrix(ax, 13.76, 5.95, 3, 2, BLUE, r"$D_t^{\parallel}$", r"$m\times r$")
        box(ax, 10.19, 4.48, 3.92, .55, BLUE, BLUE_BG, radius=.09, lw=.9)
        text(ax, 12.15, 4.755, r"$D_t^{\parallel}=\mathrm{Adam}(Z_t;M_t,V_t)$",
             14.7, INK, ha="center")

        matrix(ax, 16.02, 5.95, 4, 3, PURPLE, r"$W_t$", r"$m\times d$")
        matrix(ax, 18.00, 5.95, 4, 3, PURPLE, r"$W_{t+1}$", r"$m\times d$")
        arrow(ax, (16.68, 6.02), (17.32, 6.02), PURPLE, 2.2, 16)
        box(ax, 14.99, 4.48, 3.92, .55, PURPLE, PURPLE_BG, radius=.09, lw=.9)
        text(ax, 16.95, 4.755,
             r"$\Delta W_t=-\eta_{\parallel}D_t^{\parallel}U_t^{\top}-\eta_{\perp}s_tG_t^{\perp}$",
             12.8, INK, ha="center")

        for x, color, title in [
            (.35, TEAL, "Grassmann ascent"),
            (6.82, BLUE, "Momentum + QR retraction"),
            (13.29, GREEN, "Procrustes state alignment"),
        ]:
            box(ax, x, .78, 5.86, 3.18, color)
            text(ax, x + .27, 3.58, title, 17, INK, "bold")
            ax.plot([x + .25, x + 5.61], [3.30, 3.30], color=LINE, lw=1)

        text(ax, .68, 2.81, r"$C_tU_t=G_t^{\top}(G_tU_t)$", 18)
        text(ax, .68, 2.25,
             r"$Q_t=\frac{(I-U_tU_t^{\top})G_t^{\top}Z_t}{\|G_t\|_F^2+\varepsilon}$",
             18, TEAL)
        box(ax, .66, 1.18, 5.18, .56, TEAL, TEAL_BG, radius=.08, lw=.8)
        text(ax, 3.25, 1.46, r"$U_t^{\top}Q_t=0$; no $d\times d$ covariance",
             13.5, INK, ha="center")

        text(ax, 7.15, 2.83,
             r"$H_t=\Pi_{U_t}\!\left(\mu H_{t-1}+(1-\mu)Q_t\right)$", 16)
        text(ax, 7.15, 2.19, r"$U_{t+1}=\mathrm{qf}(U_t+\rho H_t)$",
             18, BLUE)
        box(ax, 7.13, 1.18, 5.18, .56, BLUE, BLUE_BG, radius=.08, lw=.8)
        text(ax, 9.72, 1.46, r"$U_{t+1}^{\top}U_{t+1}=I_r$", 14.3, INK, ha="center")

        text(ax, 13.62, 2.83,
             r"$R_t=\mathrm{Polar}(U_{t+1}^{\top}U_t)$", 16)
        text(ax, 13.62, 2.20, r"$M_t\leftarrow M_tR_t^{\top}$", 17, GREEN)
        text(ax, 16.01, 2.20, r"$V_t\leftarrow V_t(R_t^{\top})^{\odot 2}$",
             15, GREEN)
        box(ax, 13.60, 1.18, 5.18, .56, GREEN, GREEN_BG, radius=.08, lw=.8)
        text(ax, 16.19, 1.46, "Persistent state follows the moving frame",
             13, INK, ha="center")

        arrow(ax, (6.21, 2.43), (6.81, 2.43), TEAL, 2.3, 18)
        arrow(ax, (12.68, 2.43), (13.28, 2.43), BLUE, 2.3, 18)
        arrow(ax, (5.70, 4.34), (5.70, 3.98), TEAL, 2.0, 15)
        arrow(ax, (16.95, 4.33), (16.95, 3.98), PURPLE, 2.0, 15)

        ax.plot([16.2, 16.2, 3.2], [.75, .20, .20], color=PURPLE, lw=1.7)
        arrow(ax, (3.2, .20), (3.2, .77), PURPLE, 1.7, 13)
        text(ax, 9.78, .52, "aligned state feeds the next iteration",
             11.5, PURPLE, ha="center")

        fig.savefig(output, dpi=190, bbox_inches="tight", pad_inches=.10)
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    draw()
