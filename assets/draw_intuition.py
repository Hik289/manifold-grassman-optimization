"""Draw the GrassRoute method diagram from vector shapes and editable labels.

The diagram shows the diagonal second-moment variant used in the later
experiments. The geometry sketches are illustrative, not measured trajectories.
Run from the repository root with Python, NumPy, and Matplotlib installed.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np


INK = "#243342"
MUTED = "#5B6875"
BLUE = "#2867A8"
GREEN = "#398369"
ORANGE = "#C17B31"
PURPLE = "#8566AA"
BLUE_FILL = "#EDF4FC"
GREEN_FILL = "#EEF7F1"
ORANGE_FILL = "#FFF5E6"
PURPLE_FILL = "#F4EFF9"


def draw(output, preview=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in available), "DejaVu Sans")
    style = {
        "font.family": family,
        "mathtext.fontset": "stix",
        "font.size": 20,
        "text.color": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14.4, 7.5), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 1440), ylim=(750, 0))
        ax.axis("off")
        labels = []

        def text(x, y, s, size=21, color=INK, ha="center", weight="normal", width=None, z=5):
            artist = ax.text(x, y, s, ha=ha, va="center", fontsize=size * .92,
                             color=color, fontweight=weight, zorder=z,
                             linespacing=1.2)
            if width is not None:
                labels.append((artist, width))
            return artist

        def box(x, y, w, h, fill="white", edge=INK, radius=12, lw=1.0):
            artist = FancyBboxPatch((x, y), w, h,
                                   boxstyle=f"round,pad=0,rounding_size={radius}",
                                   facecolor=fill, edgecolor=edge,
                                   linewidth=lw * .72, zorder=1)
            ax.add_patch(artist)
            return artist

        def arrow(start, end, color=INK, lw=2.1, rad=0, dash=None, z=3):
            artist = FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=15, linewidth=lw * .72,
                                     color=color, connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=0, shrinkB=0, zorder=z,
                                     linestyle=dash or "-")
            ax.add_patch(artist)
            return artist

        def elbow(points, color=INK, lw=2.1, dash=None):
            if len(points) > 2:
                xx, yy = zip(*points[:-1])
                ax.plot(xx, yy, color=color, lw=lw * .72,
                        linestyle=dash or "-", solid_joinstyle="round", zorder=2)
            arrow(points[-2], points[-1], color=color, lw=lw, dash=dash)

        def heading(x, y, number, title, color, fill, width):
            box(x, y - 19, width, 39, fill=fill, edge="none", radius=17)
            ax.add_patch(Circle((x + 21, y), 13.5, facecolor=color, edgecolor="none", zorder=3))
            text(x + 21, y + .5, str(number), 20, "white", weight="bold")
            text(x + 45, y, title, 25, ha="left", weight="bold", width=width-58)

        def grid(x, y, rows, cols, cell, color, label=None, width=None):
            base = np.asarray(colors.to_rgb(color))
            for i in range(rows):
                for j in range(cols):
                    # Fixed, schematic shading; no experimental values are encoded.
                    strength = .22 + .16 * ((i + 2*j) % 4)
                    fill = base * strength + (1-strength)
                    ax.add_patch(Polygon([(x+j*cell,y+i*cell), (x+(j+1)*cell,y+i*cell),
                                          (x+(j+1)*cell,y+(i+1)*cell), (x+j*cell,y+(i+1)*cell)],
                                         closed=True, facecolor=fill, edgecolor=color,
                                         linewidth=.5, zorder=3))
            if label:
                text(x + cols*cell/2, y-17, label, 23, width=width)

        def plane(vertices, color, fill, alpha=1, dashed=False):
            v = np.asarray(vertices, dtype=float)
            ax.add_patch(Polygon(v, closed=True, facecolor=fill, edgecolor=color,
                                 linewidth=1.25*.72, alpha=alpha,
                                 linestyle=(0,(4,3)) if dashed else "-", zorder=1))
            for s in [.25,.5,.75]:
                for a,b,c,d in [(v[0],v[1],v[3],v[2]),(v[0],v[3],v[1],v[2])]:
                    p=(1-s)*a+s*b;q=(1-s)*c+s*d
                    ax.plot([p[0],q[0]],[p[1],q[1]],color=color,alpha=.18*alpha,lw=.7,zorder=1)

        # Weight update: a visual split, two branches, and one parameter update.
        heading(31, 44, 1, "Split the gradient", BLUE, BLUE_FILL, 350)
        heading(459, 44, 2, "Update the weights", BLUE, BLUE_FILL, 454)

        plane([(69,305),(259,341),(340,259),(150,223)], BLUE, "#EAF2FB")
        origin=(130,286); projected=(270,287); full=(285,126)
        arrow(origin, projected, BLUE, 2.8)
        arrow(projected, full, ORANGE, 2.8)
        arrow(origin, full, INK, 2.8)
        ax.add_patch(Circle(origin, 4, facecolor=INK, edgecolor="white", lw=.6,zorder=5))
        text(196,176,r"$G_t$",29)
        text(202,309,r"$Z_tU_t^{\top}$",25,BLUE)
        text(313,197,r"$G_t^{\perp}$",27,ORANGE)
        text(163,343,r"basis $U_t$",21,BLUE)
        text(205,379,r"$G_t=Z_tU_t^{\top}+G_t^{\perp}$",25,width=340)

        # Branching is separate from the illustrative geometry, avoiding implied
        # numerical angles or a particular embedding of the Grassmann manifold.
        ax.add_patch(Circle((405,245),4,facecolor=INK,edgecolor="none",zorder=4))
        arrow((350,245),(399,245),INK,1.8)
        elbow([(405,245),(405,173),(459,173)],BLUE,2.3)
        elbow([(405,245),(405,343),(459,343)],ORANGE,2.3)
        text(421,145,r"$Z_t$",24,BLUE)
        text(419,371,r"$G_t^{\perp}$",24,ORANGE)

        box(459,99,454,157,BLUE_FILL,BLUE,13,1.15)
        text(486,122,"Adam in the learned subspace",23,ha="left",weight="bold",width=405)
        grid(500,165,4,2,12.5,BLUE,r"$M_t$")
        grid(550,165,4,2,12.5,BLUE,r"$V_t$")
        arrow((603,188),(649,188),BLUE,2.1)
        text(777,177,r"$\eta_tD_t^{\parallel}U_t^{\top}$",30,BLUE,width=224)
        text(777,216,"Map back to full weights",18.5,BLUE,width=245)
        text(537,237,r"$m\times r$ each",19,BLUE)

        box(459,294,454,103,ORANGE_FILL,ORANGE,13,1.15)
        text(486,319,"Keep the residual",23,ha="left",weight="bold")
        text(600,358,"No stored moments",19,ORANGE,width=240)
        text(793,353,r"$\eta_{\perp,t}s_tG_t^{\perp}$",29,ORANGE,width=220)

        elbow([(913,178),(1036,178),(1036,224)],BLUE,2.3)
        elbow([(913,350),(1036,350),(1036,266)],ORANGE,2.3)
        ax.add_patch(Circle((1036,245),21,facecolor="white",edgecolor=INK,lw=1.0,zorder=4))
        text(1036,245,"+",32)
        arrow((1059,245),(1100,245),INK,2.2)

        box(1100,99,305,298,"#F8F9FB","#9BA9B6",13,1.0)
        text(1252,126,"Updated weights",24,weight="bold")
        grid(1198,175,4,5,17,PURPLE,r"$W_{t+1}$")
        text(1252,276,r"$W_{t+1}=(1-\eta_t\lambda_{\rm wd})W_t$",24,width=280)
        text(1252,316,r"$-\,\eta_tD_t^{\parallel}U_t^{\top}$",25,width=266)
        text(1252,354,r"$-\,\eta_{\perp,t}s_tG_t^{\perp}$",25,width=266)

        # The basis update uses this same gradient, then re-expresses the moments
        # before the next optimizer step. Green and purple identify the two roles.
        heading(31,455,3,"Track the subspace",GREEN,GREEN_FILL,679)
        heading(772,455,4,"Align the stored moments",PURPLE,PURPLE_FILL,633)
        box(31,495,679,197,GREEN_FILL,GREEN,13,1.0)
        box(772,495,633,197,PURPLE_FILL,PURPLE,13,1.0)

        plane([(73,617),(234,647),(278,597),(117,567)],"#8F9BA4","#F2F4F6",alpha=.9,dashed=True)
        plane([(102,632),(248,606),(228,532),(82,558)],GREEN,"#D7EBDD",alpha=.95)
        arrow((238,635),(266,552),GREEN,2.1,rad=.38)
        text(86,657,r"$U_t$",24,MUTED)
        text(176,543,r"$U_{t+1}$",25,GREEN)
        text(185,677,"Basis step + QR",18.5,GREEN,width=250)
        text(493,534,r"$Q_t=(I-U_tU_t^{\top})\widehat C_tU_t$",26,GREEN,width=385)
        text(493,575,"Grassmann momentum",21,GREEN)
        arrow((493,592),(493,611),GREEN,1.6)
        text(493,638,r"$U_{t+1}=\operatorname{qf}(U_t+\rho H_t)$",26,GREEN,width=391)
        text(493,676,r"Use $G_t^{\top}Z_t$; no $d\times d$ covariance",18.5,GREEN,width=400)
        arrow((710,590),(772,590),GREEN,2.2)

        text(1087,530,r"$R_t=\operatorname{Polar}(U_{t+1}^{\top}U_t)$",27,PURPLE,width=575)
        grid(809,584,3,3,17,PURPLE,r"$R_t$")
        arrow((889,611),(932,611),PURPLE,2.0)
        text(1146,584,r"$M_t\leftarrow M_tR_t^{\top}$",27,PURPLE,width=443)
        text(1146,630,r"$V_t\leftarrow V_t(R_t^{\top})^{\odot 2}$",27,PURPLE,width=443)
        text(1087,674,"Diagonal second-moment approximation",18.5,PURPLE,width=575)

        # A short footer makes the iteration order explicit without a long,
        # visually dominant return arrow around the diagram.
        text(720,730,r"Next step: reuse $U_{t+1}$ and the aligned $M_t,V_t$.",21,MUTED)

        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        scale=ax.transData.transform((1,0))[0]-ax.transData.transform((0,0))[0]
        for artist,width in labels:
            measured=artist.get_window_extent(renderer).width
            if measured > width*scale:
                artist.set_fontsize(artist.get_fontsize()*width*scale/measured)
        fig.canvas.draw()
        # All text must remain legible after reduction to a 7.1-inch paper figure.
        minimum=min(a.get_fontsize() for a in ax.texts)
        if minimum < 16.0:
            small=[(a.get_text(),round(a.get_fontsize(),2)) for a in ax.texts if a.get_fontsize()<16.0]
            raise RuntimeError(f"Diagram text is too small before print reduction: {small}")
        fig.savefig(output.with_suffix(".png"),dpi=300,facecolor="white")
        fig.savefig(output.with_suffix(".pdf"),facecolor="white",metadata={"Creator":"Matplotlib","CreationDate":None})
        fig.savefig(output.with_suffix(".svg"),facecolor="white",metadata={"Date":None})
        svg=output.with_suffix(".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
        if preview:
            fig.savefig(preview,dpi=80,facecolor="white")
        plt.close(fig)
        print(output.with_suffix(".png"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("grassroute_intuition.png"))
    parser.add_argument("--preview",type=Path)
    args=parser.parse_args()
    draw(args.output,args.preview)
