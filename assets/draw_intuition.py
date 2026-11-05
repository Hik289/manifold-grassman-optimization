"""Draw the detailed GrassRoute overview; run with Python + Matplotlib + NumPy.

Five upper panels, three lower panels, condensed headings, STIX mathematics,
shaded matrix glyphs, saturated outlines, and the outer state-feedback path.
Arial Narrow is used when installed; DejaVu Sans is the portable fallback.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

BLUE='#087bff'
GREEN='#00ba62'
ORANGE='#ff6700'
PURPLE='#a239ff'
DARKBLUE='#002cbb'
DARKGREEN='#005b4d'
DARKRED='#970c00'
DARKPURPLE='#5700c8'


def draw():
    available={f.name for f in font_manager.fontManager.ttflist}
    heading_font='Arial Narrow' if 'Arial Narrow' in available else 'DejaVu Sans'
    with plt.rc_context({'font.family':heading_font, 'mathtext.fontset':'stix',
                         'font.size':14, 'savefig.facecolor':'white'}):
        fig=plt.figure(figsize=(19.33001,8.13001),dpi=200)
        ax=fig.add_axes([0,0,1,1])
        ax.set_xlim(0,1933);ax.set_ylim(813,0);ax.axis('off')
        fitted=[]

        def label(x,y,s,size=25,color='black',weight='normal',align='center',width=None,bg=None):
            t=ax.text(x,y,s,fontsize=size*.72,color=color,fontweight=weight,
                      ha=align,va='center',family=heading_font,zorder=9,
                      bbox=dict(facecolor='white',edgecolor='none',pad=3) if bg else None)
            if width:fitted.append((t,width))
            return t

        def roundbox(x,y,w,h,edge,fill='white',lw=1.2,r=7):
            p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                             linewidth=lw*.72,edgecolor=edge,facecolor=fill,zorder=1)
            ax.add_patch(p)
            return p

        def panel(x,y,w,h,color,title,titlecolor):
            # Subtle colored perimeter, then a nearly white inner working area.
            fill=np.array(mc.to_rgb(color))*.035+np.ones(3)*.965
            roundbox(x,y,w,h,color,fill,1.8,8)
            roundbox(x+12,y+41,w-24,h-50,mc.to_rgba(color,.55),'white',.75,8)
            label(x+w/2,y+22,title,28,titlecolor,'bold',width=w-26)

        def inset(x,y,w,h,color):
            tint=np.array(mc.to_rgb(color))*.045+np.ones(3)*.955
            return roundbox(x,y,w,h,mc.to_rgba(color,.48),tint,.7,6)

        def matrix(x,y,nr,nc,color,name,dim,cell=20):
            w,h=nc*cell,nr*cell
            base=np.array(mc.to_rgb(color))
            light=base*.63+np.ones(3)*.37
            palette=[base,light,base*.82+np.ones(3)*.18,base*.48+np.ones(3)*.52]
            # One raster per matrix, with independent diagonal shading in each cell.
            side=24
            yy,xx=np.mgrid[0:side,0:side]/(side-1)
            shading=(.13*(1-xx)+.08*(1-yy))[...,None]
            bitmap=np.zeros((nr*side,nc*side,3))
            for i in range(nr):
                for j in range(nc):
                    c=palette[(i*2+j*3)%4]
                    bitmap[i*side:(i+1)*side,j*side:(j+1)*side]=c*(1-shading)+shading
            ax.imshow(bitmap,extent=(x,x+w,y+h,y),interpolation='bilinear',zorder=3,aspect='auto')
            border=base*.52
            for j in range(nc+1):ax.plot([x+j*cell]*2,[y,y+h],color=border,lw=.8,zorder=4)
            for i in range(nr+1):ax.plot([x,x+w],[y+i*cell]*2,color=border,lw=.8,zorder=4)
            ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=border,lw=1,zorder=4))
            if name:label(x+w/2,y-21,name,27,width=w+80)
            if dim:label(x+w/2,y+h+21,dim,24,width=w+90)

        def arrow(points,color=BLUE,lw=3.3,scale=17):
            if len(points)>2:
                px,py=zip(*points[:-1]);line,=ax.plot(px,py,color=color,lw=lw*.72,zorder=6)
                line.set_path_effects([pe.Stroke(linewidth=(lw+3)*.72,foreground='white'),pe.Normal()])
            a=FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=scale,
                              linewidth=lw*.72,color=color,shrinkA=0,shrinkB=0,zorder=7,
                              joinstyle='miter',capstyle='butt')
            a.set_path_effects([pe.Stroke(linewidth=(lw+3)*.72,foreground='white'),pe.Normal()])
            ax.add_patch(a)

        # Upper row: same five-panel proportions as the local reference figure.
        panel(19,11,381,339,BLUE,'Current gradient & basis',DARKBLUE)
        panel(415,11,403,339,ORANGE,'Orthogonal gradient routing',DARKRED)
        panel(830,11,383,339,BLUE,'Low-dimensional Adam',DARKBLUE)
        panel(1225,11,348,339,GREEN,'Lift + residual update',DARKGREEN)
        panel(1585,11,331,339,PURPLE,'Updated parameters & state',DARKPURPLE)

        matrix(61,100,5,3,'#ac75dd',r'$W_t$',r'$m\times d$',19)
        matrix(183,100,5,3,'#f7a345',r'$G_t$',r'$m\times d$',19)
        matrix(306,100,5,3,'#229cff',r'$U_t$',r'$d\times r$',19)
        inset(46,246,328,86,BLUE)
        label(210,269,r'$G_t=\nabla_W\ell_t$',28)
        label(210,307,r'$U_t^{\mathsf{T}}U_t=I_r$',28)

        matrix(457,100,5,3,'#f7a345',r'$G_t$',r'$m\times d$',19)
        matrix(582,100,5,3,'#229cff',r'$U_t$',r'$d\times r$',19)
        matrix(714,100,5,3,'#68b57c',r'$Z_t$',r'$m\times r$',19)
        label(549,150,r'$\times$',30);label(679,150,r'$=$',30)
        inset(437,246,358,86,ORANGE)
        label(615,268,r'$Z_t=G_tU_t,\quad G_t^{\parallel}=Z_tU_t^{\mathsf{T}}$',25,width=338)
        label(615,307,r'$G_t^{\perp}=G_t-G_t^{\parallel}$',27,width=338)

        matrix(869,100,4,3,'#68b57c',r'$M_t$',r'$m\times r$',18)
        matrix(988,100,4,3,'#68b57c',r'$V_t$',r'$m\times r$',18)
        matrix(1110,100,4,3,'#229cff',r'$D_t^{\parallel}$',r'$m\times r$',18)
        inset(851,219,340,113,BLUE)
        label(1021,239,r'$M_t=\beta_1 M_{t-1}+(1-\beta_1)Z_t$',24,width=322)
        label(1021,275,r'$V_t=\beta_2 V_{t-1}+(1-\beta_2)Z_t^{\odot2}$',24,width=322)
        label(1021,311,r'$D_t^{\parallel}=\widehat M_t\oslash(\sqrt{\widehat V_t}+\epsilon_a)$',24,width=322)

        matrix(1259,104,3,4,'#229cff',r'$D_t^{\parallel}U_t^{\mathsf{T}}$',r'$m\times d$',19)
        matrix(1450,104,3,4,'#68b57c',r'$s_tG_t^{\perp}$',r'$m\times d$',19)
        label(1392,132,r'$+$',31)
        inset(1245,207,307,125,GREEN)
        label(1398,230,r'$s_t=\mathrm{clip}\!\left(\frac{\|G_t\|_F}{\|G_t^{\perp}\|_F+\epsilon_r}\right)$',25,width=285)
        label(1398,278,r'$W_{t+1}=W_t-\eta_{\parallel}D_t^{\parallel}U_t^{\mathsf{T}}$',24,width=283)
        label(1398,309,r'$-\eta_{\perp}s_tG_t^{\perp}$',24,width=283)

        matrix(1625,100,5,3,'#ac75dd',r'$W_{t+1}$',r'$m\times d$',19)
        matrix(1808,100,5,3,'#229cff',r'$U_{t+1}$',r'$d\times r$',19)
        inset(1608,244,285,88,PURPLE)
        label(1750,270,r'$U_{t+1}^{\mathsf{T}}U_{t+1}=I_r$',27,width=265)
        label(1750,308,'Save the aligned moments',21,DARKPURPLE,width=260)

        for start,end in [(400,420),(818,835),(1213,1230),(1573,1590)]:
            arrow([(start,158),(end,158)],BLUE,4,20)

        # Geometry branch: one green chain, visually aligned with the reference.
        panel(32,383,418,347,GREEN,'Grassmann ascent direction',DARKGREEN)
        panel(500,383,818,347,GREEN,'Transported momentum + QR',DARKGREEN)
        panel(1357,383,543,347,GREEN,'Procrustes state alignment',DARKGREEN)

        matrix(67,459,3,3,'#f7a345',r'$G_t^{\mathsf{T}}$',r'$d\times m$',18)
        matrix(204,459,3,3,'#68b57c',r'$Z_t$',r'$m\times r$',18)
        matrix(342,459,3,3,'#68b57c',r'$A_t$',r'$d\times r$',18)
        label(163,486,r'$\times$',29);label(302,486,r'$=$',29)
        inset(55,552,372,76,GREEN)
        label(241,589,r'$Q_t=\frac{(I-U_tU_t^{\mathsf{T}})G_t^{\mathsf{T}}Z_t}{\|G_t\|_F^2+\epsilon_c}$',31,width=347)
        inset(55,641,372,46,GREEN)
        label(241,664,r'$U_t^{\mathsf{T}}Q_t=0$',26)
        label(241,711,r'Implicit product: no $d\times d$ covariance',21,DARKGREEN,width=380)

        # QR visual occupies the left half; the exact recursion sits on the right.
        matrix(545,474,5,3,'#229cff',r'$U_t$',r'$d\times r$',20)
        matrix(694,474,5,3,'#68b57c',r'$\rho H_t$',r'$d\times r$',20)
        matrix(843,474,5,3,'#229cff',r'$U_{t+1}$',r'$d\times r$',20)
        label(648,523,r'$+$',32)
        arrow([(779,523),(823,523)],'black',1.8,12)
        label(801,499,'QR',18)
        label(724,646,'Continuous online subspace update',23,DARKGREEN,width=365)
        label(724,680,'Orthonormal routing frame',21,DARKBLUE,width=365)
        inset(932,447,363,245,GREEN)
        label(1113,475,r'$\Pi_U(A)=(I-UU^{\mathsf{T}})A$',27,width=340)
        label(1113,539,r'$H_t=\Pi_{U_t}\!\left[\mu H_{t-1}+(1-\mu)Q_t\right]$',26,width=337)
        label(1113,608,r'$U_{t+1}=\mathrm{qf}(U_t+\rho H_t)$',27,width=339)
        label(1113,660,r'$U_{t+1}^{\mathsf{T}}U_{t+1}=I_r$',25,width=337)

        inset(1380,440,497,58,GREEN)
        label(1628,469,r'$R_t=\mathrm{Polar}(U_{t+1}^{\mathsf{T}}U_t)$',29,width=473)
        matrix(1402,539,3,3,'#68b57c',r'$R_t$',r'$r\times r$',19)
        matrix(1576,539,3,3,'#68b57c',r'$M_t$',r'$m\times r$',19)
        matrix(1747,539,3,3,'#68b57c',r'$V_t$',r'$m\times r$',19)
        inset(1380,640,497,55,GREEN)
        label(1628,667,r'$M_t\leftarrow M_tR_t^{\mathsf{T}},\qquad V_t\leftarrow V_t(R_t^{\mathsf{T}})^{\odot2}$',27,width=471)
        label(1628,713,'Diagonal covariance approximation',22,DARKGREEN,width=470)

        # The lower path tracks the basis from the same G and Z used above.
        arrow([(619,351),(619,367),(241,367),(241,383)],GREEN,4,19)
        arrow([(450,558),(500,558)],GREEN,4,20)
        arrow([(1318,558),(1357,558)],GREEN,4,20)
        arrow([(1751,383),(1751,339)],GREEN,4,20)
        # Exterior state loop, like the local reference (keeps panel interiors clear).
        arrow([(1917,350),(1926,350),(1926,748),(16,748),(16,260),(47,260)],PURPLE,3.8,19)
        label(968,748,r'Next iteration: $(W_t,U_t)\leftarrow(W_{t+1},U_{t+1})$',25,PURPLE,bg=True,width=660)

        legend=[(315,BLUE,'Routing subspace / adaptive update'),
                (704,'#68b57c','Low-dimensional moments'),
                (1014,'#f7a345','Gradient'),
                (1203,'#ac75dd','Parameters')]
        for x,c,s in legend:
            roundbox(x,773,27,27,c,mc.to_rgba(c,.65),1.5,3)
            label(x+40,787,s,20,DARKBLUE,align='left')
        arrow([(1410,787),(1457,787)],BLUE,2,13)
        label(1470,787,'Computation / current flow',20,DARKBLUE,align='left')

        # Font metrics enforce the boxes without raster cropping or post-editing.
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        scale=ax.transData.transform((1,0))[0]-ax.transData.transform((0,0))[0]
        for artist,width in fitted:
            actual=artist.get_window_extent(renderer).width
            if actual>width*scale:
                artist.set_fontsize(artist.get_fontsize()*width*scale/actual)
        output=Path(__file__).with_name('grassroute_intuition.png')
        fig.savefig(output,dpi=200,facecolor='white')
        plt.close(fig)
        print(output)


if __name__=='__main__':
    draw()
