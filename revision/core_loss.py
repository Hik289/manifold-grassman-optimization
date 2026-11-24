"""Core scientific-learning loss figure, sourced only from full stored runs."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifact_io import save_figure
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIG=ROOT/'paper/figures'
COLORS={'GrassRoute':'#cf3b36','AdamW':'#293b52','PeriodicSVD':'#2576ad',
        'PeriodicDiag':'#2576ad','Fixed':'#949494'}


def generate():
    cases=[('Burgers PINN (archive)','archive', ['AdamW','PeriodicSVD','GrassRoute']),
           ('Poisson PINN (new)','sciml_poisson',['AdamW','Fixed','PeriodicDiag','GrassRoute']),
           ('Forced Burgers PINN (new)','sciml_burgers',['AdamW','Fixed','PeriodicDiag','GrassRoute'])]
    rows=[];ledger=[]
    previous_rc=plt.rcParams.copy()
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':11.5,'axes.titlesize':12.5,
                         'axes.labelsize':12,'legend.fontsize':10,'pdf.fonttype':42,
                         'axes.linewidth':.65,'lines.solid_capstyle':'round'})
    fig,axes=plt.subplots(1,2,figsize=(7.8,3.15),layout='constrained')
    control,control_ax=plt.subplots(figsize=(4.8,3.6),layout='constrained')
    plot_axes=[axes[0],control_ax,axes[1]]
    for panel,(title,task,methods),ax in zip(['a','control','b'],cases,plot_axes):
        series={}
        for method in methods:
            if task=='archive':
                bid={'AdamW':'B2','PeriodicSVD':'B4','GrassRoute':'B5'}[method]
                path=ROOT/'hpc_archive/experiments/runs'/f'traj_sciml_{bid}_r8.json'
                records=json.loads(path.read_text())['records']
            else:
                path=HERE/'strength_results/pinn'/task/method/'formal/trajectory.json'
                records=json.loads(path.read_text())
                summary=json.loads(path.with_name('summary.json').read_text())
                assert summary['status']=='complete' and summary['formal'] is True
                assert summary['n_epochs']==summary['epochs']==2000
            assert len(records)==2000
            assert [r['epoch'] for r in records]==list(range(0 if task=='archive' else 1,2000 if task=='archive' else 2001))
            loss=np.array([r['train_loss'] for r in records],dtype=float)
            assert np.isfinite(loss).all() and (loss>0).all()
            x=np.arange(1,2001)
            smooth=np.convolve(loss,np.ones(25)/25,mode='valid')
            color=COLORS[method]
            ax.semilogy(x,loss,color=color,alpha=.15,lw=.5,rasterized=True)
            ax.semilogy(x[24:],smooth,color=color,lw=2 if method=='GrassRoute' else 1.3,
                        label='Periodic SVD' if method=='PeriodicSVD' else method,zorder=5 if method=='GrassRoute' else 3)
            series[method]=dict(final=float(loss[-1]),mean_last100=float(loss[-100:].mean()))
            ledger.append(dict(task=task,method=method,path=str(path.relative_to(ROOT)),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epochs=2000,
                               **series[method]))
        ax.set_title((f'({panel}) ' if panel!='control' else '')+title,loc='left',pad=9)
        ax.set_xlabel('Epoch');ax.set_ylabel('Training objective')
        ax.set_xlim(0,2000);ax.set_xticks([0,500,1000,1500,2000])
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which='major',alpha=.18,lw=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper right',frameon=False)
        ref='PeriodicSVD' if task=='archive' else 'PeriodicDiag'
        ratio=series['GrassRoute']['mean_last100']/series[ref]['mean_last100']
        # This fixed terminal-window statistic is descriptive; the entire run
        # remains plotted, including crossings and less favorable comparators.
        ax.text(.035,.035,f'Terminal ratio: {ratio:.2f}',
                transform=ax.transAxes,fontsize=10,
                bbox=dict(boxstyle='square,pad=.25',facecolor='white',edgecolor='#dddddd',alpha=.9))
        rows.append(dict(task=task,metrics=series))
    FIG.mkdir(exist_ok=True)
    save_figure(fig,FIG/'fig_core_loss.pdf',bbox_inches='tight')
    save_figure(fig,FIG/'fig_core_loss.png',dpi=240,bbox_inches='tight')
    save_figure(control,FIG/'fig_core_loss_poisson_control.pdf',bbox_inches='tight')
    save_figure(control,FIG/'fig_core_loss_poisson_control.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    plt.close(control)
    plt.rcParams.update(previous_rc)
    manifest=dict(scope='Descriptive core loss figure; archive and independently validation-selected follow-ups kept separate',
                  seed=42,epochs_per_run=2000,smoothing='25-epoch trailing arithmetic mean; all raw values plotted faintly',
                  terminal_ratio='Ratio of final-100-epoch mean losses: GrassRoute / periodic baseline',
                  placement='Archive and forced-Burgers panels in main text; Poisson control in appendix',
                  uncertainty='No seed confidence interval; one seed per protocol',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=ledger,comparisons=rows)
    (HERE/'core_loss_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    result=generate()
    print(json.dumps(result['comparisons'],indent=2))
