"""Core scientific-learning loss figure, sourced only from full stored runs."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import (RCPARAMS, export_figure, finish_axis, legend_handles,
                        plot_training)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIG=ROOT/'paper/figures'


def generate(*, write_manifest=True):
    cases=[('Burgers', 'Archived runs', 'archive',
            ['AdamW','PeriodicSVD','GrassRoute']),
           ('Poisson', 'Validation-selected runs', 'sciml_poisson',
            ['AdamW','Fixed','PeriodicDiag','GrassRoute']),
           ('Forced Burgers', 'Validation-selected runs', 'sciml_burgers',
            ['AdamW','Fixed','PeriodicDiag','GrassRoute'])]
    rows=[];ledger=[]
    with plt.rc_context(RCPARAMS):
        fig,axes=plt.subplots(1,2,figsize=(7.3,3.6))
        fig.subplots_adjust(left=.091,right=.985,bottom=.22,top=.72,wspace=.34)
        control,control_ax=plt.subplots(figsize=(4.6,3.7))
        control.subplots_adjust(left=.155,right=.97,bottom=.21,top=.74)
        plot_axes=[axes[0],control_ax,axes[1]]
        for panel,(title,subtitle,task,methods),ax in zip(['a',None,'b'],cases,plot_axes):
            series={};raw=[]
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
                plot_training(ax,np.arange(1,2001),loss,method)
                raw.append(loss)
                series[method]=dict(final=float(loss[-1]),mean_last100=float(loss[-100:].mean()))
                ledger.append(dict(task=task,method=method,path=str(path.relative_to(ROOT)),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epochs=2000,
                                   **series[method]))
            finish_axis(ax,raw,2000)
            owner=ax.figure;box=ax.get_position()
            owner.text(box.x0,.952,(f'({panel})  ' if panel else '')+title,
                       fontsize=12,fontweight='bold',va='top')
            owner.text(box.x0,.887,subtitle,fontsize=9,color='#68717B',va='top')
            order=['GrassRoute','AdamW']+[m for m in methods if m not in ('GrassRoute','AdamW')]
            ax.legend(handles=legend_handles(order),loc='lower left',bbox_to_anchor=(0,1.025),
                      ncol=2,borderaxespad=0,fontsize=8.7)
            ref='PeriodicSVD' if task=='archive' else 'PeriodicDiag'
            ratio=series['GrassRoute']['mean_last100']/series[ref]['mean_last100']
            # The mean uses the same fixed final 100 epochs as the original plot.
            # All raw observations remain in view; the annotation is descriptive.
            owner.text(box.x0,.04,f'Final 100 epochs: loss ratio {ratio:.2f}',
                       fontsize=8.8,color='#4D535B',va='bottom')
            rows.append(dict(task=task,metrics=series))
        export_figure(fig,FIG/'fig_core_loss')
        export_figure(control,FIG/'fig_core_loss_poisson_control')
    manifest=dict(scope='Descriptive core loss figure; archive and independently validation-selected follow-ups kept separate',
                  seed=42,epochs_per_run=2000,smoothing='25-epoch trailing arithmetic mean; all raw values plotted faintly',
                  terminal_ratio='Ratio of final-100-epoch mean losses: GrassRoute / periodic baseline',
                  placement='Archive and forced-Burgers panels in main text; Poisson control in appendix',
                  uncertainty='No seed confidence interval; one seed per protocol',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=ledger,comparisons=rows)
    if write_manifest:
        (HERE/'core_loss_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    result=generate()
    print(json.dumps(result['comparisons'],indent=2))
