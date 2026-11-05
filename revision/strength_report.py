"""Generate the complete follow-up evidence, including adverse conditions."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifact_io import save_figure
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parent/'paper/figures'
OUT = HERE/'strength_results'

COLORS = {'GrassRoute':'#cf3b36', 'NoMomentum':'#bf8e1b', 'Fixed':'#8a8a8a',
          'Periodic5':'#237ab4', 'Periodic25':'#4b9b75', 'Periodic100':'#8664aa',
          'AdamW':'#26384b', 'PeriodicDiag':'#237ab4'}


def scientific(x):
    if 0 < abs(x) < .001:
        exponent=int(math.floor(math.log10(abs(x))))
        return '$'+f'{x/10**exponent:.2f}'+r'\!\times\!10^{'+str(exponent)+'}$'
    return f'{x:.3g}'


def bold(value):
    if value.startswith('$') and value.endswith('$'):
        return r'$\mathbf{'+value[1:-1]+r'}$'
    return r'\textbf{'+value+'}'


def generate(completed):
    FIG.mkdir(exist_ok=True)
    from core_loss import generate as generate_core_loss
    generate_core_loss()
    metrics = []
    for task, methods in completed.items():
        if not all(k in methods for k in ['AdamW','GrassRoute','PeriodicSVD']): continue
        a,g,p = [methods[k] for k in ['AdamW','GrassRoute','PeriodicSVD']]
        high = task.startswith(('graph_','mlp_','cnn_','vit_','recsys_','continual_'))
        metrics.append(dict(task=task, higher=high, state_ratio=g['state_bytes']/a['state_bytes'],
                            adam=a['test']['score'], grass=g['test']['score'], periodic=p['test']['score'],
                            grass_better_adam=(g['test']['score']>a['test']['score'] if high else g['test']['score']<a['test']['score'])))
    # Include every completed paired task, rather than just the favorable rows.
    lines = [r'\begin{table}[htbp]\centering\scriptsize',
             r'\caption{All completed paired extension endpoints: state retained relative to AdamW, and final test metrics. Lower state is beneficial; metric direction is explicit. These are the original versioned periodic endpoints, not the transport-corrected follow-up. Wall time is not encoded as isolated speed.}',
             r'\label{tab:state-quality}',
             r'\begin{tabular}{lrrrr}\toprule Task & State / AdamW & AdamW & GrassRoute & Periodic\\\midrule']
    from report import label
    for x in metrics:
        fmt = (lambda z:f'{100*z:.2f}') if x['higher'] else scientific
        direction = r'$\uparrow$' if x['higher'] else r'$\downarrow$'
        lines.append(f"{label(x['task'])} {direction} & {100*x['state_ratio']:.1f}\\% & {fmt(x['adam'])} & {fmt(x['grass'])} & {fmt(x['periodic'])}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (FIG/'strength_state_quality.tex').write_text('\n'.join(lines)+'\n')
    fig, axs = plt.subplots(1,2,figsize=(9,3.1),constrained_layout=True)
    for ax, high in zip(axs,[True,False]):
        selected = [x for x in metrics if x['higher']==high]
        for x in selected:
            delta = 100*(x['grass']-x['adam']) if high else x['grass']/x['adam']
            ax.scatter(100*x['state_ratio'],delta,color=COLORS['GrassRoute'],s=35)
            ax.annotate(x['task'].replace('graph_','').replace('recsys_','').replace('sciml_',''),
                        (100*x['state_ratio'],delta),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.axhline(0 if high else 1,color='gray',ls='--',lw=1)
        ax.set_xlabel('Persistent optimizer state retained (% of AdamW)')
        ax.grid(alpha=.2)
    axs[0].set_ylabel('GrassRoute minus AdamW (percentage points)')
    axs[0].set_title('Accuracy / Recall@20: higher is better')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('GrassRoute error / AdamW error')
    axs[1].set_title('Regression / solution error: lower is better')
    save_figure(fig,FIG/'strength_state_quality.pdf')
    plt.close(fig)
    audit = dict(completed_pairs=len(metrics), paired_state_quality=metrics)

    track = OUT/'tracking_summary.json'
    if track.exists():
        data=json.loads(track.read_text()); conditions=data['conditions']; methods=data['protocol']['methods']
        fig,axs=plt.subplots(1,3,figsize=(10,3.15),constrained_layout=True)
        for ax,m in zip(axs[:2],data['protocol']['rows']):
            cc=[x for x in conditions if x['rows']==m]
            for name in methods:
                ax.plot([x['drift'] for x in cc],[x['summary'][name]['mean_sin2_last500'] for x in cc],
                        'o-',color=COLORS[name],lw=2 if name=='GrassRoute' else 1.2,ms=3,label=name)
            ax.set_title(f'{m} gradient rows / step');ax.set_xlabel('Population rotation (radians / step)')
            ax.set_ylabel('Pre-update mean squared sine error');ax.grid(alpha=.2)
            ax.set_yscale('log')
        axs[0].legend(fontsize=6,frameon=False,ncol=2)
        # This is measured CPU implementation cost, not an accelerator claim.
        cc=[x for x in conditions if x['rows']==128]
        cost=[1000*np.mean([x['summary'][name]['optimizer_seconds']/2000 for x in cc]) for name in methods]
        axs[2].barh(methods,cost,color=[COLORS[n] for n in methods])
        axs[2].set_xlabel('CPU optimizer call (ms)');axs[2].set_title('CPU call time (128 rows)')
        axs[2].grid(axis='x',alpha=.2)
        save_figure(fig,FIG/'strength_tracking.pdf');plt.close(fig)
        lines=[r'\begin{table}[htbp]\centering\scriptsize',
               r'\caption{Complete predeclared tracking grid, 2000 steps, seed 42. Mean $r^{-1}\|\sin\Theta\|_F^2$ before each update on steps 1501--2000; lower is better. All methods see identical streams and start from the same random frame. Periodic variants use corrected diagonal transport. Bold marks the minimum among all displayed methods.}',
               r'\label{tab:tracking-strength}',
               r'\begin{tabular}{rrrrrrrr}\toprule Rows & Drift & GrassRoute & No mom. & Fixed & Periodic5 & Periodic25 & Periodic100\\\midrule']
        wins=0
        for x in conditions:
            vals=[x['summary'][n]['mean_sin2_last500'] for n in methods];best=min(vals)
            if vals[0] < vals[4]: wins+=1
            v=[(r'\textbf{'+f'{z:.4f}'+r'}') if z==best else f'{z:.4f}' for z in vals]
            lines.append(f"{x['rows']} & {x['drift']:.3f} & "+' & '.join(v)+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        (FIG/'strength_tracking_table.tex').write_text('\n'.join(lines)+'\n')
        text=f'With the fixed tracking rate $\\rho=3$, GrassRoute has lower steady tracking error than interval-25 instantaneous SVD in {wins} of the {len(conditions)} predeclared conditions. '
        text+='Table~\\ref{tab:tracking-strength} includes the static and fast-drift controls, the no-momentum ablation, and interval-5 and interval-100 alternatives. '
        text+='These comparisons measure subspace identification on synthetic streams, not an end-to-end accuracy advantage.\n'
        (FIG/'strength_tracking_summary.tex').write_text(text)
        audit['tracking']=dict(conditions=len(conditions), grass_better_periodic25=wins,
                               protocol_sha256=data['protocol']['source_sha256'])
    else:
        (FIG/'strength_tracking_summary.tex').write_text('The preregistered tracking comparison is still running; no endpoint is claimed.\n')
    extended=OUT/'tracking_extended_summary.json'
    if extended.exists():
        data=json.loads(extended.read_text());cc=data['conditions'];methods=data['protocol']['methods']
        names={'GrassRoute':'GrassRoute','Periodic5':'Instant SVD / 5','Periodic25':'Instant SVD / 25',
               'EMA5_90':'EMA .9 / 5','EMA25_90':'EMA .9 / 25','EMA5_99':'EMA .99 / 5','EMA25_99':'EMA .99 / 25'}
        palette=['#cf3b36','#619ac6','#a2c9dc','#14653c','#6ca37c','#614692','#af9ac6']
        fig,axs=plt.subplots(1,2,figsize=(9,3.2),constrained_layout=True)
        for ax,m in zip(axs,[16,128]):
            rr=[x for x in cc if x['rows']==m]
            for n,color in zip(methods,palette):
                ax.plot([x['drift'] for x in rr],[x['summary'][n]['mean_sin2_last500'] for x in rr],
                        'o-',label=names[n],color=color,lw=2.2 if n=='GrassRoute' else 1.2,ms=3)
            ax.set_xscale('symlog',linthresh=.001);ax.set_yscale('log')
            ax.set_xlabel('Population rotation (radians / step)');ax.set_ylabel('Pre-update mean squared sine error')
            ax.set_title(f'{m} gradient rows / step');ax.grid(alpha=.2)
        axs[0].legend(ncol=2,fontsize=6,frameon=False)
        save_figure(fig,FIG/'strength_tracking_ema.pdf');plt.close(fig)
        lines=[r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}',
               r'\caption{Stronger tracking controls on all ten predeclared conditions. EMA SVD stores the full normalized covariance and uses decay .9 or .99; its refresh period is 5 or 25. Mean pre-update $r^{-1}\|\sin\Theta\|_F^2$ on steps 1501--2000. Bold identifies the best displayed error; no temporal observations are treated as independent replicates.}',
               r'\label{tab:tracking-ema}',
               r'\begin{tabular}{rrrrrrrrr}\toprule Rows & Drift & GrassRoute & SVD/5 & SVD/25 & EMA.9/5 & EMA.9/25 & EMA.99/5 & EMA.99/25\\\midrule']
        wins=0;inst_wins=0
        for x in cc:
            vals=[x['summary'][n]['mean_sin2_last500'] for n in methods];best=min(vals)
            wins+=int(vals[0]==best);inst_wins+=int(vals[0]<min(vals[1:3]))
            vals=[r'\textbf{'+f'{v:.5f}'+'}' if v==best else f'{v:.5f}' for v in vals]
            lines.append(f"{x['rows']} & {x['drift']:.3f} & "+' & '.join(vals)+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        (FIG/'strength_tracking_ema_table.tex').write_text('\n'.join(lines)+'\n')
        text=f'After adding four full-covariance EMA controls and the nonadiabatic rate $0.1$, GrassRoute has the lowest error in {wins} of {len(cc)} conditions and beats both instantaneous refresh baselines in {inst_wins} of {len(cc)}. '
        text+='EMA SVD is stronger in the static or slow-drift conditions. At 128 rows and drift $0.02$, GrassRoute has mean error '
        ref=next((x for x in cc if x['rows']==128 and x['drift']==.02),None)
        if ref:
            g=ref['summary']['GrassRoute'];baseline=ref['summary']['EMA5_90']
            text+=f"{g['mean_sin2_last500']:.5f} versus {baseline['mean_sin2_last500']:.5f} for the best displayed EMA alternative, with {g['state_bytes']:,} versus {baseline['state_bytes']:,} persistent state bytes. "
        text+='This isolates a noise--lag--memory trade-off; the current implementation does not establish a CPU speed advantage.\n'
        (FIG/'strength_tracking_ema_summary.tex').write_text(text)
        audit['tracking_extended']=dict(conditions=len(cc),grass_best=wins,grass_better_instant=inst_wins,
                                         protocol_sha256=data['protocol']['source_sha256'])
    pinn=OUT/'pinn_summary.json'
    if pinn.exists():
        data=json.loads(pinn.read_text()); rr=data['results']; methods=data['protocol']['methods']
        # Verify complete trajectory counts directly, not just an aggregate flag.
        for x in rr:
            if x['status']!='complete':continue
            tr=json.loads((OUT/'pinn'/x['task']/x['method']/'formal/trajectory.json').read_text())
            assert x['epochs']==x['n_epochs']==len(tr)==2000
            assert all(r['epoch']==i+1 for i,r in enumerate(tr))
        lines=[r'\begin{table}[htbp]\centering\small',
               r'\caption{Independent expanded-search PINN follow-up with a transport-corrected periodic comparator. Nine validation-screen candidates per method (100 epochs each), then a fresh 2000-epoch run of the selected configuration. Final held-out solution relative $L_2$ error; lower is better. -- denotes unfinished, not omitted, training.}',
               r'\label{tab:pinn-strength}',
               r'\begin{tabular}{lrrrr}\toprule Task & AdamW & Fixed & PeriodicDiag & GrassRoute\\\midrule']
        config=[r'\begin{table}[htbp]\centering\scriptsize',r'\caption{Follow-up selections and resources. Shared local CPU wall times include evaluation; persistent state is in decimal MB. All selected endpoints are reported, irrespective of their ranking.}',
                r'\begin{tabular}{llrrrl}\toprule Task & Method & LR & State MB & Seconds & Other selection\\\midrule']
        wins=[]
        for task in data['protocol']['tasks']:
            cells={x['method']:x for x in rr if x['task']==task and x['status']=='complete'}
            best=min([x['test']['score'] for x in cells.values()],default=None) if len(cells)==4 else None
            vals=[]
            for m in methods:
                if m not in cells: vals.append('--');continue
                x=cells[m];s=scientific(x['test']['score'])
                vals.append(bold(s) if x['test']['score']==best else s)
                hp=x['hp'];labels={'residual_ratio':'residual ratio','rho':r'$\rho$','period':'period'}
                other=', '.join(f'{labels.get(k,k)}={v:g}' for k,v in hp.items() if k!='lr') or '--'
                config.append(f"{label(task)} & {m} & {hp['lr']:.4g} & {x['state_bytes']/1e6:.3f} & {x['wall_s']:.1f} & {other}"+r'\\')
            if len(cells)==4 and all(cells['GrassRoute']['test']['score']<cells[m]['test']['score'] for m in methods if m!='GrassRoute'):wins.append(task)
            lines.append(label(task)+' & '+' & '.join(vals)+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}'];config += [r'\bottomrule\end{tabular}\end{table}']
        (FIG/'strength_pinn_table.tex').write_text('\n'.join(lines)+'\n')
        (FIG/'strength_pinn_config.tex').write_text('\n'.join(config)+'\n')
        text=f'The independent follow-up has completed {sum(x["status"]=="complete" for x in rr)} of 8 selected formal runs. '
        text+=('GrassRoute has the lowest final test error on '+', '.join(label(t) for t in wins)+'. ') if wins else 'The currently completed follow-up endpoints do not establish an all-baseline GrassRoute win. '
        text+='Its endpoints are not pooled with the original six-method extension.\n'
        (FIG/'strength_pinn_summary.tex').write_text(text)
        audit['pinn']=dict(complete=sum(x['status']=='complete' for x in rr),required=8,grass_best=wins)
    else:
        (FIG/'strength_pinn_summary.tex').write_text('The independent PINN follow-up is running. Its selected final test endpoints will appear after completion.\n')
    graph=OUT/'graph_summary.json'
    if graph.exists():
        data=json.loads(graph.read_text());rr=data['results'];methods=data['protocol']['methods']
        names={'SGD':'SGD-M','AdamW':'AdamW','Fixed':'Fixed','PeriodicDiag':'PeriodicDiag',
               'GrassRoute':'GrassRoute','SubTrackOfficial':'SubTrack++'}
        verified=[]
        for x in rr:
            if x['status']!='complete':continue
            folder=OUT/'graph'/x['task']/x['method']
            tr=json.loads((folder/'formal/trajectory.json').read_text())
            selection=json.loads((folder/'selection.json').read_text())
            assert x['epochs']==x['n_epochs']==len(tr)==1000
            assert all(r['epoch']==i+1 for i,r in enumerate(tr))
            assert x['hp']==selection['hp']
            verified.append(x)
        lines=[r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}',
               r'\caption{Independent Planetoid follow-up: nine 100-epoch validation candidates per method, then 1000 fresh formal epochs, seed 42. All accuracies are final public-test percentages; -- indicates unfinished training. The periodic method now transports both moments. SubTrack++ is the same pinned official implementation. These are transductive graph experiments.}',
               r'\label{tab:graph-strength}',
               r'\begin{tabular}{lrrrrrr}\toprule Task & SGD-M & AdamW & Fixed & PeriodicDiag & GrassRoute & SubTrack++\\\midrule']
        resource=[r'\begin{table}[htbp]\centering\scriptsize',r'\caption{Independent graph follow-up selections and persistent optimizer state. The local comparator correction and enlarged selection budgets define a separate comparison from the original extension.}',
                  r'\begin{tabular}{llrrl}\toprule Task & Method & LR & State MB & Other selection\\\midrule']
        wins=[];paired=[]
        for task in data['protocol']['tasks']:
            cells={x['method']:x for x in verified if x['task']==task}
            best=max((x['test']['score'] for x in cells.values()),default=None) if len(cells)==6 else None
            vals=[]
            for m in methods:
                if m not in cells:vals.append('--');continue
                x=cells[m];v=f"{100*x['test']['score']:.2f}"
                vals.append(r'\textbf{'+v+'}' if x['test']['score']==best else v)
                labels={'residual_ratio':'residual ratio','rho':r'$\rho$','period':'period'}
                other=', '.join(f'{labels.get(k,k)}={v:g}' for k,v in x['hp'].items() if k!='lr') or '--'
                resource.append(f"{label(task)} & {names[m]} & {x['hp']['lr']:.4g} & {x['state_bytes']/1e6:.3f} & {other}"+r'\\')
            if all(k in cells for k in ['GrassRoute','AdamW','PeriodicDiag']):
                if cells['GrassRoute']['test']['score']>max(cells[k]['test']['score'] for k in ['AdamW','PeriodicDiag']):paired.append(task)
            if len(cells)==6 and cells['GrassRoute']['test']['score']==best:wins.append(task)
            lines.append(label(task)+' & '+' & '.join(vals)+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}'];resource += [r'\bottomrule\end{tabular}\end{table}']
        (FIG/'strength_graph_table.tex').write_text('\n'.join(lines)+'\n')
        (FIG/'strength_graph_config.tex').write_text('\n'.join(resource)+'\n')
        text=f'The independent Planetoid follow-up has completed {len(verified)} of 18 selected formal runs. '
        text+=('Completed paired scores favor GrassRoute over both AdamW and corrected periodic routing on '+', '.join(label(t) for t in paired)+'. ') if paired else 'Completed follow-up pairs do not currently establish a GrassRoute gain over both AdamW and corrected periodic routing. '
        text+=('It has the best displayed six-method endpoint on '+', '.join(label(t) for t in wins)+'. ') if wins else ''
        text+='Pending cells and full selection/state records are retained.\n'
        (FIG/'strength_graph_summary.tex').write_text(text)
        audit['graph']=dict(complete=len(verified),required=18,grass_best=wins,grass_better_adam_periodic=paired)
    else:
        (FIG/'strength_graph_summary.tex').write_text('The independent Planetoid follow-up is running on CPU alongside the original GPU suite; its selected formal endpoints are still pending.\n')
    dump=HERE/'strength_evidence.json'
    dump.write_text(json.dumps(audit,indent=2))
    return audit


if __name__=='__main__':
    from report import load
    generate(load()[0])
