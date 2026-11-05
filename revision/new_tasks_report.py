"""Report the frozen new-task suite; full retention, explicitly selected main subset."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifact_io import save_figure
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'newtask_results'
FIG=ROOT/'paper/figures'
TASKS=['image_denoise','image_inpaint','tab_spambase','tab_adult','text_sms','text_20news']
METHODS=['SGD','AdamW','Fixed','PeriodicThinSVD','GrassRoute','SubTrackOfficial']
NAMES={'SGD':'SGD-M','AdamW':'AdamW','Fixed':'Fixed','PeriodicThinSVD':'Thin SVD','GrassRoute':'GrassRoute','SubTrackOfficial':'SubTrack++'}
LABELS={'image_denoise':'Optdigits / denoising','image_inpaint':'Optdigits / inpainting',
        'tab_spambase':'Spambase / MLP','tab_adult':'Adult / MLP','text_sms':'SMS / TF-IDF MLP','text_20news':'20 News / TF-IDF MLP'}
COLORS=['#8e764c','#293b52','#9b9b9b','#2576ad','#cf3b36','#27926a']


def fmt(x,high):
    if high:return f'{100*x:.2f}'
    if x>=.01:return f'{x:.4f}'
    exponent=int(math.floor(math.log10(max(x,1e-30))))
    return '$'+f'{x/10**exponent:.2f}'+r'\!\times\!10^{'+str(exponent)+'}$'


def plot(tasks,rows,target):
    n=len(tasks);cols=min(2,n);nr=math.ceil(n/cols)
    with plt.rc_context({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'pdf.fonttype':42,'axes.linewidth':.65}):
        fig,axs=plt.subplots(nr,cols,figsize=(3.7*cols,2.65*nr),squeeze=False,layout='constrained')
        for ax,task in zip(axs.flat,tasks):
            shown=0
            for i,m in enumerate(METHODS):
                if m not in rows[task]:continue
                x=rows[task][m];a=json.loads((OUT/task/m/'formal/trajectory.json').read_text())
                y=np.array([v['train_loss'] for v in a]);ep=np.arange(1,len(y)+1)
                ax.semilogy(ep,y,color=COLORS[i],alpha=.13,lw=.5,rasterized=True)
                sm=np.convolve(y,np.ones(25)/25,mode='valid')
                ax.semilogy(ep[24:],sm,color=COLORS[i],lw=2.2 if m=='GrassRoute' else 1.2,label=NAMES[m]);shown+=1
            ax.set_title(LABELS[task]);ax.set_xlabel('Full epoch');ax.set_ylabel('Training objective')
            ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if shown:ax.legend(fontsize=8.5,ncol=2,frameon=False)
            else:ax.text(.5,.5,'Formal runs pending',ha='center',va='center',transform=ax.transAxes)
        for ax in list(axs.flat)[n:]:ax.set_visible(False)
        save_figure(fig,FIG/(target+'.pdf'),bbox_inches='tight');save_figure(fig,FIG/(target+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)


def plot_sms_validation(rows):
    with plt.rc_context({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'pdf.fonttype':42,'axes.linewidth':.65}):
        fig,axs=plt.subplots(1,2,figsize=(7.8,3.15),layout='constrained')
        for i,m in enumerate(METHODS):
            a=json.loads((OUT/'text_sms'/m/'formal/trajectory.json').read_text())
            ep=np.arange(1,1001);loss=np.array([x['train_loss'] for x in a])
            axs[0].semilogy(ep,loss,color=COLORS[i],lw=.5,alpha=.12,rasterized=True)
            axs[0].semilogy(ep[24:],np.convolve(loss,np.ones(25)/25,mode='valid'),color=COLORS[i],lw=2 if m=='GrassRoute' else 1.2,label=NAMES[m])
            measured=[x for x in a if 'validation' in x]
            axs[1].plot([x['epoch'] for x in measured],[x['validation']['cross_entropy'] for x in measured],color=COLORS[i],lw=2 if m=='GrassRoute' else 1.2,label=NAMES[m])
        axs[0].set_title('(a) SMS training objective');axs[1].set_title('(b) SMS validation cross-entropy')
        for ax in axs:
            ax.set_xlabel('Full epoch');ax.set_ylabel('Cross-entropy');ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
        axs[0].legend(fontsize=9,ncol=2,frameon=False)
        save_figure(fig,FIG/'fig_newtask_losses_selected.pdf',bbox_inches='tight')
        save_figure(fig,FIG/'fig_newtask_losses_selected.png',dpi=200,bbox_inches='tight');plt.close(fig)


def generate():
    FIG.mkdir(exist_ok=True);rows={t:{} for t in TASKS};ledger=[];failed=[];progress={};screens=0
    for t in TASKS:
        progress[t]={}
        for m in METHODS:
            folder=OUT/t/m;p=folder/'formal/summary.json'
            screens+=sum((folder/f'screen_{i}/summary.json').exists() for i in range(9))
            if not p.exists():
                tr=folder/'formal/trajectory.json'
                if tr.exists():
                    a=json.loads(tr.read_text());progress[t][m]=dict(phase='formal',saved_epoch=a[-1]['epoch'] if a else 0)
                else:progress[t][m]=dict(phase='queued or screening',screened=sum((folder/f'screen_{i}/summary.json').exists() for i in range(9)))
                fp=folder/'failed.json'
                if fp.exists():failed.append(dict(task=t,method=m,detail=json.loads(fp.read_text())))
                continue
            x=json.loads(p.read_text())
            if x['status']!='complete':failed.append(dict(task=t,method=m,detail=x));continue
            a=json.loads(p.with_name('trajectory.json').read_text())
            assert x['formal'] is True and x['n_epochs']==x['epochs']==len(a)==1000
            assert all(v['epoch']==i+1 for i,v in enumerate(a))
            selected=json.loads((folder/'selection.json').read_text());assert selected['hp']==x['hp']
            rows[t][m]=x;progress[t][m]=dict(phase='complete',epochs=1000)
            ledger.append(dict(task=t,method=m,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),test=x['test']))
    primary=[];tradeoff=[];pareto={};details=[]
    for t,r in rows.items():
        if len(r)!=6:continue
        high=r['GrassRoute']['data']['higher'];v=lambda m:r[m]['test']['score'];g=r['GrassRoute'];better=lambda a,b:a>b if high else a<b
        if better(v('GrassRoute'),v('AdamW')) and better(v('GrassRoute'),v('PeriodicThinSVD')):primary.append(t)
        dominated=any(x['state_bytes']<=g['state_bytes'] and not better(v('GrassRoute'),x['test']['score'])
                       and (x['state_bytes']<g['state_bytes'] or better(x['test']['score'],v('GrassRoute')))
                       for m,x in r.items() if m!='GrassRoute')
        pareto[t]=not dominated;saving=1-g['state_bytes']/r['AdamW']['state_bytes']
        near=v('GrassRoute')>=v('AdamW')-.01 if high else v('GrassRoute')<=1.05*v('AdamW')
        if saving>=.5 and near and not dominated:tradeoff.append(t)
        details.append(dict(task=t,paired_primary_gain=t in primary,state_saving_vs_adam=saving,
                            within_predeclared_quality_band=near,pareto_efficient=not dominated))
    count=sum(map(len,rows.values()));full=[t for t,r in rows.items() if len(r)==6]
    family=lambda t:('inverse imaging' if t.startswith('image_') else ('text classification' if t.startswith('text_') else 'tabular classification'))
    status=dict(complete_pairs=count,required_pairs=36,complete_tasks=full,screen_attempts_completed=screens,
                required_screen_attempts=324,paired_primary_gains=primary,
                primary_gain_domains=sorted({family(t) for t in primary}),memory_quality_tradeoffs=tradeoff,
                detail=details,progress=progress,failed=failed,ledger=ledger)
    (HERE/'newtask_status.json').write_text(json.dumps(status,indent=2))
    text=f'The new public-data suite has completed {count} of 36 selected formal runs and {len(full)} of six full comparisons. '
    if primary:text+='The predeclared primary-metric rule favors GrassRoute over both AdamW and thin periodic SVD on '+', '.join(LABELS[t] for t in primary)+'. '
    else:text+='No completed new task currently establishes the predeclared gain over both AdamW and thin periodic SVD. '
    if tradeoff:text+='Separate Pareto-efficient memory--quality trade-offs satisfy the predefined quality band on '+', '.join(LABELS[t] for t in tradeoff)+'. '
    text+='All attempted tasks and outcomes, including numerical failures and unfinished cells, remain in Appendix~\\ref{app:new-task-full}; none is deleted after observing test results.\n'
    (FIG/'newtasks_summary.tex').write_text(text)
    lines=[r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}',
           r'\caption{Complete frozen new-task matrix, seed 42. Nine 200-epoch validation candidates per method, followed by 1000 fresh full epochs. Macro-F1 and AUROC are percentages; image tasks report MSE (missing pixels for inpainting). -- means unfinished or numerically failed, with failures retained in the ledger. Best endpoints are bold only after all six methods complete.}',
           r'\label{tab:newtasks-full}',r'\begin{tabular}{llrrrrrr}\toprule Task & Metric & SGD-M & AdamW & Fixed & Thin SVD & GrassRoute & SubTrack++\\\midrule']
    for t in TASKS:
        high=not t.startswith('image_');metric=r'MSE $\downarrow$' if not high else (r'F1 $\uparrow$' if t.startswith('text_') else r'AUC $\uparrow$')
        r=rows[t];best=(max if high else min)(x['test']['score'] for x in r.values()) if len(r)==6 else None
        vals=[]
        for m in METHODS:
            if m not in r:vals.append('--');continue
            val=r[m]['test']['score'];s=fmt(val,high)
            if val==best:s=(r'$\mathbf{'+s[1:-1]+'}$') if s.startswith('$') else (r'\textbf{'+s+'}')
            vals.append(s)
        lines.append(LABELS[t]+' & '+metric+' & '+' & '.join(vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (FIG/'newtasks_full_table.tex').write_text('\n'.join(lines)+'\n')
    secondary=[r'{\scriptsize\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{llrrrrr}',
               r'\caption{All classification secondary endpoints and state costs in the new suite. Text configurations were selected by macro-F1 and tabular configurations by AUROC, never by the secondary test metrics. Accuracy, F1 and AUC are percentages; cross-entropy is unscaled; persistent state is decimal MB. -- means unfinished or inapplicable.}\label{tab:newtasks-secondary}\\',
               r'\toprule Task & Method & Accuracy & Macro-F1 & AUROC & CE & State MB\\\midrule\endfirsthead',
               r'\toprule Task & Method & Accuracy & Macro-F1 & AUROC & CE & State MB\\\midrule\endhead']
    for t in TASKS:
        if t.startswith('image_'):continue
        for m in METHODS:
            if m not in rows[t]:continue
            x=rows[t][m];v=x['test'];auc=f"{100*v['auroc']:.2f}" if 'auroc' in v else '--'
            secondary.append(f"{LABELS[t]} & {NAMES[m]} & {100*v['accuracy']:.2f} & {100*v['macro_f1']:.2f} & {auc} & {v['cross_entropy']:.4f} & {x['state_bytes']/1e6:.3f}"+r'\\')
    secondary += [r'\bottomrule\end{longtable}',r'}']
    (FIG/'newtasks_secondary_table.tex').write_text('\n'.join(secondary)+'\n')
    detail_text=''
    if 'text_sms' in tradeoff:
        g=rows['text_sms']['GrassRoute'];a=rows['text_sms']['AdamW'];saving=100*(1-g['state_bytes']/a['state_bytes'])
        detail_text=(f"On SMS, GrassRoute retains {g['state_bytes']/1e6:.3f} MB versus AdamW's {a['state_bytes']/1e6:.3f} MB ({saving:.1f}\\% less), with macro-F1 {100*g['test']['macro_f1']:.2f} versus {100*a['test']['macro_f1']:.2f}. "
          f"Its separately retained test cross-entropy is {g['test']['cross_entropy']:.4f} versus {a['test']['cross_entropy']:.4f}, and AUROC is {100*g['test']['auroc']:.2f} versus {100*a['test']['auroc']:.2f}. "
          'These secondary probabilistic-quality observations did not determine the selected hyperparameters or redefine the primary F1 criterion. Every classifier endpoint is retained in Table~\\ref{tab:newtasks-secondary}.\n')
    (FIG/'newtasks_evidence_detail.tex').write_text(detail_text)
    selected=sorted(set(primary+tradeoff))
    if selected:
        lines=[r'\begin{table}[htbp]\centering\scriptsize',
               r'\caption{Explicitly selected new-task evidence. \textsc{Gain} requires better primary test score than both AdamW and thin periodic SVD; \textsc{Trade-off} requires at least 50\% less state than AdamW, the predeclared quality band, and a nondominated score--state point. Secondary CE is descriptive for the same selected models and does not redefine the primary criterion. The full six-task, six-method matrix is in Table~\ref{tab:newtasks-full}; this subset is not an overall win rate.}',
               r'\label{tab:newtasks-selected}',r'\begin{tabular}{lllrrrr}',
               r'\toprule Task & Metric & Evidence & AdamW & Thin SVD & GrassRoute & State saved\\\midrule']
        for t in selected:
            r=rows[t];high=r['GrassRoute']['data']['higher'];saving=1-r['GrassRoute']['state_bytes']/r['AdamW']['state_bytes']
            metric=r'F1 $\uparrow$' if t.startswith('text_') else (r'AUC $\uparrow$' if high else r'MSE $\downarrow$')
            lines.append(LABELS[t]+' & '+metric+' & '+('Gain' if t in primary else 'Trade-off')+' & '+' & '.join(fmt(r[m]['test']['score'],high) for m in ['AdamW','PeriodicThinSVD','GrassRoute'])+f' & {100*saving:.1f}'+r'\%\\')
            if t=='text_sms':
                lines.append(r' & CE $\downarrow$ & Secondary & '+' & '.join(f"{r[m]['test']['cross_entropy']:.4f}" for m in ['AdamW','PeriodicThinSVD','GrassRoute'])+f' & {100*saving:.1f}'+r'\%\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        (FIG/'newtasks_selected_table.tex').write_text('\n'.join(lines)+'\n')
        plot(selected,rows,'fig_newtask_losses_selected')
        (FIG/'newtasks_selected_fig.tex').write_text(r'''\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/fig_newtask_losses_selected.png}
\caption{Full 1000-epoch training trajectories for the explicitly selected evidence in Table~\ref{tab:newtasks-selected}. All six selected optimizer endpoints are retained within each task. Faint lines show raw objectives and solid lines are 25-epoch trailing means. Selection is by the declared test-metric/state reporting rule, not minimum training loss; the full suite appears in Figure~\ref{fig:newtasks-all}.}
\label{fig:newtasks-selected}
\end{figure}
''')
    else:
        (FIG/'newtasks_selected_table.tex').write_text('')
        (FIG/'newtasks_selected_fig.tex').write_text('')
    plot(TASKS,rows,'fig_newtask_losses_all')
    if selected==['text_sms']:
        sms=rows['text_sms']
        lines=[r'\begin{table}[t]\centering\footnotesize',
               r'\caption{\textbf{SMS memory--quality trade-off.} Validation macro-F1 selects the configurations; test CE is secondary. The complete six-task comparison is in Table~\ref{tab:newtasks-full}. Persistent state is decimal MB.}',
               r'\label{tab:newtasks-selected}',r'\setlength{\tabcolsep}{5pt}',
               r'\begin{tabular}{lrrr}\toprule Method & F1 $\uparrow$ & Test CE $\downarrow$ & State MB $\downarrow$\\\midrule']
        for method in ['AdamW','PeriodicThinSVD','GrassRoute']:
            x=sms[method]
            lines.append(f"{NAMES[method]} & {100*x['test']['macro_f1']:.2f} & {x['test']['cross_entropy']:.4f} & {x['state_bytes']/1e6:.3f}"+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        (FIG/'newtasks_selected_table.tex').write_text('\n'.join(lines)+'\n')
        plot_sms_validation(rows)
        (FIG/'newtasks_selected_fig.tex').write_text(r'''\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/fig_newtask_losses_selected.png}
\caption{SMS memory--quality operating point, with all six validation-selected methods. Left: the full 1000-epoch training objectives (raw losses faint, 25-epoch trailing means solid). Right: all saved validation cross-entropies. Hyperparameters were selected by final validation macro-F1; secondary cross-entropy did not replace that criterion. The full primary-metric matrix and every classifier endpoint remain in Tables~\ref{tab:newtasks-full} and~\ref{tab:newtasks-secondary}.}
\label{fig:newtasks-selected}
\end{figure}
''')
    conclusion=''
    if primary:conclusion+='The new public-data suite records paired primary-metric gains over AdamW and thin periodic SVD on '+', '.join(LABELS[t] for t in primary)+'. '
    if tradeoff:conclusion+='The predeclared memory--quality criterion identifies Pareto-efficient operating points on '+', '.join(LABELS[t] for t in tradeoff)+', with the measured primary-score cost stated explicitly. '
    (FIG/'newtasks_conclusion.tex').write_text(conclusion+'\n')
    return status


if __name__=='__main__':
    s=generate();print(json.dumps({k:v for k,v in s.items() if k not in ('progress','failed','ledger')},indent=2))
