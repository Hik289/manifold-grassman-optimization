"""Regenerate paper tables and scientific figures from verified completed runs."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifact_io import save_figure

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from tasks_extended import TASKS

METHODS=['SGD','AdamW','Fixed','PeriodicSVD','GrassRoute','SubTrackOfficial']
NAMES={'SGD':'SGD-M','AdamW':'AdamW','Fixed':'Fixed','PeriodicSVD':'Periodic SVD','GrassRoute':'GrassRoute','SubTrackOfficial':'SubTrack++'}
FIG=ROOT/'paper/figures';FIG.mkdir(parents=True,exist_ok=True)


def label(task):
    return {'sciml_poisson':'Poisson / PINN','sciml_burgers':'Forced Burgers / PINN',
            'sciml_darcy':'Darcy / DeepONet','sciml_fno':'Burgers operator / FNO',
            'mlp_mnist':'MNIST / MLP','graph_cora':'Cora / SAGE','graph_citeseer':'CiteSeer / SAGE',
            'graph_pubmed':'PubMed / SAGE','graph_arxiv':'ogbn-arxiv / SAGE',
            'recsys_movielens':'MovieLens-1M / MF','recsys_amazonbooks':'Amazon Books subset / MF',
            'recsys_yelp':'Yelp subset / MF','continual_rotmnist':'Rotated MNIST / MLP',
            'continual_splitcifar100':'Split CIFAR-100 / ResNet-18'}.get(task,task.replace('_',' / ').replace('matreg','Matrix reg.').replace('lowrank','Low-rank reg.').replace('cifar100','CIFAR-100').replace('cifar10','CIFAR-10').replace('tinyimagenet','Tiny-ImageNet').replace('imagenet100','ImageNet-100').replace('cnn','ResNet-18').replace('vit','ViT-Tiny'))


def load():
    rows={};ledger=[]
    for task in TASKS:
        rows[task]={}
        for method in METHODS:
            path=HERE/'results'/task/method/'formal/summary.json'
            if not path.exists():continue
            try:
                x=json.loads(path.read_text());tr=json.loads(path.with_name('trajectory.json').read_text())
                assert x['status']=='complete' and x['formal'] is True
                assert x['n_epochs']==x['epochs']==len(tr) and tr[-1]['epoch']==x['epochs']
                assert x['task']==task and x['method']==method and x['test'] is not None
                assert all(r['epoch']==i+1 for i,r in enumerate(tr))
                assert math.isfinite(x['test']['score'])
                if task.startswith('lowrank_'):assert x['meta'].get('target_rank')==8
                if task=='recsys_movielens':assert x['meta'].get('subset','').startswith('all input users')
                rows[task][method]=x
                ledger.append({'task':task,'method':method,'summary':str(path.relative_to(ROOT)),
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                               'epochs':x['epochs'],'test_score':x['test']['score']})
            except Exception as e:
                ledger.append({'task':task,'method':method,'invalid_result':str(e)})
    return rows,ledger


def number(x,high):
    if high:return f'{100*x:.2f}'
    if x>=.01:return f'{x:.4f}'
    s=f'{x:.2e}';a,b=s.split('e');return '$'+a+r'\!\times\!10^{'+str(int(b))+'}$'


def write_completed_primary_gains(rows, full, paired):
    """Write only verified, six-method, strict two-baseline gains to the main text."""
    selected=[t for t in TASKS if t in full and t in paired]
    path=FIG/'revision_main_positive.tex'
    if not selected:
        path.write_text('')
        return selected
    s=[r'\paragraph{Additional completed primary-metric gains.}',
       r'We use the final held-out metric after validation-based learning-rate selection,',
       r'and list every fully completed proposal configuration for which GrassRoute',
       r'strictly improves on both AdamW and periodic SVD. All six methods finished',
       r'the formal budget before a configuration entered this comparison.',
       r'Each comparison has one seed and does not establish statistical significance.',
       r'The complete six-method matrix, including unfavorable outcomes, appears in',
       r'Appendix~\ref{app:full-results}.',
       r'\begin{table*}[t]',r'\centering\footnotesize',
       r'\setlength{\tabcolsep}{5pt}',
       r'\caption{\textbf{Completed primary-metric gains.} All proposal configurations',
       r'with a strict GrassRoute gain over both named baselines. Accuracy and Recall@20 are',
       r'percentages; the other metrics are errors. State reduction is relative to',
       r'AdamW persistent optimizer state. Seed 42.}',
       r'\label{tab:completed-primary-gains}',
       r'\begin{tabular}{llrrrr}',r'\toprule',
       r'Task / model & Metric & AdamW & Periodic SVD & GrassRoute & State reduction \\',
       r'\midrule']
    for t in selected:
        rr=rows[t]
        high=t.startswith(('graph_','mlp_','cnn_','vit_','recsys_','continual_'))
        metric=(r'R@20 $\uparrow$' if t.startswith('recsys_') else
                r'Acc. $\uparrow$' if high else
                r'MSE $\downarrow$' if t.startswith(('matreg_','lowrank_')) else
                r'Rel.$L_2\downarrow$')
        values=[number(rr[m]['test']['score'],high)
                for m in ('AdamW','PeriodicSVD','GrassRoute')]
        reduction=100*(1-rr['GrassRoute']['state_bytes']/rr['AdamW']['state_bytes'])
        s.append(label(t)+' & '+metric+' & '+' & '.join(values)+
                 f' & {reduction:+.1f}\\%'+r' \\')
    s.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}'])
    path.write_text('\n'.join(s)+'\n')
    return selected


def table(rows,keys,name,caption,tablabel):
    s=[r'\begin{table}[htbp]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',
       r'\caption{'+caption+'}',r'\label{'+tablabel+'}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
       r'Task / model & Metric & SGD-M & AdamW & Fixed & Periodic & GrassRoute & SubTrack++ \\',r'\midrule']
    for task in keys:
        high=task.startswith(('graph_','mlp_','cnn_','vit_','recsys_','continual_'))
        metric=r'R@20 $\uparrow$' if task.startswith('recsys_') else (r'Acc. $\uparrow$' if high else (r'MSE $\downarrow$' if task.startswith(('matreg_','lowrank_')) else r'Rel.$L_2\downarrow$'))
        rr=rows[task];best=None
        if len(rr)==6:best=(max if high else min)(r['test']['score'] for r in rr.values())
        vals=[]
        for m in METHODS:
            if m not in rr:vals.append('--');continue
            v=rr[m]['test']['score'];n=number(v,high)
            if best is not None and abs(v-best)<1e-12:
                n=r'{\boldmath '+n+'}' if n.startswith('$') else r'\textbf{'+n+'}'
            vals.append(n)
        s.append(label(task)+' & '+metric+' & '+' & '.join(vals)+r' \\')
    s.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (FIG/name).write_text('\n'.join(s)+'\n')


def generate():
    rows,ledger=load();total=sum(len(r) for r in rows.values());full=[t for t,r in rows.items() if len(r)==6]
    paired=[];best=[];adam_positive=[]
    for t,r in rows.items():
        if all(m in r for m in ('GrassRoute','AdamW')):
            high=t.startswith(('graph_','mlp_','cnn_','vit_','recsys_','continual_'))
            g,a=[r[m]['test']['score'] for m in ('GrassRoute','AdamW')]
            if (g>a if high else g<a):adam_positive.append(t)
        if not all(m in r for m in ('GrassRoute','AdamW','PeriodicSVD')):continue
        high=t.startswith(('graph_','mlp_','cnn_','vit_','recsys_','continual_'))
        g,a,p=[r[m]['test']['score'] for m in ('GrassRoute','AdamW','PeriodicSVD')]
        better=lambda x,y:x>y if high else x<y
        if better(g,a) and better(g,p):paired.append(t)
        if len(r)==6 and all(not better(x['test']['score'],g) for x in r.values()):best.append(t)
    status={'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'complete_pairs':total,'required_pairs':162,'complete_configurations':full,
            'original_suite_scope':{'planned_pairs':162,'completed_before_reprioritization':117,
                                    'remaining_queue':'stopped on 2026-11-04; incomplete records retained',
                                    'decision':'post-hoc compute reprioritization after inspecting outcomes'},
            'grassroute_better_than_adam_and_periodic':paired,'grassroute_best_all_six':best,
            'grassroute_better_than_adam':adam_positive,
            'by_task':{t:{'complete':len(r),'methods':list(r)} for t,r in rows.items()},'ledger':ledger}
    status['main_text_primary_gains']=write_completed_primary_gains(rows,full,paired)
    for t in TASKS:
        partial={}
        for m in METHODS:
            path=HERE/'results'/t/m/'formal/trajectory.json'
            if m in rows[t] or not path.exists():continue
            try:
                records=json.loads(path.read_text())
                if records:partial[m]={'saved_epoch':records[-1]['epoch'],'wall_s':records[-1]['wall_s']}
            except (ValueError,KeyError):pass
        if partial:status['by_task'][t]['incomplete_formal_checkpoints']=partial
    (HERE/'status.json').write_text(json.dumps(status,indent=2))
    progress=[f"更新时间 UTC: {status['generated_utc']}",f"正式实验已完成 {total}/162；六个方法全部齐全的配置 {len(full)}/27。",
              '', '原162项长实验于完成117项后停止排队；保留未完成单元和已保存检查点。转向明确标记的探索性复核。',
              '历史正面结果：Burgers/PINN 是唯一同时优于 AdamW 和 tuned Periodic-SVD 的明确任务族；Cora 只优于 AdamW；CNN partial 精度较好但仅省约1.8%状态内存。',
              '新增结果与历史结果分开；没有完成的实验不计入结论；单 seed 不声称统计显著。',
              '', '新增同时优于 AdamW 和 Periodic-SVD 的配置：'+(', '.join(paired) if paired else '目前没有。'),
              '新增优于 AdamW 的配置（不代表胜过其余基线）：'+(', '.join(adam_positive) if adam_positive else '目前没有。'),
              '', '逐配置正式完成数：']
    progress += [f"{t}: {len(rows[t])}/6" for t in TASKS]
    progress += ['', '探索性动态回归和Cora GCN完整结果：revision/focused_results。',
                 '实时机器可读明细：revision/status.json；远端日志副本：revision/remote_status/worker_gpu*.log。',
                 '按需单次运行 revision/watch.py --once 同步、验证并编译；原有定时任务保持删除状态。']
    (HERE/'进度.txt').write_text('\n'.join(progress)+'\n')
    text=f'The current extension contains {total} completed formal task--method runs out of 162, with {len(full)} configurations complete across all six methods. '
    text+='The original queue was stopped after 117 completed runs; unfinished cells are marked -- and have not been used to declare a winner. '
    if paired:text+='GrassRoute improves on both AdamW and periodic SVD on the completed paired comparisons '+', '.join(label(t) for t in paired)+'. '
    else:text+='The completed extension comparisons do not yet establish a new case that improves on both AdamW and periodic SVD. '
    if adam_positive:text+='Against AdamW alone, the completed paired scores favor GrassRoute on '+', '.join(label(t) for t in adam_positive)+'; this is a qualified comparison, not an all-baseline win. '
    text+='All values are final held-out test scores of the validation-selected configuration; classification and Recall@20 are percentages.\n'
    (FIG/'revision_summary.tex').write_text(text)
    key=[t for t in TASKS if not t.startswith(('matreg_','lowrank_'))]
    table(rows,key,'revision_main_table.tex',
          'Validation-selected extension, seed 42. Formal budgets are 1000 epochs for classification and recommendation, 2000 for MLP and scientific learning. Accuracy and Recall@20 are percentages; scientific results report solution relative error. Bold marks a best score only after all six methods have completed. -- denotes an unfinished formal comparison. GrassRoute uses diagonal moment transport; the periodic comparator retains its old second-moment diagonal at refreshes. A transport-corrected follow-up is reported separately. SubTrack++ is the pinned official implementation.',
          'tab:extension')
    table(rows,[t for t in TASKS if t.startswith(('matreg_','lowrank_'))],'revision_regression_table.tex',
          'Generated regression extension at 2000 full epochs. Low-rank tasks have an explicitly planted rank-eight target matrix. The dimension in the task label is input dimension; output dimension is 32, with 2048 training and two independent 256-example validation/test sets. Covariance and observation-noise definitions are recorded in the configuration. -- denotes an unfinished formal comparison.',
          'tab:regression-extension')
    pilot=[r'\begin{table}[htbp]\centering\scriptsize',
           r'\caption{Additional covariance-concentration pilot, retained separately from the proposal-completion count. These 2000-epoch runs have full-rank target matrices and concentrated input covariance; they do not satisfy the planted-low-rank-target benchmark. Values are final test MSE.}',
           r'\begin{tabular}{rrrrrrr}\toprule $d$ & SGD-M & AdamW & Fixed & Periodic & GrassRoute & SubTrack++\\\midrule']
    pilot_count=0
    for d in [128,256,512]:
        vals=[]
        for m in METHODS:
            pp=HERE/'exploratory_archive/initial_coverage_pilots'/f'lowrank_{d}'/m/'formal/summary.json'
            if pp.exists():
                xx=json.loads(pp.read_text());vals.append(number(xx['test']['score'],False));pilot_count+=1
            else:vals.append('--')
        pilot.append(str(d)+' & '+' & '.join(vals)+r' \\')
    pilot.extend([r'\bottomrule\end{tabular}\end{table}'])
    (FIG/'revision_covariance_pilot_table.tex').write_text('\n'.join(pilot)+'\n')
    # Audit detail rows are source-linked through the machine-readable ledger.
    audit=[r'{\scriptsize\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{llrrrr}',r'\caption{Completed-run resource and selection audit. Persistent state is in decimal MB. Wall time includes validation, diagnostics and checkpointing on a shared host; it is not an isolated speed benchmark.}\\',
           r'\toprule Task & Method & LR & State MB & Seconds & Epochs \\ \midrule\endfirsthead',
           r'\toprule Task & Method & LR & State MB & Seconds & Epochs \\ \midrule\endhead']
    for t in TASKS:
        for m in METHODS:
            if m not in rows[t]:continue
            x=rows[t][m]
            audit.append(f"{label(t)} & {NAMES[m]} & {x['lr']:.4g} & {x['state_bytes']/1e6:.3f} & {x['wall_s']:.1f} & {x['epochs']}"+r' \\')
    audit.extend([r'\bottomrule\end{longtable}',r'}'])
    (FIG/'revision_resources.tex').write_text('\n'.join(audit)+'\n')
    # Scientific trajectories are generated directly from complete run records.
    fig,axs=plt.subplots(2,2,figsize=(9,6.3),constrained_layout=True)
    for ax,t in zip(axs.flat,['sciml_poisson','sciml_burgers','sciml_darcy','sciml_fno']):
        for m in METHODS:
            p=HERE/'results'/t/m/'formal/trajectory.json'
            if m not in rows[t] or not p.exists():continue
            a=json.loads(p.read_text());ep=[r['epoch'] for r in a if 'validation' in r];v=[r['validation']['score'] for r in a if 'validation' in r]
            ax.semilogy(ep,v,label=NAMES[m],color=f'C{METHODS.index(m)}',lw=2.1 if m=='GrassRoute' else 1.3)
        ax.set_title(label(t),fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Validation relative $L_2$');ax.grid(alpha=.2)
    handles,labels=axs.flat[0].get_legend_handles_labels()
    if handles:axs.flat[0].legend(handles,labels,ncol=2,frameon=False,fontsize=7)
    save_figure(fig,FIG/'revision_sciml.pdf');plt.close(fig)
    mech=HERE/'remote_status/mechanism_results.json'
    if mech.exists():
        mm=json.loads(mech.read_text());c=mm['checks']
        fig,axs=plt.subplots(1,3,figsize=(11,3.1),constrained_layout=True)
        a=c['landscape'];th=np.array([r['theta'] for r in a]);axs[0].plot(th,[r['gradient_squared'] for r in a],label='Gradient squared');axs[0].plot(th,[r['pl_lower'] for r in a],'--',label='Angular PL bound');axs[0].axvline(np.pi/4,color='gray',ls=':');axs[0].set_xlabel('Principal angle (radians)');axs[0].set_title('Fixed eigengap, varied angle');axs[0].legend(fontsize=7)
        a=c['qr'];axs[1].loglog([r['step'] for r in a],[r['projector_error'] for r in a],'o-',label='Measured');axs[1].loglog([r['step'] for r in a],[r['cubic_bound'] for r in a],'--',label='Cubic upper bound');axs[1].set_xlabel('Tangent step norm');axs[1].set_title('QR versus exponential projector');axs[1].legend(fontsize=7)
        for gap in [.05,.2,.5]:
            a=[r for r in mm['tracking'] if r['gap']==gap and r['rho']==.15]
            axs[2].plot([r['drift_radians_per_step'] for r in a],[r['mean_sin2_last500']['OnlineQR'] for r in a],'o-',label=f'gap={gap}')
        axs[2].set_xlabel('Target rotation / step');axs[2].set_ylabel('Mean squared sine error');axs[2].set_title('Controlled raw-covariance tracking');axs[2].legend(fontsize=7)
        for ax in axs:ax.grid(alpha=.2)
        save_figure(fig,FIG/'revision_geometry.pdf');plt.close(fig)
    cifar=HERE/'remote_status/cifar_stability_results.json'
    if cifar.exists():
        cc=json.loads(cifar.read_text())
        fig,axs=plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
        for m,x in cc['results'].items():
            a=x['records'];axs[0].semilogy([r['epoch'] for r in a],[max(r['parameter_distance'],1e-12) for r in a],label=m)
            a=[r for r in a if 'mean_bounded_loss_sensitivity' in r];axs[1].semilogy([r['epoch'] for r in a],[max(r['mean_bounded_loss_sensitivity'],1e-12) for r in a],label=m)
        axs[0].set_ylabel('Neighboring-run parameter distance');axs[1].set_ylabel('Mean bounded-loss sensitivity')
        for ax in axs:ax.set_xlabel('Full-batch epoch');ax.grid(alpha=.2)
        axs[1].legend(fontsize=8)
        save_figure(fig,FIG/'revision_stability.pdf');plt.close(fig)
        if all(m in cc['results'] for m in ['AdamW','GrassRoute','HardReset','Fixed']):
            last={m:cc['results'][m]['records'][-1] for m in ['AdamW','GrassRoute','HardReset','Fixed']}
            text='In this auxiliary probe, final mean bounded-loss sensitivity is '
            text+=', '.join(f"{last[m]['mean_bounded_loss_sensitivity']:.5f} ({m})" for m in ['AdamW','GrassRoute','HardReset','Fixed'])+'. '
            text+=f"AdamW and GrassRoute obtain {100*last['AdamW']['test_accuracy']:.1f}\\% and {100*last['GrassRoute']['test_accuracy']:.1f}\\% test accuracy, respectively; hard reset obtains {100*last['HardReset']['test_accuracy']:.1f}\\% and fixed routing {100*last['Fixed']['test_accuracy']:.1f}\\%. "
            text+=f"GrassRoute's training cross-entropy is {last['GrassRoute']['train_loss']:.3f} versus AdamW's {last['AdamW']['train_loss']:.3f}; its lower sensitivity is therefore accompanied by substantially weaker fitting. The fixed router is less sensitive still, but also less accurate. These measurements illustrate a fit--sensitivity trade-off and reset instability, not superior population generalization.\n"
            (FIG/'revision_sensitivity_summary.tex').write_text(text)
    from strength_report import generate as generate_strength
    strength=generate_strength(rows)
    from new_tasks_report import generate as generate_newtasks
    nt=generate_newtasks()
    status['independent_followup']={k:v for k,v in strength.items() if k in ('tracking','tracking_extended','pinn','graph')}
    status['new_public_tasks']={k:v for k,v in nt.items() if k not in ('progress','failed','ledger')}
    sync_path=HERE/'data_sync_status.json'
    if sync_path.exists():status['data_sync']=json.loads(sync_path.read_text())
    from review_tables import generate as generate_review_tables
    generate_review_tables(rows)
    from focused_report import generate as generate_focused
    status['focused_followup'] = generate_focused()
    from focused_replication_report import generate as generate_replications
    status['focused_followup'].update(generate_replications())
    (HERE/'status.json').write_text(json.dumps(status,indent=2))
    with open(HERE/'进度.txt','a') as f:
        f.write('\n独立补充实验（不混入162个原定对比）：\n')
        if 'pinn' in strength:f.write(f"精确解PINN复核：{strength['pinn']['complete']}/8 个2000-epoch正式训练完成；72个验证筛选完成。\n")
        if 'tracking_extended' in strength:f.write(f"强跟踪对照：{strength['tracking_extended']['conditions']} 个条件完成；GrassRoute在其中 {strength['tracking_extended']['grass_best']} 个条件误差最低，包含完整EMA-SVD对照。\n")
        if 'graph' in strength:f.write(f"Planetoid扩大搜索复核：{strength['graph']['complete']}/18 个1000-epoch正式训练完成。\n")
        else:f.write('Planetoid扩大搜索复核：3个CPU进程训练中，18个正式对比待完成。\n')
        f.write(f"新增三类六任务：{nt['complete_pairs']}/36 正式对比；{nt['screen_attempts_completed']}/324 个验证候选已完成尝试。\n")
        f.write('新任务的主指标双基线优势：'+(', '.join(nt['paired_primary_gains']) if nt['paired_primary_gains'] else '尚无完整新对比满足预设条件。')+'\n')
        ff=status['focused_followup']
        f.write(f"探索性动态回归：{ff['regression_complete']}/{ff['regression_required']}；Cora GCN：{ff['graph_complete']}/{ff['graph_required']}。\n")
        f.write(f"冻结超参的独立初始化：Cora GraphSAGE {ff['cora_complete']}/{ff['cora_required']}；SMS {ff['sms_complete']}/{ff['sms_required']}。\n")
        f.write(f"Cora 复核中的数值失败（保留，不用测试集重选）：{ff['cora_numerical_failures']}。\n")
    return status


if __name__=='__main__':
    s=generate();print(json.dumps({k:v for k,v in s.items() if k not in ('by_task','ledger')},indent=2))
