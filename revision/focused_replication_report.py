"""Audit and render seed replications of the graph and SMS evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'paper' / 'figures'
ROOT = HERE / 'focused_results'
SEEDS = [43, 44, 45]
METHODS = ['SGD', 'AdamW', 'PeriodicDiag', 'GrassRoute']
SMS_METHODS = ['SGD', 'AdamW', 'PeriodicThinSVD', 'GrassRoute']


def verified(family, seed, method):
    base = ROOT / family
    protocol_path = base / 'protocol.json'
    p = base / f'seed_{seed}' / method / 'summary.json'
    if not p.exists() or not protocol_path.exists():
        return None
    protocol = json.loads(protocol_path.read_text())
    x = json.loads(p.read_text())
    trajectory = json.loads(p.with_name('trajectory.json').read_text())
    assert x['status'] == 'complete' and x['formal'] is True
    assert x['epochs'] == x['n_epochs'] == len(trajectory) == 1000
    assert trajectory[-1]['epoch'] == 1000
    assert x['seed'] == seed and x['method'] == method
    assert x['hp'] == protocol['fixed_hyperparameters'][method]
    assert x['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert math.isfinite(x['test']['score'])
    return x


def table(family, methods, title, label, file):
    results = {m: [verified(family, seed, m) for seed in SEEDS] for m in methods}
    rows = [r'\begin{table*}[t]\centering\footnotesize',
            r'\caption{\textbf{' + title + r'} Three fresh 1000-epoch runs use',
            r'hyperparameters frozen from the original seed-42 validation selection.',
            r'The public split and architecture are fixed; seeds 43--45 vary',
            r'initialization and stochastic training. Accuracy or macro-F1 is in',
            r'percent. Mean and sample standard deviation are descriptive.',
            r'Persistent optimizer state is in decimal MB. All endpoints are shown;',
            r'\emph{fail} denotes a retained numerical failure.}',
            r'\label{' + label + r'}',r'\setlength{\tabcolsep}{7pt}',
            r'\begin{tabular}{lrrrrr}\toprule',
            r'Method & Seed 43 & Seed 44 & Seed 45 & Mean $\pm$ SD & State MB \\ \midrule']
    for method in methods:
        points = results[method]
        vals = []
        for i, x in enumerate(points):
            if x:
                vals.append(f"{100*x['test']['score']:.2f}")
            elif (ROOT / family / f'seed_{SEEDS[i]}' / method / 'failed.json').exists():
                vals.append(r'\emph{fail}')
            else:
                vals.append('--')
        scores = [100*x['test']['score'] for x in points if x]
        avg = (f'{statistics.mean(scores):.2f} $\\pm$ {statistics.stdev(scores):.2f}'
               if len(scores) == 3 else '--')
        completed = next((x for x in points if x), None)
        state = f'{completed["state_bytes"]/1e6:.3f}' if completed else '--'
        if method == 'GrassRoute':
            vals = [r'\cellcolor{oursbg}' + value for value in vals]
            avg = r'\cellcolor{oursbg}' + avg
        name = method.replace('PeriodicDiag', 'Periodic SVD').replace('PeriodicThinSVD', 'Periodic SVD')
        rows.append(name + ' & ' + ' & '.join(vals + [avg, state]) + r' \\')
    rows += [r'\bottomrule\end{tabular}\end{table*}']
    (FIG / file).write_text('\n'.join(rows) + '\n')
    return results


def generate():
    cora = table('cora_sage_replication', METHODS,
                 'Cora GraphSAGE fixed-hyperparameter seed replication.',
                 'tab:focused-cora-replication', 'focused_cora_replication_table.tex')
    sms = table('sms_replication', SMS_METHODS,
                'SMS text classification fixed-hyperparameter seed replication.',
                'tab:focused-sms-replication', 'focused_sms_replication_table.tex')
    graph_positive = False
    failures = list((ROOT / 'cora_sage_replication').glob('seed_*/*/failed.json'))
    if all(all(v) for v in cora.values()):
        mean = {m:statistics.mean(100*x['test']['score'] for x in cora[m]) for m in METHODS}
        graph_positive = mean['GrassRoute'] > mean['AdamW'] and mean['GrassRoute'] > mean['PeriodicDiag']
        if graph_positive:
            wins = {m:sum(cora['GrassRoute'][i]['test']['score'] > cora[m][i]['test']['score']
                          for i in range(3)) for m in ('AdamW','PeriodicDiag')}
            saving = 100*(1-cora['GrassRoute'][0]['state_bytes']/cora['AdamW'][0]['state_bytes'])
            (FIG / 'focused_cora_replication_main.tex').write_text(
                f'With seed-42 validation choices frozen, three new Cora GraphSAGE '
                f'initializations give GrassRoute {mean["GrassRoute"]:.2f}\\% mean final '
                f'test accuracy versus {mean["AdamW"]:.2f}\\% for AdamW and '
                f'{mean["PeriodicDiag"]:.2f}\\% for corrected periodic SVD. '
                f'GrassRoute wins {wins["AdamW"]}/3 and {wins["PeriodicDiag"]}/3 '
                f'paired seeds, respectively, while using {saving:.1f}\\% less '
                f'persistent state than AdamW. Table~\\ref{{tab:focused-cora-replication}} '
                f'retains SGD and every seed. These are repeated initializations '
                f'on the same public graph split.\\par\n')
        else:
            (FIG / 'focused_cora_replication_main.tex').write_text('')
    else:
        terminal = sum(x is not None for v in cora.values() for x in v) + len(failures)
        if terminal == 12 and all(all(cora[m]) for m in ('SGD', 'AdamW', 'GrassRoute')):
            mean = {m:statistics.mean(100*x['test']['score'] for x in cora[m])
                    for m in ('SGD', 'AdamW', 'GrassRoute')}
            wins = sum(cora['GrassRoute'][i]['test']['score'] >
                       cora['AdamW'][i]['test']['score'] for i in range(3))
            periodic = [100*x['test']['score'] for x in cora['PeriodicDiag'] if x]
            periodic_text = ', '.join(f'{x:.2f}\\%' for x in periodic)
            (FIG / 'focused_cora_replication_main.tex').write_text(
                f'With the seed-42 Cora GraphSAGE hyperparameters frozen, three '
                f'new initializations give mean final test accuracy '
                f'{mean["GrassRoute"]:.2f}\\% for GrassRoute, '
                f'{mean["AdamW"]:.2f}\\% for AdamW and '
                f'{mean["SGD"]:.2f}\\% for SGD-M. GrassRoute exceeds AdamW '
                f'in {wins}/3 paired seeds. Corrected periodic SVD records '
                f'{periodic_text} on its completed seeds and a retained '
                f'numerical failure on the other. The favorable seed-42 '
                f'accuracy--state observation is therefore not consistent '
                f'across these new initializations; all endpoints and the '
                f'failure appear in Table~\\ref{{tab:focused-cora-replication}}.\\par\n')
        else:
            (FIG / 'focused_cora_replication_main.tex').write_text('')
    sms_tradeoff = False
    if all(all(v) for v in sms.values()):
        mean = {m:statistics.mean(100*x['test']['score'] for x in sms[m]) for m in SMS_METHODS}
        saving = 100*(1-sms['GrassRoute'][0]['state_bytes']/sms['AdamW'][0]['state_bytes'])
        within = sum(100*(sms['AdamW'][i]['test']['score'] -
                          sms['GrassRoute'][i]['test']['score']) <= 1 for i in range(3))
        periodic_wins = sum(sms['GrassRoute'][i]['test']['score'] >
                            sms['PeriodicThinSVD'][i]['test']['score'] for i in range(3))
        sms_tradeoff = (saving >= 50 and mean['GrassRoute'] >= mean['AdamW']-1
                        and mean['GrassRoute'] > mean['PeriodicThinSVD'])
        if sms_tradeoff:
            (FIG / 'focused_sms_replication_main.tex').write_text(
                f'With the seed-42 validation choices frozen, the three new SMS '
                f'initializations yield {mean["GrassRoute"]:.2f}\\% mean macro-F1 '
                f'for GrassRoute, {mean["AdamW"]:.2f}\\% for AdamW and '
                f'{mean["PeriodicThinSVD"]:.2f}\\% for thin periodic SVD. '
                f'The GrassRoute state is {saving:.1f}\\% smaller than AdamW\'s; '
                f'its mean F1 is {mean["AdamW"]-mean["GrassRoute"]:.2f} points '
                f'below AdamW. The one-point quality band holds in {within}/3 '
                f'seeds, and GrassRoute exceeds periodic SVD in '
                f'{periodic_wins}/3 seeds. This is a descriptive mean '
                f'memory--quality trade-off, with visible seed variability. '
                f'All seeds and SGD appear in '
                f'Table~\\ref{{tab:focused-sms-replication}}.\\par\n')
        else:
            (FIG / 'focused_sms_replication_main.tex').write_text('')
    else:
        (FIG / 'focused_sms_replication_main.tex').write_text('')
    return dict(cora_complete=sum(x is not None for v in cora.values() for x in v),
                cora_required=12, cora_mean_joint_gain=graph_positive,
                cora_numerical_failures=len(failures),
                sms_complete=sum(x is not None for v in sms.values() for x in v),
                sms_required=12, sms_mean_memory_quality_tradeoff=sms_tradeoff)


if __name__ == '__main__':
    print(json.dumps(generate(), indent=2))
