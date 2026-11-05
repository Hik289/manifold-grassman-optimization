"""Render non-loss diagnostics as native tables, using the WWW reference style.

The numerical records and experiment selection are unchanged. Called after
every report refresh so the background updater preserves the manuscript style.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / 'paper/figures'


def sci(x):
    if x == 0:
        return '0'
    if .001 <= abs(x) < 1000:
        return f'{x:.4g}'
    a, b = f'{x:.2e}'.split('e')
    return '$' + a + r'\times10^{' + str(int(b)) + '}$'


def table(name, caption, label, header, rows, spec=None, wide=False):
    env = 'table*' if wide else 'table'
    spec = spec or ('l' + 'r' * (len(header) - 1))
    parts = [r'\begin{' + env + r'}[t]\centering\footnotesize',
             r'\caption{\textbf{' + caption.split('.')[0] + '.} ' + '.'.join(caption.split('.')[1:]).strip() + '}',
             r'\label{' + label + '}', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{' + spec + '}', r'\toprule',
             ' & '.join(header) + r'\\\midrule']
    parts.extend(' & '.join(map(str, row)) + r'\\' for row in rows)
    parts.extend([r'\bottomrule\end{tabular}', r'\end{' + env + '}'])
    (FIG / name).write_text('\n'.join(parts) + '\n')


def generate_diagnostics(completed):
    mm = json.loads((HERE / 'remote_status/mechanism_results.json').read_text())
    c = mm['checks']
    rows = [
        ['Random covariance/subspace cases', str(c['random_cases'])],
        ['Minimum measured / PL lower bound', sci(c['min_pl_ratio'])],
        ['Minimum measured / quadratic-growth bound', sci(c['min_growth_ratio'])],
        ['Maximum directional-derivative error', sci(c['max_directional_derivative_error'])],
    ]
    table('geometry_checks_table.tex',
          'Controlled geometry checks. Random pairs cover $(d,r)=(6,1),(8,3),(8,6),(32,8)$; ratios at least one verify the sampled lower bounds.',
          'tab:geometry-checks', ['Check', 'Value'], rows)
    a = c['landscape']
    selected = [a[i] for i in sorted(set([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, len(a)-1]))]
    table('landscape_checks_table.tex',
          'Principal-angle diagnostic. Eleven evenly spaced stored angles from the 200-point sweep at fixed eigengap; all original records remain in the diagnostic archive. The angular PL lower bound coincides with the exact rank-one squared gradient.',
          'tab:landscape-checks', [r'$\theta$', r'$\|\mathrm{grad}\,\Phi\|^2$', 'PL bound', 'Hessian'],
          [[f"{x['theta']:.4f}", sci(x['gradient_squared']), sci(x['pl_lower']), sci(x['hessian'])] for x in selected])
    table('qr_checks_table.tex',
          'QR versus exponential-map projectors. All 16 tested tangent norms; the measured discrepancy is compared with the cubic bound.',
          'tab:qr-checks', ['Tangent norm', 'Projector error', 'Cubic bound'],
          [[sci(x['step']), sci(x['projector_error']), sci(x['cubic_bound'])] for x in c['qr']])
    table('raw_tracking_table.tex',
          'Raw-covariance tracking diagnostic. All 27 gap--drift--step configurations; mean squared sine error on the final 500 of 2000 steps. Each method observes the same stream. No momentum or covariance normalization is used in this auxiliary bed.',
          'tab:raw-tracking', ['Eigengap', 'Drift', r'$\rho$', 'Online QR', 'Periodic / 25', 'Fixed'],
          [[sci(x['gap']), sci(x['drift_radians_per_step']), sci(x['rho'])] +
           [sci(x['mean_sin2_last500'][k]) for k in ['OnlineQR', 'Periodic25', 'Fixed']] for x in mm['tracking']])
    cc = json.loads((HERE / 'remote_status/cifar_stability_results.json').read_text())
    rows = []
    for method, val in cc['results'].items():
        x = val['records'][-1]
        rows.append([method, sci(x['train_loss']), f"{100*x['test_accuracy']:.2f}",
                     sci(x['parameter_distance']), sci(x['mean_bounded_loss_sensitivity']),
                     sci(x['max_bounded_loss_sensitivity'])])
    table('sensitivity_checks_table.tex',
          'Neighboring-data probe at epoch 1000. Frozen random ResNet-18 features on CIFAR-100; one replacement in 2048 training examples and a 1000-image test probe. Sensitivity uses cross-entropy clipped at 10 and divided by 10. These measurements are conditional on the specified dataset pair.',
          'tab:stability-checks', ['Method', 'Train CE', 'Test acc.', 'Parameter distance', 'Mean sensitivity', 'Max. sensitivity'], rows)
    td = json.loads((HERE / 'strength_results/tracking_summary.json').read_text())
    rows = []
    for name in td['protocol']['methods']:
        cond = [x['summary'][name] for x in td['conditions'] if x['rows'] == 128]
        ms = 1000 * sum(x['optimizer_seconds'] / 2000 for x in cond) / len(cond)
        rows.append([name, sci(ms), f"{cond[0]['state_bytes']/1000:.2f}"])
    table('tracking_resource_table.tex',
          'Tracking implementation resources. Mean CPU optimizer-call time across all four drift rates at 128 gradient rows, including diagnostics; persistent state is decimal kB. Shared implementation timing is reported separately from task accuracy.',
          'tab:tracking-resources', ['Method', 'CPU call / ms', 'State / kB'], rows, wide=True)
    rows = []
    for task in ['sciml_poisson', 'sciml_burgers', 'sciml_darcy', 'sciml_fno']:
        for method, x in completed[task].items():
            a = json.loads((HERE / 'results' / task / method / 'formal/trajectory.json').read_text())
            v = [r for r in a if 'validation' in r]
            from report import label, NAMES
            rows.append([label(task), NAMES[method], sci(v[0]['validation']['score']),
                         sci(v[-1]['validation']['score']), sci(x['test']['score'])])
    table('scientific_endpoints_table.tex',
          'Scientific-learning validation and test endpoints. Complete 2000-epoch formal runs from the original proposal extension, selected on validation data. All entries are solution relative error. Initial validation denotes the first stored validation checkpoint; models and tasks follow the versioned protocol.',
          'tab:new-sciml', ['Task', 'Method', 'Initial val.', 'Final val.', 'Final test'], rows)


def style_tabular(text):
    """Highlight a method row or the GrassRoute result column; preserve values."""
    if '% WWW table style' in text:
        return text
    text = text.replace(r'\rowcolor{gray!12}', r'\rowcolor{oursbg}')
    token = r'\begin{tabular}'
    pos = 0
    while (start := text.find(token, pos)) >= 0:
        left = start + len(token)
        while text[left].isspace():
            left += 1
        assert text[left] == '{'
        level, right = 1, left + 1
        while level:
            level += (text[right] == '{') - (text[right] == '}')
            right += 1
        end = text.index(r'\end{tabular}', right)
        replacement = style_body(text[left:right], text[right:end])
        text = text[:start] + replacement + text[end+len(r'\end{tabular}'):]
        pos = start + len(replacement)
    # Repeated headers and method rows in multipage audit tables.
    def long_body(m):
        body = m.group(2)
        body = re.sub(r'(\\toprule\s*)(.*?)(\\\\)',
                      lambda h: h.group(1) + ' & '.join(r'{\bfseries ' + x.strip() + '}' for x in h.group(2).split('&')) + h.group(3), body, flags=re.S)
        body = re.sub(r'(?m)^([^\n]* & )GrassRoute( &[^\n]*)$', r'\\rowcolor{oursbg}\1\\textbf{GrassRoute}\2', body)
        return r'\begin{longtable}' + m.group(1) + body + r'\end{longtable}'
    text = re.sub(r'\\begin\{longtable\}(\{[^\n]*?\})(.*?)\\end\{longtable\}', long_body, text, flags=re.S)
    return '% WWW table style: three rules, bold headers, reference green highlight.\n' + text


def style_body(spec, body):
    if r'\toprule' not in body or r'\midrule' not in body:
        return r'\reviewfit{\begin{tabular}' + spec + body + r'\end{tabular}}'
    pre, rest = body.split(r'\toprule', 1)
    head, tail = rest.split(r'\midrule', 1)
    header_rows = head.split(r'\\')
    cells = header_rows[0].split('&')
    ours = next((i for i, cell in enumerate(cells) if 'GrassRoute' in cell), None)
    styled = []
    for row in header_rows:
        if not row.strip():
            styled.append(row)
            continue
        out = []
        for i, cell in enumerate(row.split('&')):
            color = r'\cellcolor{oursbg}' if i == ours else ''
            out.append(color + r'{\bfseries ' + cell.strip() + '}')
        styled.append(' & '.join(out))
    head = r'\\'.join(styled)
    data = []
    for row in tail.split(r'\\'):
        cell = row.split('&')
        if len(cell) > 1:
            if ours is not None and len(cell) > ours:
                cell[ours] = r' \cellcolor{oursbg}' + cell[ours].strip()
            elif any('GrassRoute' in x for x in cell[:2]) and r'\rowcolor' not in row:
                for j in range(min(2,len(cell))):
                    if cell[j].strip()=='GrassRoute':
                        cell[j]=r'\textbf{GrassRoute}'
                cell[0] = '\n' + r'\rowcolor{oursbg}' + cell[0].lstrip()
            row = ' & '.join(cell)
        data.append(row)
    return r'\reviewfit{\begin{tabular}' + spec + pre + r'\toprule ' + head + r'\midrule ' + r'\\'.join(data) + r'\end{tabular}}'


def apply_style():
    # Main-text wide tables; the theory and complete audit remain one column.
    wide = ['strength_tracking_table.tex', 'strength_tracking_ema_table.tex',
            'strength_pinn_table.tex']
    for path in list(FIG.glob('*.tex')) + list((ROOT / 'paper/sections').glob('*.tex')):
        text = path.read_text()
        if path.name in wide or path.name == '2_related_work.tex':
            text = text.replace(r'\begin{table}', r'\begin{table*}').replace(r'\end{table}', r'\end{table*}')
        if 'newtasks_selected_fig' in path.name:
            text = text.replace(r'\begin{figure}', r'\begin{figure*}').replace(r'\end{figure}', r'\end{figure*}')
        if r'\begin{tabular}' in text or r'\begin{longtable}' in text:
            text = style_tabular(text)
        if text != path.read_text():
            path.write_text(text)


def generate(completed):
    generate_diagnostics(completed)
    apply_style()


if __name__ == '__main__':
    from report import load
    generate(load()[0])
