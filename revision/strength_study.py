"""Independent, preregistered mechanism and PINN follow-up; original runs immutable.

Run --mode tracking or --mode pinn. Protocol is written before any result.
Only the corrected PeriodicDiag baseline is used in these follow-up comparisons.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'experiments'))
from grassroute import GrassRoute, _polar_orthogonal
from baselines import PeriodicSVDRouter

OUT = HERE / 'strength_results'
torch.set_num_threads(2)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def seeded(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class PeriodicDiag(PeriodicSVDRouter):
    """Upstream local periodic router with its omitted V transport restored.

    Base class transports M and changes U, but leaves V in old coordinates.
    Recover R from pre/post frames and rotate the already-updated diagonal.
    V age is retained. No change to the original baseline source or checkpoints.
    """
    @torch.no_grad()
    def _step_routed(self, group, p):
        old = self.state[p].get('U')
        super()._step_routed(group, p)
        st = self.state[p]
        if old is not None and st['svd_ctr'] % group['svd_every'] == 0:
            rot = _polar_orthogonal(st['U'].T @ old)
            st['V'].copy_(st['V'] @ rot.T.square())


def register(mode, protocol):
    path = OUT / (mode + '_protocol.json')
    protocol = dict(protocol, seed=42,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if path.exists():
        old = json.loads(path.read_text())
        assert old['source_sha256'] == protocol['source_sha256'], 'Version changed: preserve old result first.'
        return old
    dump(path, protocol)
    return protocol


def memory(opt):
    from suite import state_bytes
    return state_bytes(opt)


def tracking():
    # A fixed Cartesian grid, including static and fast-drift negative controls.
    protocol = register('tracking', dict(
        d=64, rank=4, rows=[16, 128], drift=[0., .001, .005, .02],
        signal_eigenvalue=1., tail_eigenvalue=.03, steps=2000,
        evaluation='pre-update mean sin^2 angle over steps 1501:2000',
        initialization='common random QR, not population oracle',
        methods=['GrassRoute', 'NoMomentum', 'Fixed', 'Periodic5', 'Periodic25', 'Periodic100'],
        grassmann_lr=3., momentum=.9, reset_mode='diagonal',
        scope='Gaussian gradient stream; actual routed optimizer states; not task accuracy'))
    rows = []
    for m, omega in itertools.product(protocol['rows'], protocol['drift']):
        key = f'm{m}_omega{omega:g}'
        path = OUT / 'tracking' / (key + '.json')
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        seeded()
        d, r = protocol['d'], protocol['rank']
        initial = torch.linalg.qr(torch.randn(d, r)).Q
        opts = {}
        for method in protocol['methods']:
            p = torch.nn.Parameter(torch.zeros(m, d))
            kw = dict(lr=0., rank=r, grassmann_lr=3., reset_mode='diagonal')
            if method.startswith('Periodic'):
                opt = PeriodicDiag([p], svd_every=int(method[8:]), **kw)
            else:
                if method == 'Fixed': kw['grassmann_lr'] = 0.
                if method == 'NoMomentum': kw['grassmann_momentum'] = False
                opt = GrassRoute([p], **kw)
            opt._init_routed_state(opt.state[p], p, r, 'cpu', p.dtype)
            opt.state[p]['U'] = initial.clone()
            if method.startswith('Periodic'): opt.state[p]['svd_ctr'] = 0
            opts[method] = (p, opt)
        gen = torch.Generator().manual_seed(4201)
        records = []
        elapsed = {k: 0. for k in opts}
        maxorth = 0.
        for t in range(protocol['steps']):
            angle = omega*t
            target = torch.zeros(d, r)
            target[:r] = math.cos(angle)*torch.eye(r)
            target[r:2*r] = math.sin(angle)*torch.eye(r)
            z = torch.randn(m, d, generator=gen)
            # Gaussian covariance tail I + (signal-tail) P. Rotating paired axes
            # leaves all tail directions isotropic; normalized population has
            # the same top eigenspace by block rotational symmetry.
            z[:, :r] *= 1.
            z[:, r:] *= math.sqrt(.03)
            a, b = z[:, :r].clone(), z[:, r:2*r].clone()
            z[:, :r] = math.cos(angle)*a - math.sin(angle)*b
            z[:, r:2*r] = math.sin(angle)*a + math.cos(angle)*b
            rec = {'step': t+1, 'methods': {}}
            for name, (p, opt) in opts.items():
                u = opt.state[p]['U']
                sin2 = max(0., 1.-float((u.T@target).square().sum())/r)
                cap = float((z@u).square().sum()/z.square().sum())
                p.grad = z
                start = time.perf_counter()
                opt.step()
                elapsed[name] += time.perf_counter()-start
                unew = opt.state[p]['U']
                maxorth = max(maxorth, float((unew.T@unew-torch.eye(r)).norm()))
                rec['methods'][name] = {'sin2': sin2, 'captured': cap}
            records.append(rec)
        summary = {name: {
            'mean_sin2_last500': float(np.mean([x['methods'][name]['sin2'] for x in records[-500:]])),
            'captured_last500': float(np.mean([x['methods'][name]['captured'] for x in records[-500:]])),
            'state_bytes': memory(opt), 'optimizer_seconds': elapsed[name]}
            for name, (_, opt) in opts.items()}
        result = dict(key=key, rows=m, drift=omega, summary=summary,
                      max_orthogonality_error=maxorth, records=records)
        assert maxorth < 1e-4
        dump(path, result)
        rows.append(result)
        print(key, summary, flush=True)
    dump(OUT/'tracking_summary.json', dict(protocol=protocol, conditions=[{k:v for k,v in x.items() if k!='records'} for x in rows]))


def candidates(method):
    if method == 'AdamW':
        return [dict(lr=float(x)) for x in np.geomspace(.0001, .03, 9)]
    base = [.001, .003, .01]
    if method == 'GrassRoute':
        return [dict(lr=lr, rho=rho) for lr, rho in itertools.product(base, [.3, 1., 3.])]
    if method == 'PeriodicDiag':
        return [dict(lr=lr, period=p) for lr, p in itertools.product(base, [5, 25, 100])]
    return [dict(lr=lr, residual_ratio=rat) for lr, rat in itertools.product(base, [.3, 1., 3.])]


def pinn_run(taskname, method, hp, epochs, path, formal):
    if (path/'summary.json').exists(): return json.loads((path/'summary.json').read_text())
    from tasks_extended import build
    from suite import groups
    seeded()
    task = build(taskname)
    gs, routed = groups(task, hp['lr'])
    if method == 'AdamW': opt = torch.optim.AdamW(task.model.parameters(), lr=hp['lr'], weight_decay=0.)
    else:
        kw = dict(lr=hp['lr'], grassmann_lr=hp.get('rho', 0.), reset_mode='diagonal',
                  lr_perp=hp['lr']*hp.get('residual_ratio', 1.))
        opt = PeriodicDiag(gs, svd_every=hp['period'], **kw) if method == 'PeriodicDiag' else GrassRoute(gs, **kw)
    # First random QR is initialized lazily in optimizer.step; isolate it from
    # subsequent collocation random streams, as in the formal extension.
    records = []
    start = time.monotonic()
    for ep in range(epochs):
        rate = hp['lr']*(.01+.99*.5*(1+math.cos(math.pi*ep/epochs)))
        for g in opt.param_groups:
            g['lr'] = rate
            if 'lr_perp' in g: g['lr_perp'] = rate*hp.get('residual_ratio', 1.)
        opt.zero_grad()
        loss = task.loss(task.model, None)
        if not torch.isfinite(loss):
            result = dict(status='diverged', epoch=ep+1, method=method, hp=hp, formal=formal)
            dump(path/'summary.json', result)
            return result
        loss.backward()
        rng = torch.get_rng_state()
        opt.step()
        torch.set_rng_state(rng)
        rec = dict(epoch=ep+1, train_loss=float(loss.detach()), wall_s=time.monotonic()-start)
        if (ep+1)%50 == 0 or ep == 0 or ep == epochs-1:
            rec['validation'] = task.evaluate(task.model, 'val')
        records.append(rec)
        if (ep+1)%100 == 0:
            dump(path/'trajectory.json', records)
    result = dict(status='complete', task=taskname, method=method, hp=hp, formal=formal,
                  epochs=epochs, n_epochs=len(records), seed=42, validation=task.evaluate(task.model, 'val'),
                  test=task.evaluate(task.model, 'test') if formal else None,
                  state_bytes=memory(opt), routed_layers=routed, wall_s=time.monotonic()-start)
    dump(path/'trajectory.json', records)
    dump(path/'summary.json', result)
    return result


def pinn():
    methods = ['AdamW', 'Fixed', 'PeriodicDiag', 'GrassRoute']
    protocol = register('pinn', dict(
        tasks=['sciml_poisson', 'sciml_burgers'], methods=methods,
        search={m:candidates(m) for m in methods}, screening_epochs=100, formal_epochs=2000,
        selection='minimum FINAL validation relative L2; held-out test evaluated once per selected formal run',
        paired_initialization=True, paired_collocation_stream=True, no_early_stopping=True,
        scope='Independent expanded-search PINN follow-up, separate from original 162 comparisons',
        rationale='Historical scientific-learning strength; two exact-solution tasks and transport-corrected periodic comparator'))
    results = []
    for task, method in itertools.product(protocol['tasks'], methods):
        folder = OUT/'pinn'/task/method
        screens = [pinn_run(task, method, hp, 100, folder/f'screen_{i}', False) for i,hp in enumerate(candidates(method))]
        good = [x for x in screens if x['status']=='complete' and math.isfinite(x['validation']['score'])]
        if not good:
            dump(folder/'failed.json', dict(reason='No finite validation candidate'))
            continue
        selected = min(good, key=lambda x:x['validation']['score'])
        dump(folder/'selection.json', dict(hp=selected['hp'], criterion=protocol['selection'],
                                            candidates=[dict(hp=x['hp'], status=x['status'], validation=x.get('validation')) for x in screens]))
        result = pinn_run(task, method, selected['hp'], 2000, folder/'formal', True)
        results.append(result)
        dump(OUT/'pinn_summary.json', dict(protocol=protocol, results=results))
        print(task, method, result, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['tracking', 'pinn'], required=True)
    a = ap.parse_args()
    tracking() if a.mode == 'tracking' else pinn()
