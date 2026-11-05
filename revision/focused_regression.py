"""Frozen, validation-selected dynamic low-rank regression follow-up.

This is a controlled nonstationary optimization task, separate from the
original 162-run proposal suite. Every condition and comparator is retained.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'experiments'))
from grassroute import GrassRoute
from new_tasks_suite import StaticPeriodic
from suite import state_bytes

OUT = HERE / 'focused_results'
METHODS = ['SGD', 'AdamW', 'Fixed', 'PeriodicThinSVD', 'GrassRoute']
CONDITIONS = {'stationary': 0.0, 'drift005': 0.005, 'drift020': 0.02}
torch.set_num_threads(1)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def candidates(method):
    if method == 'SGD':
        return [dict(lr=x) for x in (.03, .1, .3, 1.)]
    if method == 'AdamW':
        return [dict(lr=x) for x in (.001, .003, .01, .03)]
    base = [dict(lr=.003, ratio=10.), dict(lr=.003, ratio=100.),
            dict(lr=.01, ratio=10.), dict(lr=.01, ratio=100.)]
    if method == 'PeriodicThinSVD':
        return [dict(x, period=p) for x, p in zip(base, (5, 25, 25, 5))]
    if method == 'GrassRoute':
        return [dict(x, rho=r) for x, r in zip(base, (.3, 3., 3., .3))]
    return base


def teacher(step, drift, amplitude):
    theta = drift * step
    frame = torch.zeros(64, 4)
    frame[:4] = math.cos(theta) * torch.eye(4)
    frame[4:8] = math.sin(theta) * torch.eye(4)
    return amplitude @ frame.T, frame


def optimizer(method, model, hp):
    if method == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=hp['lr'], momentum=.9)
    if method == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=hp['lr'])
    groups = [dict(params=[model.weight], routed=True, rank=4)]
    kw = dict(lr=hp['lr'], lr_perp=hp['lr'] * hp['ratio'],
              reset_mode='diagonal', weight_decay=0.)
    if method == 'GrassRoute':
        return GrassRoute(groups, grassmann_lr=hp['rho'], **kw)
    return StaticPeriodic(groups, period=hp.get('period'), **kw)


def run(condition, method, hp, steps, folder, formal):
    complete = folder / 'summary.json'
    if complete.exists():
        return json.loads(complete.read_text())
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    amp_gen = torch.Generator().manual_seed(4201)
    amplitude = torch.randn(16, 4, generator=amp_gen) / 2
    stream = torch.Generator().manual_seed(4203 if formal else 4202)
    val_x = torch.randn(256, 64, generator=torch.Generator().manual_seed(4301))
    test_x = (torch.randn(512, 64, generator=torch.Generator().manual_seed(4302))
              if formal else None)
    model = nn.Linear(64, 16, bias=False)
    nn.init.zeros_(model.weight)
    opt = optimizer(method, model, hp)
    drift = CONDITIONS[condition]
    records = []
    start = time.monotonic()
    for i in range(steps):
        target, frame = teacher(i, drift, amplitude)
        x = torch.randn(128, 64, generator=stream)
        noise = .05 * torch.randn(128, 16, generator=stream)
        y = x @ target.T + noise
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {i + 1}')
        loss.backward()
        opt.step()
        record = dict(step=i + 1, train_mse=float(loss.detach()))
        if (i + 1) % 10 == 0 or i == 0:
            with torch.no_grad():
                record['validation_mse'] = float(F.mse_loss(model(val_x), val_x @ target.T))
                if formal:
                    record['test_mse'] = float(F.mse_loss(model(test_x), test_x @ target.T))
                basis = opt.state[model.weight].get('U')
                if basis is not None:
                    record['subspace_sin2'] = max(0., 1. - float((basis.T @ frame).square().sum()) / 4)
        records.append(record)
    tail = [x for x in records if x['step'] > .8 * steps and 'validation_mse' in x]
    result = dict(condition=condition, drift=drift, method=method, hp=hp,
                  status='complete', formal=formal, steps=steps, seed=42,
                  selection_score=float(np.mean([x['validation_mse'] for x in tail])),
                  test_score=(float(np.mean([x['test_mse'] for x in tail])) if formal else None),
                  mean_subspace_sin2=(float(np.mean([x['subspace_sin2'] for x in tail
                                                    if 'subspace_sin2' in x]))
                                      if any('subspace_sin2' in x for x in tail) else None),
                  state_bytes=state_bytes(opt), wall_s=time.monotonic() - start,
                  trajectory_length=len(records))
    save(folder / 'trajectory.json', records)
    save(complete, result)
    return result


def protocol():
    return dict(scope='Controlled time-varying low-rank regression, not public-data task family',
                conditions=CONDITIONS, methods=METHODS, shape=[16, 64], rank=4,
                teacher='W_t=A U_t^T, paired four-plane rotation by drift*t; A fixed by seed 4201',
                train='128 fresh Gaussian inputs and Gaussian label noise sd 0.05 per step',
                validation='256 fixed independent Gaussian inputs, seed 4301',
                test='512 fixed independent Gaussian inputs, seed 4302; formal only',
                screening_steps=300, formal_steps=1500,
                candidates={m:candidates(m) for m in METHODS},
                selection='minimum mean validation MSE over final 20 percent of screen',
                formal_metric='mean independent test MSE over final 20 percent of steps',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                comparator='direct thin SVD, transported first and diagonal second moment',
                rank_policy='rank four in d=64, fixed across routed methods',
                retention='all conditions, methods, screen outcomes and complete trajectories',
                seed=42, no_early_stopping=True)


def main(conditions):
    OUT.mkdir(exist_ok=True)
    proto = protocol()
    existing = OUT / 'protocol.json'
    if existing.exists():
        if json.loads(existing.read_text()) != proto:
            raise RuntimeError('Frozen protocol differs from this source; preserve old results')
    else:
        save(existing, proto)
    for condition, method in itertools.product(conditions, METHODS):
        folder = OUT / condition / method
        screens = []
        for i, hp in enumerate(candidates(method)):
            try:
                screens.append(run(condition, method, hp, 300, folder / f'screen_{i}', False))
            except (FloatingPointError, RuntimeError) as e:
                save(folder / f'screen_{i}' / 'failed.json',
                     dict(method=method, hp=hp, error=str(e)))
        if not screens:
            save(folder / 'failed.json', dict(reason='All validation candidates failed'))
            continue
        selected = min(screens, key=lambda x: x['selection_score'])
        save(folder / 'selection.json', dict(method=method, hp=selected['hp'],
             criterion=proto['selection'], candidates=[dict(hp=x['hp'],
             validation_mse=x['selection_score']) for x in screens]))
        result = run(condition, method, selected['hp'], 1500, folder / 'formal', True)
        print(condition, method, 'test', result['test_score'], 'state', result['state_bytes'],
              'seconds', round(result['wall_s'], 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS),
                        default=list(CONDITIONS))
    args = parser.parse_args()
    main(args.conditions)
