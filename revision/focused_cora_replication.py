"""Independent seeds for the prior favorable Cora GraphSAGE comparison.

All four hyperparameters are frozen from the seed-42 validation search before
seeds 43--45 run. The public Planetoid split is held fixed. No new test-based
selection is performed, and every endpoint is retained.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from strength_study import PeriodicDiag, GrassRoute, HERE, dump, memory
from suite import groups
from tasks_extended import build

OUT = HERE / 'focused_results' / 'cora_sage_replication'
METHODS = ['SGD', 'AdamW', 'PeriodicDiag', 'GrassRoute']
SEEDS = [43, 44, 45]
torch.set_num_threads(2)


def selected():
    root = HERE / 'strength_results' / 'graph' / 'graph_cora'
    result = {}
    hashes = {}
    for method in METHODS:
        p = root / method / 'selection.json'
        result[method] = json.loads(p.read_text())['hp']
        hashes[method] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result, hashes


def run(seed, method, hp):
    folder = OUT / f'seed_{seed}' / method
    done = folder / 'summary.json'
    if done.exists():
        return json.loads(done.read_text())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    task = build('graph_cora')
    gs, routed = groups(task, hp['lr'])
    if method == 'SGD':
        opt = torch.optim.SGD(task.model.parameters(), lr=hp['lr'], momentum=.9)
    elif method == 'AdamW':
        opt = torch.optim.AdamW(task.model.parameters(), lr=hp['lr'], weight_decay=0.)
    elif method == 'PeriodicDiag':
        opt = PeriodicDiag(gs, lr=hp['lr'], svd_every=hp['period'],
                           reset_mode='diagonal')
    else:
        opt = GrassRoute(gs, lr=hp['lr'], grassmann_lr=hp['rho'],
                         reset_mode='diagonal')
    records = []
    start = time.monotonic()
    for ep in range(1000):
        task.model.train()
        rate = hp['lr'] * (.01 + .99 * .5 * (1 + math.cos(math.pi * ep / 1000)))
        for group in opt.param_groups:
            group['lr'] = rate
            if 'lr_perp' in group:
                group['lr_perp'] = rate
        opt.zero_grad()
        loss = task.loss(task.model, None)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite objective at epoch {ep + 1}')
        loss.backward()
        rng = torch.get_rng_state()
        opt.step()
        torch.set_rng_state(rng)
        rec = dict(epoch=ep + 1, train_loss=float(loss.detach()))
        if ep == 0 or (ep + 1) % 50 == 0:
            rec['validation'] = task.evaluate(task.model, 'val')
        records.append(rec)
    result = dict(task='graph_cora', model='two-layer GraphSAGE', method=method,
                  seed=seed, status='complete', formal=True, epochs=1000,
                  n_epochs=len(records), hp=hp,
                  validation=task.evaluate(task.model, 'val'),
                  test=task.evaluate(task.model, 'test'), state_bytes=memory(opt),
                  routed_layers=routed, wall_s=time.monotonic() - start,
                  protocol_sha256=hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest())
    dump(folder / 'trajectory.json', records)
    dump(done, result)
    return result


def main():
    hp, selection_hashes = selected()
    protocol = dict(task='graph_cora', model='two-layer GraphSAGE', methods=METHODS,
                    seeds=SEEDS, fixed_hyperparameters=hp,
                    original_seed42_selection_sha256=selection_hashes,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    formal_epochs=1000, selection='none on new seeds; seed-42 choices frozen',
                    metric='final public-test accuracy; descriptive mean and sample SD across new seeds',
                    paired_initialization=True, paired_dropout=True,
                    split='same public transductive Planetoid split',
                    device='cpu', no_early_stopping=True,
                    retention='all four methods and all three seeds')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / 'protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError('Frozen Cora replication protocol changed')
    else:
        dump(frozen, protocol)
    for seed in SEEDS:
        for method in METHODS:
            result = run(seed, method, hp[method])
            print(seed, method, result['test']['score'],
                  'seconds', round(result['wall_s'], 1), flush=True)


if __name__ == '__main__':
    main()
