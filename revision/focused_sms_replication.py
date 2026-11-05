"""Three independent initialization seeds for the SMS state-quality result.

Hyperparameters are fixed from the original seed-42 validation searches.
The same public train/validation/test split and 1000-step budget are retained.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from new_tasks_suite import Task, optimizer, state_bytes, atomic, DATA, HERE

OUT = HERE / 'focused_results' / 'sms_replication'
METHODS = ['SGD', 'AdamW', 'PeriodicThinSVD', 'GrassRoute']
SEEDS = [43, 44, 45]
torch.set_num_threads(2)


def selected():
    root = HERE / 'newtask_results' / 'text_sms'
    hp, hashes = {}, {}
    for method in METHODS:
        p = root / method / 'selection.json'
        hp[method] = json.loads(p.read_text())['hp']
        hashes[method] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hp, hashes


def run(seed, method, hp):
    folder = OUT / f'seed_{seed}' / method
    done = folder / 'summary.json'
    if done.exists():
        return json.loads(done.read_text())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    task = Task('text_sms')
    opt, layers = optimizer(task, method, hp)
    records = []
    start = time.monotonic()
    for ep in range(1000):
        task.model.train()
        rate = hp['lr'] * (.01 + .99 * .5 * (1 + math.cos(math.pi * ep / 1000)))
        for group in opt.param_groups:
            group['lr'] = rate
            if 'lr_perp' in group:
                group['lr_perp'] = rate * hp.get('residual_ratio', 1.)
        opt.zero_grad()
        loss = task.loss()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at epoch {ep + 1}')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in task.model.parameters()):
            raise FloatingPointError(f'non-finite gradient at epoch {ep + 1}')
        rng = torch.get_rng_state()
        opt.step()
        torch.set_rng_state(rng)
        rec = dict(epoch=ep + 1, train_loss=float(loss.detach()))
        if ep == 0 or (ep + 1) % 50 == 0:
            rec['validation'] = task.evaluate('val')
        records.append(rec)
    result = dict(task='text_sms', method=method, seed=seed,
                  status='complete', formal=True, epochs=1000,
                  n_epochs=len(records), hp=hp,
                  validation=task.evaluate('val'), test=task.evaluate('test'),
                  state_bytes=state_bytes(opt), routed_layers=layers,
                  wall_s=time.monotonic() - start,
                  protocol_sha256=hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest())
    atomic(folder / 'trajectory.json', records)
    atomic(done, result)
    return result


def main():
    hp, selection_hashes = selected()
    protocol = dict(task='text_sms', methods=METHODS, seeds=SEEDS,
                    data_meta_sha256=hashlib.sha256((DATA/'text_sms'/'meta.json').read_bytes()).hexdigest(),
                    original_seed42_selection_sha256=selection_hashes,
                    fixed_hyperparameters=hp,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    formal_epochs=1000, selection='none on new seeds; seed-42 choices frozen',
                    primary_metric='final held-out macro-F1',
                    secondary_metric='test cross-entropy, descriptive only',
                    paired_split=True, paired_initialization=True, paired_dropout=True,
                    no_early_stopping=True,
                    retention='all four methods and all three seeds')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / 'protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError('Frozen SMS replication protocol changed')
    else:
        atomic(frozen, protocol)
    for seed in SEEDS:
        for method in METHODS:
            result = run(seed, method, hp[method])
            print(seed, method, result['test']['score'],
                  'seconds', round(result['wall_s'], 1), flush=True)


if __name__ == '__main__':
    main()
