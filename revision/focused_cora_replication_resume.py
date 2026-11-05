"""Finish the frozen Cora seed grid while retaining numerical failures.

The optimizer and hyperparameters remain unchanged. A non-finite candidate
is recorded as a failure; remaining prespecified cells continue.
"""
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

import focused_cora_replication as original
from strength_study import dump


def main():
    hp, _ = original.selected()
    frozen = original.OUT / 'protocol.json'
    if not frozen.exists():
        raise RuntimeError('Original frozen protocol missing')
    protocol = json.loads(frozen.read_text())
    if hp != protocol['fixed_hyperparameters']:
        raise RuntimeError('Hyperparameters differ from frozen protocol')
    dump(original.OUT / 'numerical_recovery_protocol.json',
         dict(original_protocol_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
              original_source_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              rule='record numerical failure and continue remaining fixed cells; no replacement or retuning',
              recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    for seed in original.SEEDS:
        for method in original.METHODS:
            folder = original.OUT / f'seed_{seed}' / method
            if (folder / 'summary.json').exists() or (folder / 'failed.json').exists():
                continue
            try:
                x = original.run(seed, method, hp[method])
                print(seed, method, x['test']['score'], flush=True)
            except (FloatingPointError, torch.linalg.LinAlgError, RuntimeError) as error:
                msg = str(error).lower()
                if not any(term in msg for term in ('non-finite', 'nonfinite', 'nan',
                                                    'linalg.svd', 'linalg.eigh')):
                    raise
                dump(folder / 'failed.json',
                     dict(seed=seed, method=method, hp=hp[method],
                          status='numerical_failure', error=str(error),
                          traceback=traceback.format_exc(),
                          test=None, recovery='numerical_recovery_protocol.json'))
                print(seed, method, 'numerical failure retained', flush=True)


if __name__ == '__main__':
    main()
