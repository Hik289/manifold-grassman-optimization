"""Complete the pinned official comparator on CPU after its device error.

Only upstream hardcoded CUDA allocation targets are adapted at runtime;
candidate search, optimizer arithmetic and the frozen GCN protocol remain fixed.
"""
import hashlib
from pathlib import Path
import time

import focused_cora_gcn as focused
import strength_graph as parent
from official_cpu_resume import DeviceOfficial
from strength_study import HERE, dump


def main():
    parent.build = focused.build_gcn
    parent.Official = DeviceOfficial
    folder = focused.OUT / 'SubTrackOfficial'
    source = HERE / 'third_party/SubTrack/low_rank_torch/low_rank_projector.py'
    dump(focused.OUT / 'device_recovery.json',
         dict(reason='Pinned upstream projector uses six hardcoded CUDA allocations on CPU',
              repair='Existing AST device-only adapter; no arithmetic or search-grid change',
              upstream_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              adapter_sha256=hashlib.sha256((HERE / 'official_cpu_resume.py').read_bytes()).hexdigest(),
              recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    screens = [parent.run('graph_cora_gcn', 'SubTrackOfficial', hp, 100,
                          folder / f'screen_{i}', False)
               for i, hp in enumerate(parent.candidates('SubTrackOfficial'))]
    valid = [x for x in screens if x['status'] == 'complete']
    if not valid:
        dump(folder / 'failed.json', {'reason': 'No validation candidate completed'})
        return
    chosen = max(valid, key=lambda x: x['validation']['score'])
    dump(folder / 'selection.json',
         dict(hp=chosen['hp'], criterion='maximum final validation accuracy',
              candidates=[dict(hp=x['hp'], status=x['status'],
                               validation=x.get('validation')) for x in screens],
              recovery='device_recovery.json; no test-driven selection'))
    result = parent.run('graph_cora_gcn', 'SubTrackOfficial', chosen['hp'], 1000,
                        folder / 'formal', True)
    print('SubTrackOfficial', result['status'], result.get('test'), flush=True)


if __name__ == '__main__':
    main()
