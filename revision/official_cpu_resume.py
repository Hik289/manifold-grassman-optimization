"""Device-only adapter for pinned SubTrack++; resume failed CPU comparisons.

The upstream source is not written. AST replacements change six hardcoded
CUDA allocation targets to the current gradient device, and no arithmetic.
"""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import time

import strength_graph_rescue as recovery

original=recovery.original
BaseOfficial=original.Official


def install_device_adapter():
    module=importlib.import_module('low_rank_torch.low_rank_projector')
    cls=module.LowRankProjector
    if getattr(cls,'_grassroute_device_adapter',False):return
    changed=0
    class Target(ast.NodeTransformer):
        def visit_Constant(self,node):
            nonlocal changed
            if node.value=='cuda':
                changed+=1
                return ast.copy_location(ast.Attribute(value=ast.Name(id='full_rank_grad',ctx=ast.Load()),
                                                       attr='device',ctx=ast.Load()),node)
            return node
    for name in ['subtrack_projector','track_the_subspace']:
        source=textwrap.dedent(inspect.getsource(getattr(cls,name)))
        tree=ast.fix_missing_locations(Target().visit(ast.parse(source)))
        namespace=dict(module.__dict__)
        exec(compile(tree,f'<device-only adapter: {name}>','exec'),namespace)
        setattr(cls,name,namespace[name])
    assert changed==6, f'Unexpected upstream allocation count: {changed}'
    cls._grassroute_device_adapter=True


class DeviceOfficial(BaseOfficial):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        install_device_adapter()


def main(tasks,wait_pid=None):
    if wait_pid:
        for _ in range(360):
            cmd=Path(f'/proc/{wait_pid}/cmdline')
            if not cmd.exists():break
            value=cmd.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'strength_graph_rescue.py' not in value:break
            time.sleep(5)
        else:raise RuntimeError('Existing recovery worker still active; avoid duplicate task execution')
    upstream=original.HERE/'third_party/SubTrack/low_rank_torch/low_rank_projector.py'
    manifest=dict(adapter='six CUDA allocation targets -> full_rank_grad.device; AST-only runtime patch',
                  upstream_source_modified=False,optimizer_arithmetic_modified=False,
                  upstream_sha256=hashlib.sha256(upstream.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tasks=['graph_cora','graph_citeseer','graph_pubmed'],
                  recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    recovery.atomic(original.OUT/'graph_cpu_device_adapter.json',manifest)
    original.Official=DeviceOfficial
    original.METHODS=['SubTrackOfficial']
    recovery.main(tasks)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',required=True)
    ap.add_argument('--wait-pid',type=int)
    a=ap.parse_args();main(a.tasks,a.wait_pid)
