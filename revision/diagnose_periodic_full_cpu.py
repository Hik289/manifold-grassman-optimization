"""Capture the actual numerical-failure tensors without changing updates."""
import hashlib
import json
from pathlib import Path

import torch
import strength_graph as original


def stats(tensor):
    if not torch.is_tensor(tensor):return str(type(tensor))
    x=tensor.detach()
    finite=torch.isfinite(x)
    return dict(shape=list(x.shape),dtype=str(x.dtype),finite_fraction=float(finite.float().mean()),
                norm=float(x.double().norm()) if bool(finite.all()) else None,
                max_abs=float(x.abs().max()) if bool(finite.all()) else None)


if __name__=='__main__':
    path=original.OUT/'periodic_full_failure_diagnostic'
    original.dump(path/'protocol.json',dict(task='graph_citeseer',hp=dict(lr=.003,period=25),
                  purpose='Full-horizon failure diagnosis only; no test evaluation or candidate selection',
                  optimizer_update_changed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        result=original.run('graph_citeseer','PeriodicDiag',dict(lr=.003,period=25),1000,path/'run',False)
        original.dump(path/'diagnostic.json',dict(status='unexpectedly_completed',result=result))
    except RuntimeError as error:
        frames=[];tb=error.__traceback__
        while tb:
            loc=tb.tb_frame.f_locals
            entry=dict(function=tb.tb_frame.f_code.co_name,line=tb.tb_lineno,tensors={})
            for name in ['G','C','U','U_new','M','V','Z','G_perp','D_par','Rp','p','loss']:
                if name in loc:entry['tensors'][name]=stats(loc[name])
            if 'ep' in loc:entry['epoch']=int(loc['ep'])+1
            if 'group' in loc:
                entry['hyperparameters']={k:v for k,v in loc['group'].items() if k!='params' and isinstance(v,(int,float,str,bool))}
            if entry['tensors']:frames.append(entry)
            tb=tb.tb_next
        result=dict(error=str(error),frames=frames)
        original.dump(path/'diagnostic.json',result)
        print(json.dumps(result,indent=2),flush=True)
