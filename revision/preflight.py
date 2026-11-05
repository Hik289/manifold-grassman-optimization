"""Exercise one real training batch for each task and all six methods."""
import gc
import json
from pathlib import Path
import time
import traceback
import torch
from suite import optimizer,seed
from tasks_extended import build,TASKS

here=Path(__file__).resolve().parent
out={}
for name in TASKS:
    if name.startswith(('matreg_','lowrank_','sciml_')):continue
    try:
        seed();t=build(name);batch=next(iter(t.train))
        initial={k:v.detach().clone() for k,v in t.model.state_dict().items()}
        out[name]={}
        for method in ['SGD','AdamW','Fixed','PeriodicSVD','GrassRoute','SubTrackOfficial']:
            start=time.time()
            try:
                t.model.load_state_dict(initial);t.model.train()
                o,names=optimizer(t,method,.001,.3)
                o.zero_grad();loss=t.loss(t.model,batch);assert torch.isfinite(loss)
                loss.backward();o.step()
                assert all(torch.isfinite(p).all() for p in t.model.parameters())
                out[name][method]={'status':'passed','loss':float(loss),'seconds':time.time()-start}
                del o
            except Exception as e:out[name][method]={'status':'failed','error':str(e),'traceback':traceback.format_exc()}
            print(name,method,out[name][method]['status'],flush=True)
        del t,batch,initial;gc.collect();torch.cuda.empty_cache()
    except Exception as e:out[name]={'build_error':str(e),'traceback':traceback.format_exc()}
    tmp=here/'preflight_results.tmp';tmp.write_text(json.dumps(out,indent=2));tmp.replace(here/'preflight_results.json')
