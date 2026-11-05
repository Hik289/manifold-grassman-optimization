"""Materialize and validate requested external datasets before queued training."""
import gc
import json
from pathlib import Path
import time
import traceback
from tasks_extended import build, EXTRA

here=Path(__file__).resolve().parent
names=['graph_citeseer','graph_pubmed','recsys_amazonbooks','recsys_yelp',
       'cnn_tinyimagenet','cnn_imagenet100','vit_cifar100','continual_splitcifar100']
out={}
for name in names:
    start=time.time()
    try:
        task=build(name)
        batch=next(iter(task.train))
        out[name]={'status':'accessible','meta':task.meta,
                   'batch_shape':list(batch[0].shape) if batch is not None else None,
                   'seconds':time.time()-start}
        del task,batch;gc.collect()
    except Exception as e:
        out[name]={'status':'failed','error':str(e),'traceback':traceback.format_exc()}
    path=here/'data_access.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
    print(name,out[name]['status'],flush=True)
