import os
"""Independent Cora GCN architecture check of the completed SAGE result.

The search grid, six methods, validation rule, and 1000-epoch formal budget
match strength_graph.py. All attempted candidates and endpoints are retained.
"""
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

import strength_graph as parent
from strength_study import HERE, dump

OUT = HERE / 'focused_results' / 'cora_gcn'
METHODS = parent.METHODS
torch.set_num_threads(2)


def build_gcn(name):
    assert name == 'graph_cora_gcn'
    ds = Planetoid(str(Path(os.environ.get('GR_DATA_ROOT', HERE.parent / 'data')) / 'planetoid'),
                   'Cora', transform=T.NormalizeFeatures())
    data = ds[0]

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GCNConv(ds.num_features, 128)
            self.c2 = GCNConv(128, ds.num_classes)

        def forward(self):
            z = F.relu(self.c1(data.x, data.edge_index))
            z = F.dropout(z, p=.5, training=self.training)
            return self.c2(z, data.edge_index)

    model = GCN()

    def loss(model, _):
        logits = model()
        return F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

    def evaluate(model, split):
        model.eval()
        with torch.no_grad():
            pred = model().argmax(1)
            mask = data.val_mask if split == 'val' else data.test_mask
            score = float((pred[mask] == data.y[mask]).float().mean())
        model.train()
        return {'score': score}

    class Task:
        family = 'graph'
        meta = {'dataset': 'Cora', 'model': 'two-layer GCN', 'hidden': 128,
                'split': 'public Planetoid', 'dropout': .5}

        def __init__(self):
            self.model = model
            self.loss = loss
            self.evaluate = evaluate

    return Task()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(task='graph_cora_gcn', dataset='Cora', model='two-layer GCN',
                    methods=METHODS, search={m:parent.candidates(m) for m in METHODS},
                    screening_epochs=100, formal_epochs=1000, seed=42,
                    selection='maximum final validation accuracy; fresh formal initialization',
                    paired_initialization=True, paired_dropout=True, device='cpu',
                    no_early_stopping=True, retention='all candidates, failures and formal endpoints',
                    scope='Architecture replication of prior Cora GraphSAGE result; transductive',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    parent_sha256=hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest())
    frozen = OUT / 'protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError('Focused GCN protocol changed after registration')
    else:
        dump(frozen, protocol)
    parent.build = build_gcn
    for method in METHODS:
        folder = OUT / method
        screens = [parent.run('graph_cora_gcn', method, hp, 100,
                              folder / f'screen_{i}', False)
                   for i, hp in enumerate(parent.candidates(method))]
        valid = [x for x in screens if x['status'] == 'complete']
        if not valid:
            dump(folder / 'failed.json', {'reason': 'All validation candidates diverged'})
            continue
        chosen = max(valid, key=lambda x: x['validation']['score'])
        dump(folder / 'selection.json',
             {'hp': chosen['hp'], 'criterion': protocol['selection'],
              'candidates': [{'hp': x['hp'], 'status': x['status'],
                              'validation': x.get('validation')} for x in screens]})
        result = parent.run('graph_cora_gcn', method, chosen['hp'], 1000,
                            folder / 'formal', True)
        print(method, result['status'], result.get('test'),
              'wall_s', round(result.get('wall_s', 0.), 1), flush=True)


if __name__ == '__main__':
    main()
