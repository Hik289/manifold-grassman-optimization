import os
from pathlib import Path
"""
7 任务族的 model + data 工厂 (骨架, 每族一个代表数据集/配置)。
每个 build_<family>() 返回:
  model, train_step(model, batch)->loss, eval_fn(model)->dict(metric), data_iter/loader, meta

数据集就绪状态见 EDA §1 (hpc 上已验证可加载)。大数据留 hpc。
骨架目标: 每族能实例化 + 跑几步 smoke test, 供 RUNNING 阶段填满长训练。

任务族: matreg / mlp / cnn / vit / graph / sciml / recsys / continual
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_ROOT = os.environ.get("GRASSROUTE_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))


# ============ 1. Matrix Regression (合成, 已在 anchor1) ============
def build_matreg(d=256, m=64, r_true=16, N=2048):
    from anchor1_matreg import make_data
    X, Y, W_star, U_star = make_data(d=d, m=m, r_true=r_true, N=N)

    class LinReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(0.01 * torch.randn(m, d))

        def forward(self, x):
            return x @ self.W.T
    model = LinReg().to(DEV)

    def train_step(model, _):
        pred = model(X)
        return F.mse_loss(pred, Y)

    def eval_fn(model):
        with torch.no_grad():
            return {"mse": F.mse_loss(model(X), Y).item()}
    return model, train_step, eval_fn, [None], {"family": "matreg", "epochs": 2000, "d": d}


# ============ 2. Deep MLP (MNIST) ============
def build_mlp():
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    tl = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=2)
    vl = torch.utils.data.DataLoader(test, batch_size=512, num_workers=2)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(784, 512); self.f2 = nn.Linear(512, 512)
            self.f3 = nn.Linear(512, 256); self.head = nn.Linear(256, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.f1(x)); x = F.relu(self.f2(x)); x = F.relu(self.f3(x))
            return self.head(x)
    model = MLP().to(DEV)

    def train_step(model, batch):
        x, y = batch
        return F.cross_entropy(model(x.to(DEV)), y.to(DEV))

    def eval_fn(model):
        model.eval(); correct = 0; n = 0
        with torch.no_grad():
            for x, y in vl:
                p = model(x.to(DEV)).argmax(1).cpu()
                correct += (p == y).sum().item(); n += y.size(0)
        model.train()
        return {"acc": correct / n}
    return model, train_step, eval_fn, tl, {"family": "mlp", "epochs": 2000}


# ============ 3. CNN (CIFAR-100 默认, ResNet-18) ============
def build_cnn(num_classes=100, dataset="CIFAR100"):
    import torchvision
    import torchvision.transforms as T
    import torchvision.models as M
    tf_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    tf_te = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    DS = getattr(torchvision.datasets, dataset)
    train = DS(DATA_ROOT, train=True, download=True, transform=tf_tr)
    test = DS(DATA_ROOT, train=False, download=True, transform=tf_te)
    tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
    vl = torch.utils.data.DataLoader(test, batch_size=512, num_workers=2)
    model = M.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # CIFAR 适配
    model.maxpool = nn.Identity()
    model = model.to(DEV)

    def train_step(model, batch):
        x, y = batch
        return F.cross_entropy(model(x.to(DEV)), y.to(DEV))

    def eval_fn(model):
        model.eval(); correct = 0; n = 0
        with torch.no_grad():
            for x, y in vl:
                p = model(x.to(DEV)).argmax(1).cpu()
                correct += (p == y).sum().item(); n += y.size(0)
        model.train()
        return {"top1": correct / n}
    return model, train_step, eval_fn, tl, {"family": "cnn", "epochs": 1000}


# ============ 4. ViT (CIFAR-100, ViT-Tiny patch4 img32) ============
def build_vit(num_classes=100):
    import timm
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    train = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=tf)
    tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
    vl = torch.utils.data.DataLoader(test, batch_size=512, num_workers=2)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=num_classes, img_size=32, patch_size=4).to(DEV)

    def train_step(model, batch):
        x, y = batch
        return F.cross_entropy(model(x.to(DEV)), y.to(DEV))

    def eval_fn(model):
        model.eval(); correct = 0; n = 0
        with torch.no_grad():
            for x, y in vl:
                p = model(x.to(DEV)).argmax(1).cpu()
                correct += (p == y).sum().item(); n += y.size(0)
        model.train()
        return {"top1": correct / n}
    return model, train_step, eval_fn, tl, {"family": "vit", "epochs": 1000}


# ============ 5. Graph (ogbn-arxiv 主表 / Cora, GraphSAGE) ============
def build_graph(dataset="ogbn-arxiv"):
    import torch_geometric.transforms as T
    from torch_geometric.nn import SAGEConv
    if dataset.startswith("ogbn"):
        # ogb 1.3.6 与 torch>=2.6 的 weights_only 默认值不兼容 (EDA §1 已记录), patch torch.load
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            ds = PygNodePropPredDataset(name=dataset, root=DATA_ROOT + "/ogb")
        finally:
            torch.load = _orig_load
        data = ds[0]
        split = ds.get_idx_split()
        data.y = data.y.squeeze()
        for k, idx in split.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            setattr(data, {"train": "train_mask", "valid": "val_mask", "test": "test_mask"}[k], mask)
        data = data.to(DEV)
        num_features, num_classes = ds.num_features, int(data.y.max()) + 1
    else:
        from torch_geometric.datasets import Planetoid
        ds = Planetoid(DATA_ROOT + "/planetoid", dataset, transform=T.NormalizeFeatures())
        data = ds[0].to(DEV)
        num_features, num_classes = ds.num_features, ds.num_classes

    class SAGE(nn.Module):
        def __init__(self, ind, hid, outd):
            super().__init__()
            self.c1 = SAGEConv(ind, hid); self.c2 = SAGEConv(hid, outd)

        def forward(self, x, ei):
            x = F.relu(self.c1(x, ei)); x = F.dropout(x, 0.5, self.training)
            return self.c2(x, ei)
    model = SAGE(num_features, 128, num_classes).to(DEV)

    def train_step(model, _):
        out = model(data.x, data.edge_index)
        return F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index).argmax(1)
            accs = {}
            for split_name in ["val", "test"]:
                mask = getattr(data, f"{split_name}_mask")
                accs[split_name] = (out[mask] == data.y[mask]).float().mean().item()
        model.train()
        return {"val": accs["val"], "test": accs["test"]}
    return model, train_step, eval_fn, [None], {"family": "graph", "epochs": 1000}


# ============ 6. SciML (Burgers PINN) ============
def build_sciml():
    nu = 0.01 / math.pi

    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 128); self.l2 = nn.Linear(128, 128)
            self.l3 = nn.Linear(128, 128); self.l4 = nn.Linear(128, 128)
            self.out = nn.Linear(128, 1)

        def forward(self, xt):
            h = torch.tanh(self.l1(xt)); h = torch.tanh(self.l2(h))
            h = torch.tanh(self.l3(h)); h = torch.tanh(self.l4(h))
            return self.out(h)
    model = PINN().to(DEV)
    g = torch.Generator().manual_seed(SEED)

    def sample_colloc(n=2048):
        x = (torch.rand(n, 1, generator=g) * 2 - 1)
        t = torch.rand(n, 1, generator=g)
        return x.to(DEV), t.to(DEV)

    def train_step(model, _):
        x, t = sample_colloc()
        x.requires_grad_(True); t.requires_grad_(True)
        u = model(torch.cat([x, t], 1))
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        res = u_t + u * u_x - nu * u_xx
        # IC/BC
        x0 = (torch.rand(256, 1, generator=g) * 2 - 1).to(DEV)
        u0 = model(torch.cat([x0, torch.zeros_like(x0)], 1))
        ic = u0 + torch.sin(math.pi * x0)
        return res.pow(2).mean() + ic.pow(2).mean()

    # 固定评估网格 (格点 PDE residual L2 + IC L2), Burgers ν=0.01/π 无闭式解,
    # 用 PDE residual + IC 误差作为标准 PINN 报告指标
    xe = torch.linspace(-1, 1, 128, device=DEV).reshape(-1, 1)
    te = torch.linspace(0, 1, 64, device=DEV).reshape(-1, 1)
    Xg, Tg = torch.meshgrid(xe.squeeze(), te.squeeze(), indexing="ij")
    grid = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], 1)

    def eval_fn(model):
        gx = grid[:, 0:1].clone().requires_grad_(True)
        gt = grid[:, 1:2].clone().requires_grad_(True)
        u = model(torch.cat([gx, gt], 1))
        u_x = torch.autograd.grad(u, gx, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, gt, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, gx, torch.ones_like(u_x), create_graph=True)[0]
        res = (u_t + u * u_x - nu * u_xx)
        with torch.no_grad():
            x0 = torch.linspace(-1, 1, 256, device=DEV).reshape(-1, 1)
            u0 = model(torch.cat([x0, torch.zeros_like(x0)], 1))
            ic_err = (u0 + torch.sin(math.pi * x0)).pow(2).mean().sqrt().item()
            pde_res_l2 = res.detach().pow(2).mean().sqrt().item()
        return {"pde_residual_l2": pde_res_l2, "ic_l2": ic_err,
                "rel_l2_proxy": pde_res_l2 + ic_err}
    return model, train_step, eval_fn, [None], {"family": "sciml", "epochs": 2000}


# ============ 7. RecSys (MovieLens-1M, MF) ============
def build_recsys(latent=64):
    import os
    import numpy as np
    path = DATA_ROOT + "/ml-1m/ratings.dat"
    if not os.path.exists(path):
        return None  # 需先下载, RUNNING 前准备
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            u, i, r, _ = line.strip().split("::")
            rows.append((int(u), int(i), float(r)))
    arr = torch.tensor(rows)
    nu = int(arr[:, 0].max()) + 1; ni = int(arr[:, 1].max()) + 1
    idx = torch.randperm(len(arr), generator=torch.Generator().manual_seed(SEED))
    split = int(0.9 * len(arr))
    tr, te = arr[idx[:split]], arr[idx[split:]]

    global_mean = float(tr[:, 2].mean())

    class MF(nn.Module):
        def __init__(self):
            super().__init__()
            self.uemb = nn.Embedding(nu, latent); self.iemb = nn.Embedding(ni, latent)
            self.ub = nn.Embedding(nu, 1); self.ib = nn.Embedding(ni, 1)
            # 小初始化 (latent=64 dot product 否则 var~64 致发散) + bias 置0 + 全局均值偏置
            nn.init.normal_(self.uemb.weight, std=0.01); nn.init.normal_(self.iemb.weight, std=0.01)
            nn.init.zeros_(self.ub.weight); nn.init.zeros_(self.ib.weight)
            self.mu = global_mean

        def forward(self, u, i):
            return (self.uemb(u) * self.iemb(i)).sum(1) + self.ub(u).squeeze() + self.ib(i).squeeze() + self.mu
    model = MF().to(DEV)
    tr = tr.to(DEV); te = te.to(DEV)

    def train_step(model, _):
        b = tr[torch.randint(0, len(tr), (4096,), device=DEV)]
        pred = model(b[:, 0].long(), b[:, 1].long())
        return F.mse_loss(pred, b[:, 2])

    # NDCG@20 / Recall@20: 隐式反馈口径 (评分>=4 为正样本)
    K = 20
    te_pos = te[te[:, 2] >= 4]
    # 每用户的正样本物品 (test)
    from collections import defaultdict
    user_pos = defaultdict(set)
    for row in te_pos:
        user_pos[int(row[0])].add(int(row[1]))
    tr_seen = defaultdict(set)
    for row in tr:
        tr_seen[int(row[0])].add(int(row[1]))
    eval_users = [u for u in user_pos if len(user_pos[u]) > 0][:1000]  # 采样 1000 用户控成本
    all_items = torch.arange(ni, device=DEV)

    def eval_fn(model):
        with torch.no_grad():
            rmse = F.mse_loss(model(te[:, 0].long(), te[:, 1].long()), te[:, 2]).sqrt().item()
            recalls, ndcgs = [], []
            iemb = model.iemb.weight; ib = model.ib.weight.squeeze()
            for u in eval_users:
                ue = model.uemb.weight[u]; ub = model.ub.weight[u]
                scores = iemb @ ue + ib + ub
                for it in tr_seen[u]:
                    scores[it] = -1e9  # 屏蔽训练已见
                topk = torch.topk(scores, K).indices.cpu().tolist()
                pos = user_pos[u]
                hits = [1 if it in pos else 0 for it in topk]
                recalls.append(sum(hits) / min(len(pos), K))
                dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
                idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos), K)))
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            return {"rmse": rmse, "recall@20": sum(recalls) / len(recalls),
                    "ndcg@20": sum(ndcgs) / len(ndcgs)}
    return model, train_step, eval_fn, [None], {"family": "recsys", "epochs": 1000}


# ============ 8. Continual (Rotated MNIST) ============
def build_continual():
    import torchvision
    import torchvision.transforms as T
    tasks = []
    for angle in [0, 30, 60, 90]:
        tf = T.Compose([T.RandomRotation((angle, angle)), T.ToTensor(),
                        T.Normalize((0.1307,), (0.3081,))])
        ds = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
        tasks.append(torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=2))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(784, 256); self.f2 = nn.Linear(256, 256); self.head = nn.Linear(256, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.f1(x)); x = F.relu(self.f2(x)); return self.head(x)
    model = MLP().to(DEV)

    def train_step(model, batch):
        x, y = batch
        return F.cross_entropy(model(x.to(DEV)), y.to(DEV))

    def eval_fn(model):
        return {"note": "avg_acc/forgetting computed in RUNNING continual loop"}
    return model, train_step, eval_fn, tasks, {"family": "continual", "epochs": 1000, "n_tasks": 4}


BUILDERS = {
    "matreg": build_matreg, "mlp": build_mlp, "cnn": build_cnn, "vit": build_vit,
    "graph": build_graph, "sciml": build_sciml, "recsys": build_recsys, "continual": build_continual,
}
