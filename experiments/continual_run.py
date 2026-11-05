from pathlib import Path
"""
真实持续学习多任务循环 (Rotated MNIST 0/30/60/90°) — 漂移主场, G2 第2族候选。
顺序训 4 个任务, 每任务 N epoch, 记录:
  - avg_acc: 全部任务训完后在 4 个任务上的平均准确率
  - forgetting: 每任务训完时 acc - 最终 acc 的均值
  - subspace_drift: 任务切换时 U 的 Grassmann 距离 (GrassRoute 特有)
对比 B2/B4/B5/B6。漂移假设预测: 在线跟踪 (B5) 在任务切换的大漂移下优于周期SVD (B4)。
seed=42。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import build_optimizer
from routing import build_param_groups

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_ROOT = os.environ.get("GRASSROUTE_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))


def get_tasks(angles=(0, 30, 60, 90), n_per=2000):
    import torchvision
    import torchvision.transforms as T
    tasks = []
    for a in angles:
        tf = T.Compose([T.RandomRotation((a, a)), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        tr = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
        te = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
        trl = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True, num_workers=2)
        tel = torch.utils.data.DataLoader(te, batch_size=512, num_workers=2)
        tasks.append((trl, tel))
    return tasks


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(784, 256); self.f2 = nn.Linear(256, 256); self.head = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x)); x = F.relu(self.f2(x)); return self.head(x)


@torch.no_grad()
def evaluate(model, tel):
    model.eval(); correct = 0; n = 0
    for x, y in tel:
        p = model(x.to(DEV)).argmax(1).cpu()
        correct += (p == y).sum().item(); n += y.size(0)
    model.train()
    return correct / n


def run(baseline, epochs_per_task=15, steps_per_epoch=100, lr=1e-2, glr=0.3, rank=16):
    torch.manual_seed(SEED)
    tasks = get_tasks()
    model = MLP().to(DEV)
    if baseline in ("B1", "B2"):
        opt = build_optimizer(baseline, model.parameters(), lr=lr)
    else:
        groups, _ = build_param_groups(model, "continual", base_lr=lr)
        opt = build_optimizer(baseline, groups, rank=rank, lr=lr, grassmann_lr=glr)
    acc_after_task = []      # acc[i][j] = 训完任务i后在任务j的acc
    subspace_drift = []
    prev_U = None
    for ti, (trl, tel) in enumerate(tasks):
        for ep in range(epochs_per_task):
            model.train()
            for si, (x, y) in enumerate(trl):
                opt.zero_grad()
                loss = F.cross_entropy(model(x.to(DEV)), y.to(DEV))
                loss.backward(); opt.step()
                if si + 1 >= steps_per_epoch:
                    break
        # 任务切换时记录子空间漂移 (GrassRoute)
        if baseline in ("B3", "B4", "B5", "B6"):
            for g in opt.param_groups:
                if g.get("routed"):
                    for p in g["params"]:
                        st = opt.state.get(p, {})
                        if "U" in st:
                            U = st["U"]
                            if prev_U is not None and prev_U.shape == U.shape:
                                M = prev_U.T @ U
                                s = torch.linalg.svdvals(M).clamp(-1, 1)
                                th = torch.arccos(s)
                                subspace_drift.append(float(th.max().item() / (math.pi / 2)))
                            prev_U = U.clone()
                            break
                    break
        accs = [evaluate(model, t[1]) for t in tasks]
        acc_after_task.append(accs)
    final_accs = acc_after_task[-1]
    avg_acc = sum(final_accs) / len(final_accs)
    # forgetting: 每任务峰值(训完该任务时) - 最终
    forgetting = sum(acc_after_task[i][i] - final_accs[i] for i in range(len(tasks))) / len(tasks)
    return {"baseline": baseline, "avg_acc": avg_acc, "forgetting": forgetting,
            "final_accs": final_accs, "mean_subspace_drift": (sum(subspace_drift) / len(subspace_drift)) if subspace_drift else None,
            "acc_matrix": acc_after_task}


if __name__ == "__main__":
    res = {}
    for b in ["B2", "B4", "B5", "B6"]:
        r = run(b)
        res[b] = r
        print(f"{b}: avg_acc={r['avg_acc']:.4f} forgetting={r['forgetting']:.4f} "
              f"drift={r['mean_subspace_drift']}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "continual_results.json")
    json.dump(res, open(out, "w"), indent=1)
    print("[saved]", out)
