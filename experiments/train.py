"""
统一训练入口: 任意 (任务族 × baseline) 组合, 逐 epoch 记录 §5.3 20 项。

用法:
  python train.py --family cnn --baseline B5 --epochs 1000 --rank 32 --gpu 0
  python train.py --smoke        # 每族×每baseline 跑2 epoch, 验证骨架打通

不自行开跑全量长训练; RUNNING 阶段由 Director 派单后批量调度。
seed=42 固定 (README §5.1)。
"""
import argparse
import json
import math
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import BUILDERS, DEV
from baselines import build_optimizer
from routing import build_param_groups, FAMILY_RANK
from trajectory import TrajectoryLogger

SEED = 42


def set_seed(s=SEED):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_optimizer(baseline_id, model, family, lr, grassmann_lr, rank, weight_decay):
    """B1/B2: 全参数; B3-B6: 路由 param_groups (按 EDA rank 分档)。"""
    bid = baseline_id.upper()
    if bid in ("B1", "B2", "SGD", "SGD-M", "ADAMW"):
        return build_optimizer(baseline_id, model.parameters(), lr=lr, weight_decay=weight_decay)
    groups, names = build_param_groups(model, family, base_lr=lr, weight_decay=weight_decay)
    # ablation flags via GR_EXTRA env (run_plan.py 注入)
    extra = {}
    if os.environ.get("GR_EXTRA"):
        extra = json.loads(os.environ["GR_EXTRA"])
        # int->bool 还原 (json 传 0/1)
        for k in ("grassmann_momentum", "use_residual", "residual_adam", "main_sgd",
                  "procrustes", "normalize_residual", "normalize_cov"):
            if k in extra:
                extra[k] = bool(extra[k])
    # rank 覆盖: 命令行 rank 优先, 否则用 EDA 分档 (build_param_groups 已按 d 分档)
    opt = build_optimizer(baseline_id, groups, rank=rank or FAMILY_RANK.get(family, 16),
                          lr=lr, grassmann_lr=grassmann_lr, weight_decay=weight_decay, **extra)
    return opt


def cosine_lr(ep, epochs, base, floor=0.01, warmup=0):
    # warmup: 前 warmup epoch 线性升到 base, 之后 cosine decay (ViT 标准)
    if warmup > 0 and ep < warmup:
        return base * (ep + 1) / warmup
    e = ep - warmup
    E = max(1, epochs - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * e / E)))


# 贵评估族 (full test set forward) 降频, 省 wall-clock; 廉价族每 epoch
EVAL_EVERY = {"cnn": 25, "vit": 25, "mlp": 10, "continual": 5,
              "graph": 1, "sciml": 25, "recsys": 25, "matreg": 20}


def train_one(family, baseline_id, epochs, rank=None, lr=3e-3, grassmann_lr=0.1,
              weight_decay=0.0, eval_every=None, out_dir=".", log_every=1, max_steps_per_epoch=None):
    if eval_every is None:
        eval_every = EVAL_EVERY.get(family, 10)
    set_seed()
    built = BUILDERS[family]()
    if built is None:
        print(f"[skip] {family}: 数据未就绪 (需 RUNNING 前下载)")
        return None
    model, train_step, eval_fn, loader, meta = built
    opt = make_optimizer(baseline_id, model, family, lr, grassmann_lr, rank, weight_decay)
    logger = TrajectoryLogger(baseline_id)
    base_lrs = [g["lr"] for g in opt.param_groups]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    is_full_batch = (loader == [None])
    # continual: loader 是 [DataLoader,...] 多任务列表; 骨架 smoke 用首个任务, RUNNING 用专门循环
    if meta.get("family") == "continual" and isinstance(loader, list) and loader and loader[0] is not None:
        loader = loader[0]
    # ViT 标准: warmup (前 5% epoch) 防止早期 AdamW 发散
    warmup = int(0.05 * epochs) if family == "vit" else 0
    for ep in range(epochs):
        # cosine LR decay (README §5) + warmup (ViT)
        for g, l0 in zip(opt.param_groups, base_lrs):
            g["lr"] = cosine_lr(ep, epochs, l0, warmup=warmup)
            if "lr_perp" in g:
                g["lr_perp"] = cosine_lr(ep, epochs, l0, warmup=warmup)
        model.train()
        last_loss = None
        if is_full_batch:
            opt.zero_grad()
            loss = train_step(model, None)
            loss.backward()
            opt.step()
            last_loss = loss.item()
        else:
            for si, batch in enumerate(loader):
                opt.zero_grad()
                loss = train_step(model, batch)
                loss.backward()
                opt.step()
                last_loss = loss.item()
                if max_steps_per_epoch and si + 1 >= max_steps_per_epoch:
                    break
        # §5.3 轨迹每 epoch 记录 (train_loss + 子空间诊断廉价); test/val 评估按 eval_every 降频 (贵)
        do_eval = (ep % eval_every == 0) or (ep == epochs - 1)
        metrics = eval_fn(model) if do_eval else {}
        val = metrics.get("val") or metrics.get("acc") or metrics.get("top1")
        test = metrics.get("test") or metrics.get("mse") or metrics.get("rmse")
        logger.log_epoch(ep, opt, last_loss, val_metric=val, test_metric=test,
                         model=model, grad_params=list(model.parameters()))
    summary = logger.summary()
    summary["final_metrics"] = eval_fn(model)
    summary["state_memory_bytes"] = opt.state_memory_bytes() if hasattr(opt, "state_memory_bytes") else None
    summary["family"] = family
    summary["baseline"] = baseline_id
    os.makedirs(out_dir, exist_ok=True)
    # tag 含 rank + ablation 后缀, 防止 sweep/ablation 覆盖主表
    tag = os.environ.get("GR_TAG") or f"{family}_{baseline_id}_r{rank or FAMILY_RANK.get(family, 16)}"
    summary["tag"] = tag
    logger.save(os.path.join(out_dir, f"traj_{tag}.json"))
    return summary


def smoke():
    """每族 × 每 baseline 跑 2 epoch, 验证骨架打通。"""
    results = {}
    for family in BUILDERS:
        results[family] = {}
        for bid in ["B1", "B2", "B3", "B4", "B5", "B6"]:
            try:
                t0 = time.time()
                s = train_one(family, bid, epochs=2, eval_every=1,
                              max_steps_per_epoch=3, out_dir="/tmp/gr_smoke")
                if s is None:
                    results[family][bid] = "SKIP(data)"
                else:
                    results[family][bid] = f"OK loss={s['final_train_loss']:.3e} mem={s['state_memory_bytes']}"
                print(f"[smoke] {family:10s} {bid}: {results[family][bid]} ({time.time()-t0:.1f}s)")
            except Exception as e:
                results[family][bid] = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
                print(f"[smoke] {family:10s} {bid}: {results[family][bid]}")
    with open("/tmp/gr_smoke/smoke_results.json", "w") as f:
        json.dump(results, f, indent=1)
    # 汇总
    n_ok = sum(1 for fam in results.values() for v in fam.values() if v.startswith("OK"))
    n_total = sum(len(fam) for fam in results.values())
    print(f"\n[smoke summary] {n_ok}/{n_total} 组合通过")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(BUILDERS.keys()))
    ap.add_argument("--baseline", default="B5")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grassmann_lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--max_steps_per_epoch", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    else:
        s = train_one(a.family, a.baseline, a.epochs, rank=a.rank, lr=a.lr,
                      grassmann_lr=a.grassmann_lr, weight_decay=a.weight_decay, out_dir=a.out_dir,
                      max_steps_per_epoch=a.max_steps_per_epoch)
        print(json.dumps(s, indent=1))
