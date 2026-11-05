"""
聚焦 HP 搜索 (§10 子集): 为 GrassRoute (B5) 在指定任务族找工作点。
默认 glr=0.1 是 anchor_1 matreg 调的, 真实任务需要更大 glr + main/residual LR 配比。

搜索维度 (§10 子集, 控成本):
  lr (参数)       ∈ {1e-2, 3e-2}
  grassmann_lr    ∈ {0.3, 1.0, 3.0}
  mu (Gmomentum)  ∈ {0.9, 0.95}
  lr_perp/lr 比   ∈ {0.5, 1, 2}   (main/residual 配比, sciml 问题疑点)

用法: python hp_search.py --family graph --epochs 1000 --gpu 2
输出: runs/hp_<family>.json (每组 test/loss) + 打印最优
"""
import argparse, json, os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train


def metric_of(s, family):
    m = s["final_metrics"]
    if family == "graph":
        return m.get("test"), "test-acc(↑)"
    if family == "sciml":
        return -s["final_train_loss"], "-train_loss(↑)"
    if family in ("cnn", "vit"):
        return m.get("top1"), "top1(↑)"
    if family == "mlp":
        return m.get("acc"), "acc(↑)"
    if family == "recsys":
        return m.get("ndcg@20"), "ndcg@20(↑)"
    return -s["final_train_loss"], "-loss(↑)"


def run(family, epochs, gpu, rank=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from routing import FAMILY_RANK
    rank = rank or FAMILY_RANK.get(family, 16)
    grid = list(itertools.product([1e-2, 3e-2], [0.3, 1.0, 3.0], [0.9, 0.95]))
    print(f"[hp {family}] rank={rank} {len(grid)} configs", flush=True)
    res = []
    best = None
    for lr, glr, mu in grid:
        os.environ["GR_EXTRA"] = json.dumps({"mu": mu})  # mu 经 GR_EXTRA→build_optimizer
        s = train.train_one(family, "B5", epochs=epochs, rank=rank, lr=lr,
                            grassmann_lr=glr, eval_every=epochs - 1, out_dir="/tmp/hp",
                            weight_decay=0.0)
        val, name = metric_of(s, family)
        rec = {"lr": lr, "glr": glr, "mu": mu, "metric": val,
               "final_loss": s["final_train_loss"], "mem": s.get("state_memory_bytes")}
        res.append(rec)
        if best is None or (val is not None and val > best["metric"]):
            best = rec
        print(f"lr={lr} glr={glr} mu={mu} ratio={ratio}: {name}={val:.4f}", flush=True)
    out = {"family": family, "metric_name": name, "best": best, "grid": res}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"hp_{family}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"[BEST] {best}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--rank", type=int, default=None)
    a = ap.parse_args()
    run(a.family, a.epochs, a.gpu, a.rank)
