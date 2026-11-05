"""
RUNNING 批次调度: 定义 88-run manifest, 按 GPU 分批顺序执行 (nohup 后台)。
用法:
  python run_plan.py --list                    # 打印 manifest
  python run_plan.py --batch cnn --gpu 0       # 顺序跑 CNN 批 (含 rank sweep + 退路占位)
  python run_plan.py --batch fast --gpu 1      # 图/SciML/mlp/recsys 快族
  python run_plan.py --batch vit --gpu 1
每个 run 调 train.py, 结果写 runs/<family>_<baseline>_r<rank>.json (traj) + 汇总 results_running.json
seed=42, 全轨迹保存。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "runs")
os.makedirs(OUT, exist_ok=True)

EPOCHS = {"matreg": 2000, "mlp": 2000, "cnn": 1000, "vit": 1000,
          "graph": 1000, "sciml": 2000, "recsys": 1000, "continual": 1000}
REC_RANK = {"matreg": 16, "mlp": 16, "cnn": 32, "vit": 16,
            "graph": 8, "sciml": 8, "recsys": 16, "continual": 16}
# 每族 informed HP (graph/sciml 已公平搜索; 其余待校准, 用保守默认)
# vit: 标准配方 lr=1e-3 (3e-3 发散致 AdamW top1=0.15); 配 warmup+wd (见 train.py)
FAM_LR = {"graph": 1e-2, "sciml": 1e-2, "mlp": 3e-3, "cnn": 3e-3,
          "vit": 1e-3, "recsys": 1e-2, "continual": 1e-2}
FAM_WD = {"vit": 0.05, "cnn": 0.0, "graph": 0.0, "sciml": 0.0,
          "mlp": 0.0, "recsys": 0.0, "continual": 0.0}
# informed glr (glr天花板扫描: graph 需大 glr; sciml/其余 0.3 最优)
FAM_GLR = {"graph": 3.0, "sciml": 0.3, "mlp": 0.3, "cnn": 0.3,
           "vit": 0.3, "recsys": 0.3, "continual": 1.0}
ALL_B = ["B1", "B2", "B3", "B4", "B5", "B6"]
ROUTED_B = ["B3", "B4", "B5", "B6"]  # 有 rank 概念的


# 部分族 SGD(B1) 需比 Adam 高的 lr (否则不公平弱下界); ViT 尤其明显 (lr=1e-3 太小 SGD→0.11)
# B1 SGD-M 专用 lr (b1_sgd_sweep 实测最优): graph/sciml/mlp/cnn→0.1, vit→0.01
FAM_SGD_LR = {"vit": 0.01, "cnn": 0.1, "graph": 0.1, "sciml": 0.1, "mlp": 0.1}


def lr_for(fam, baseline):
    if baseline == "B1" and fam in FAM_SGD_LR:
        return FAM_SGD_LR[fam]
    return FAM_LR[fam]


def manifest():
    runs = []
    # ---- 主表: 每族 B1-B6 推荐档 ----
    for fam in ["graph", "sciml", "cnn", "mlp", "vit", "continual", "recsys"]:
        for b in ALL_B:
            runs.append({"family": fam, "baseline": b, "rank": REC_RANK[fam],
                         "epochs": EPOCHS[fam], "lr": lr_for(fam, b), "glr": FAM_GLR[fam],
                         "tag": f"{fam}_{b}_r{REC_RANK[fam]}", "kind": "main"})
    # ---- rank sweep (A11): graph/cnn/sciml, 只路由 baseline, 排除推荐档重复 ----
    for fam in ["graph", "cnn", "sciml"]:
        for r in [4, 8, 16, 32]:
            if r == REC_RANK[fam]:
                continue
            for b in ROUTED_B:
                runs.append({"family": fam, "baseline": b, "rank": r,
                             "epochs": EPOCHS[fam], "lr": FAM_LR[fam], "glr": FAM_GLR[fam],
                             "tag": f"{fam}_{b}_r{r}", "kind": "sweep"})
    # ---- 消融 A1-A12 (代表族) ----
    abls = [
        ("graph", "B5", 8, "A3_no_gmom", {"grassmann_momentum": 0}),
        ("graph", "B5", 8, "A6_main_sgd", {"main_sgd": 1}),
        ("graph", "B5", 8, "A7_no_procrustes", {"procrustes": 0}),
        ("graph", "B5", 8, "A9_sub5", {"sub_every": 5}),
        ("graph", "B5", 8, "A10_sub25", {"sub_every": 25}),
        ("graph", "B5", 8, "A8_hard_reset", {"reset_mode": "hard"}),
        ("cnn", "B5", 32, "A4_no_residual", {"use_residual": 0}),
        ("cnn", "B5", 32, "A5_residual_adam", {"residual_adam": 1}),
        ("cnn", "B5", 32, "A8_hard_reset", {"reset_mode": "hard"}),
        ("cnn", "B5", 32, "A12_no_norm_res", {"normalize_residual": 0}),
    ]
    for fam, b, r, tag, extra in abls:
        runs.append({"family": fam, "baseline": b, "rank": r, "epochs": EPOCHS[fam],
                     "lr": FAM_LR[fam], "glr": FAM_GLR[fam],
                     "tag": f"{fam}_{tag}", "kind": "ablation", "extra": extra})
    return runs


BATCHES = {
    "cnn": lambda r: r["family"] == "cnn",
    "fast": lambda r: r["family"] in ("graph", "sciml", "recsys", "mlp"),
    "gs": lambda r: r["family"] in ("graph", "sciml"),      # 快且高价值: 优先
    "slow_fast": lambda r: r["family"] in ("recsys", "mlp"),  # 慢族后跑
    "vit": lambda r: r["family"] in ("vit", "continual"),
    "graph": lambda r: r["family"] == "graph",
    "sciml": lambda r: r["family"] == "sciml",
    "recsys": lambda r: r["family"] == "recsys",
    "mlp": lambda r: r["family"] == "mlp",
    "continual": lambda r: r["family"] == "continual",
    "main": lambda r: r["kind"] == "main",
    "sweep": lambda r: r["kind"] == "sweep",
    "ablation": lambda r: r["kind"] == "ablation",
}

# 慢族 (full loader × 长 epoch) 限每 epoch 步数, 控总时长 (科学价值不随步数线性)
MAX_STEPS = {"mlp": 100, "continual": 100, "cnn": None, "vit": None}


def run_one(r, gpu):
    tag = r["tag"]
    logf = os.path.join(OUT, f"{tag}.log")
    cmd = [PY, os.path.join(HERE, "train.py"),
           "--family", r["family"], "--baseline", r["baseline"],
           "--epochs", str(r["epochs"]), "--rank", str(r["rank"]),
           "--lr", str(r.get("lr", 1e-2)),
           "--grassmann_lr", str(r.get("glr", 0.3)),   # informed glr (graph/sciml 证明 0.1 欠调)
           "--weight_decay", str(FAM_WD.get(r["family"], 0.0)),
           "--out_dir", OUT]
    ms = MAX_STEPS.get(r["family"])
    if ms:
        cmd += ["--max_steps_per_epoch", str(ms)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GR_TAG=tag)
    # extra flags 通过环境变量传给 train.py (ablation)
    if "extra" in r:
        env["GR_EXTRA"] = json.dumps(r["extra"])
    t0 = time.time()
    with open(logf, "w") as f:
        ret = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return {"tag": tag, "rc": ret.returncode, "wall_s": dt, "log": logf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--batch", choices=list(BATCHES.keys()))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--kind", choices=["main", "sweep", "ablation"], help="额外过滤")
    ap.add_argument("--skip_done", action="store_true", help="跳过已有 traj 的 run")
    a = ap.parse_args()
    runs = manifest()
    if a.list:
        by_fam = {}
        for r in runs:
            by_fam.setdefault(r["family"], []).append(r["tag"])
        for fam, tags in by_fam.items():
            print(f"{fam}: {len(tags)} runs")
        print(f"TOTAL: {len(runs)} runs")
        return
    sel = [r for r in runs if BATCHES[a.batch](r)]
    if a.kind:
        sel = [r for r in sel if r["kind"] == a.kind]
    if a.skip_done:
        before = len(sel)
        sel = [r for r in sel if not os.path.exists(os.path.join(OUT, f"traj_{r['tag']}.json"))]
        print(f"[skip_done] 跳过 {before - len(sel)} 已完成", flush=True)
    print(f"[batch {a.batch}] gpu={a.gpu} {len(sel)} runs", flush=True)
    summary = []
    for i, r in enumerate(sel):
        print(f"[{i+1}/{len(sel)}] {r['tag']} ...", flush=True)
        res = run_one(r, a.gpu)
        summary.append(res)
        print(f"  -> rc={res['rc']} {res['wall_s']:.0f}s", flush=True)
        # 增量写汇总
        with open(os.path.join(OUT, f"_batch_{a.batch}_gpu{a.gpu}.json"), "w") as f:
            json.dump(summary, f, indent=1)
    print(f"[batch {a.batch}] done. ok={sum(1 for s in summary if s['rc']==0)}/{len(summary)}", flush=True)


if __name__ == "__main__":
    main()
