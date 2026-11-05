"""
聚合所有 runs/traj_*.json 为标准 results.json (主表 + sweep + 消融)。
用法: python aggregate.py > runs/results_running.json
"""
import json
import os
import glob

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def load(tag):
    p = os.path.join(RUNS, f"traj_{tag}.json")
    if not os.path.exists(p):
        return None
    try:
        recs = json.load(open(p))["records"]   # 半写入/坏文件 -> JSONDecodeError, 跳过不冻结聚合
    except Exception:
        return None
    if not recs:
        return None
    final = recs[-1]
    best_loss = min(r["train_loss"] for r in recs)
    best_test = None
    tests = [r.get("test_metric") for r in recs if r.get("test_metric") is not None]
    vals = [r.get("val_metric") for r in recs if r.get("val_metric") is not None]
    return {
        "final_train_loss": final["train_loss"],
        "best_train_loss": best_loss,
        "final_test": final.get("test_metric"),
        "best_test": max(tests) if tests else None,
        "final_val": final.get("val_metric"),
        "best_val": max(vals) if vals else None,
        "state_memory_bytes": final.get("state_memory_bytes"),
        "final_captured_energy": final.get("captured_energy"),
        "final_R_over_halfpi": final.get("principal_angle_R_over_halfpi"),
        "final_eigengap": final.get("eigengap"),
        "n_epochs": len(recs),
        "wall_clock": final.get("wall_clock"),
    }


def main():
    out = {"main_table": {}, "rank_sweep": {}, "ablation": {}}
    FAMS = ["matreg", "mlp", "cnn", "vit", "graph", "sciml", "recsys", "continual"]
    REC_RANK = {"mlp": 16, "cnn": 32, "vit": 16, "graph": 8, "sciml": 8, "recsys": 16, "continual": 16}
    for fam in FAMS:
        r = REC_RANK.get(fam, 16)
        row = {}
        for b in ["B1", "B2", "B3", "B4", "B5", "B6"]:
            d = load(f"{fam}_{b}_r{r}")
            if d:
                row[b] = d
        if row:
            out["main_table"][fam] = row
    # rank sweep
    for fam in ["graph", "cnn", "sciml"]:
        sweep = {}
        for rk in [4, 8, 16, 32]:
            rr = {}
            for b in ["B3", "B4", "B5", "B6"]:
                d = load(f"{fam}_{b}_r{rk}")
                if d:
                    rr[b] = {"best_train_loss": d["best_train_loss"], "best_test": d["best_test"]}
            if rr:
                sweep[f"r{rk}"] = rr
        if sweep:
            out["rank_sweep"][fam] = sweep
    # ablation
    for p in sorted(glob.glob(os.path.join(RUNS, "traj_*_A*.json"))):
        tag = os.path.basename(p)[5:-5]
        d = load(tag)
        if d:
            out["ablation"][tag] = {"best_train_loss": d["best_train_loss"],
                                    "best_test": d["best_test"], "final_test": d["final_test"]}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
