"""
T1-T4 理论对应实验 (theory.md §4 映射) — 从已有 trajectory 提取可观测量, 不需新训练。
T1 Landscape: captured_energy, eigengap, principal_angle_R, grassmann_distance, curvature_proxy
T2 Tracking: R_t 轨迹, retraction/tangency error, tracking-error vs eigengap
T3 Convergence: train_loss, total_grad_norm, rank_tail, residual_grad_norm, 后期退化
T4 Generalization: train-test gap, reset_count, alignment/tangency error
"""
import json
import os
import glob

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def load_traj(tag):
    p = os.path.join(RUNS, f"traj_{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["records"]


def series(recs, key):
    return [r.get(key) for r in recs if r.get(key) is not None]


def summarize(tag, recs):
    """提取 T1-T4 观测量的关键统计。"""
    out = {"tag": tag, "n_epochs": len(recs)}
    # T1 Landscape
    ce = series(recs, "captured_energy")
    eg = series(recs, "eigengap")
    R = series(recs, "principal_angle_R_over_halfpi")
    gd = series(recs, "grassmann_distance")
    cp = series(recs, "curvature_proxy")
    if ce:
        out["T1_captured_energy"] = {"init": ce[0], "max": max(ce), "final": ce[-1]}
    if R:
        out["T1_R_over_halfpi"] = {"max": max(R), "final": R[-1], "stays_below_1": max(R) < 1.0}
    if eg:
        out["T1_eigengap"] = {"mean": sum(eg) / len(eg), "final": eg[-1]}
    if cp:
        out["T1_curvature_proxy_mean"] = sum(cp) / len(cp)
    # T2 Tracking
    ret = series(recs, "retraction_error")
    tan = series(recs, "tangency_error")
    if ret:
        out["T2_retraction_error_max"] = max(ret)
    if tan:
        out["T2_tangency_error_max"] = max(tan)
    if gd:
        out["T2_grassmann_dist"] = {"mean": sum(gd) / len(gd), "final": gd[-1]}
    # T3 Convergence
    tg = series(recs, "total_grad_norm")
    rt = series(recs, "rank_tail")
    rg = series(recs, "residual_grad_norm")
    tl = series(recs, "train_loss")
    if tg:
        out["T3_total_grad_norm"] = {"init": tg[0], "final": tg[-1]}
    if rt:
        # rank-tail 后期是否增大 (isotropic degradation)
        mid = rt[len(rt) // 2] if len(rt) > 2 else rt[0]
        out["T3_rank_tail"] = {"init": rt[0], "mid": mid, "final": rt[-1],
                               "increases_late": rt[-1] > rt[0]}
    if tl:
        out["T3_train_loss"] = {"init": tl[0], "best": min(tl), "final": tl[-1]}
    # T4 Generalization
    rc = series(recs, "reset_count")
    if rc:
        out["T4_reset_count_final"] = rc[-1]
    return out


def main():
    result = {"theory_map": "theory.md §4", "families": {}}
    # T1 → matreg+graph; T2 → cnn+合成床; T3 → sciml; T4 → cnn
    targets = {
        "graph_B5_r8": "T1_landscape (graph, 谱集中静态)",
        "sciml_B5_r8": "T3_convergence (sciml, 漂移谱, 后期rank-tail)",
    }
    # 也扫所有 B5 主表
    for p in sorted(glob.glob(os.path.join(RUNS, "traj_*_B5_r*.json"))):
        tag = os.path.basename(p)[5:-5]
        recs = load_traj(tag)
        if recs:
            result["families"][tag] = summarize(tag, recs)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    with open(os.path.join(RUNS, "theory_check.json"), "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
