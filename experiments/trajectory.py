"""
逐 epoch 轨迹记录器 (README §5.3 的 20 项, 按可行性)。

20 项:
 1 train_loss          2 val_metric        3 test_metric
 4 total_grad_norm     5 routed_grad_norm  6 residual_grad_norm
 7 captured_energy     8 eigengap          9 rank_tail
10 grassmann_distance 11 principal_angles 12 curvature_proxy
13 grassmann_grad_norm 14 grassmann_momentum_norm
15 retraction_error   16 transport_error  17 alignment_error
18 reset_count        19 wall_clock       20 peak_memory

对无子空间的 B1/B2, 子空间相关项 (5-18) 记为 null/0。
principal_angles / eigengap 从路由层的 U 与 C=G^T G 谱计算 (按层聚合取代表值)。
"""
import torch
import math
import time
import json


class TrajectoryLogger:
    def __init__(self, baseline_id):
        self.bid = baseline_id
        self.records = []
        self.prev_U = {}          # 每个路由层上一次的 U (算 grassmann_distance)
        self.t0 = time.time()

    @torch.no_grad()
    def log_epoch(self, epoch, opt, train_loss, val_metric=None, test_metric=None,
                  model=None, grad_params=None):
        rec = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_metric": None if val_metric is None else float(val_metric),
            "test_metric": None if test_metric is None else float(test_metric),
            "wall_clock": time.time() - self.t0,
        }
        # peak memory (GPU)
        if torch.cuda.is_available():
            rec["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
        else:
            rec["peak_memory_mb"] = None

        diags = opt.collect_diagnostics() if hasattr(opt, "collect_diagnostics") else []

        if diags:
            # 聚合所有路由层 (加权/取代表): 这里对标量取均值, 保留 per-layer 明细
            def agg(key, default=0.0):
                vals = [d[key] for d in diags if key in d]
                return float(sum(vals) / len(vals)) if vals else default
            rec["total_grad_norm"] = agg("total_grad_norm")
            rec["routed_grad_norm"] = agg("routed_grad_norm")
            rec["residual_grad_norm"] = agg("residual_grad_norm")
            rec["captured_energy"] = agg("captured_energy")
            rec["grassmann_grad_norm"] = agg("grassmann_grad_norm")
            rec["grassmann_momentum_norm"] = agg("grassmann_momentum_norm")
            rec["retraction_error"] = agg("retraction_error")
            rec["tangency_error"] = agg("tangency_error")
            rec["residual_scale"] = agg("residual_scale", None) if any("residual_scale" in d for d in diags) else None
            rec["reset_count"] = int(sum(d.get("reset_count", 0) for d in diags))
            rec["n_routed_layers"] = len(diags)
        else:
            # B1/B2 或无诊断: 从 grad_params 直接算 total_grad_norm
            if grad_params is not None:
                gn = 0.0
                for p in grad_params:
                    if p.grad is not None:
                        gn += float(p.grad.norm().item()) ** 2
                rec["total_grad_norm"] = math.sqrt(gn)
            for k in ("routed_grad_norm", "residual_grad_norm", "captured_energy",
                      "grassmann_grad_norm", "grassmann_momentum_norm",
                      "retraction_error", "reset_count"):
                rec[k] = None

        # 子空间谱量 (eigengap / rank_tail / principal_angles / grassmann_distance)
        # 从 optimizer 的路由层 U + 缓存的 C 谱计算 (可行时)
        rec.update(self._subspace_metrics(opt))

        rec["state_memory_bytes"] = opt.state_memory_bytes() if hasattr(opt, "state_memory_bytes") else None
        self.records.append(rec)
        return rec

    @torch.no_grad()
    def _subspace_metrics(self, opt):
        """从路由层 U 计算 grassmann_distance (逐 epoch U 漂移) + eigengap 代表值。
        eigengap 需要 C 谱; 训练脚本若在 diag 里存了 eigengap 则用之, 否则 null。"""
        out = {"grassmann_distance": None, "eigengap": None,
               "principal_angle_R": None, "rank_tail": None, "curvature_proxy": None}
        if not hasattr(opt, "param_groups"):
            return out
        gdists, Rs = [], []
        for group in getattr(opt, "param_groups", []):
            if not group.get("routed", False):
                continue
            for p in group["params"]:
                st = opt.state.get(p, {})
                if "U" not in st:
                    continue
                U = st["U"]
                pid = id(p)
                if pid in self.prev_U:
                    Uprev = self.prev_U[pid]
                    M = Uprev.T @ U
                    s = torch.linalg.svdvals(M).clamp(-1, 1)
                    th = torch.arccos(s)
                    gdists.append(float(torch.sqrt((th ** 2).sum()).item()))
                    Rs.append(float(th.max().item()))
                self.prev_U[pid] = U.clone()
                # eigengap/rank_tail 从 diag 读 (若训练脚本算了)
                d = st.get("diag", {})
                if d.get("eigengap") is not None and out["eigengap"] is None:
                    out["eigengap"] = d["eigengap"]
                if d.get("rank_tail") is not None and out["rank_tail"] is None:
                    out["rank_tail"] = d["rank_tail"]
                if d.get("curvature_proxy") is not None and out["curvature_proxy"] is None:
                    out["curvature_proxy"] = d["curvature_proxy"]
        if gdists:
            out["grassmann_distance"] = float(sum(gdists) / len(gdists))
            out["principal_angle_R"] = float(max(Rs))
            out["principal_angle_R_over_halfpi"] = float(max(Rs) / (math.pi / 2))
        return out

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"baseline": self.bid, "records": self.records}, f, indent=1)

    def summary(self):
        if not self.records:
            return {}
        final = self.records[-1]
        best = min(self.records, key=lambda r: r["train_loss"])
        return {"final_train_loss": final["train_loss"],
                "best_train_loss": best["train_loss"],
                "final_test_metric": final.get("test_metric"),
                "final_state_memory_bytes": final.get("state_memory_bytes"),
                "total_wall_clock": final.get("wall_clock"),
                "n_epochs": len(self.records)}
