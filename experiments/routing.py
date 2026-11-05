"""
路由层选择 + rank 分档 (EDA §2/§5 的三条实现约束)。

约束:
 1. ViT 白名单: 仅 attn.qkv/attn.proj + mlp.fc1/mlp.fc2, 排除 cls_token/patch_embed/pos_embed/head-as-token
 2. CNN 3×3 conv 右维 d 跨度 576→4608, rank 按 d 分档; 1×1 conv 与 fc 单独档
 3. PINN out(m=1) 与 RecSys embedding(d=64) 是退化边界, 单独处理

rank 分档 (EDA §5):
  图/SciML: r=4–8 ; MLP/ViT/持续: r=8–16 ; CNN conv: r=16–32, 分类头: r=8
"""
import torch
import torch.nn as nn


# 每任务族推荐 rank (EDA §5 表)
FAMILY_RANK = {
    "matreg": 16, "mlp": 16, "cnn": 32, "vit": 16,
    "graph": 8, "sciml": 8, "recsys": 16, "continual": 16,
}

# ViT 白名单关键字 (参数名包含才路由)
VIT_WHITELIST = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2",
                 "attn.q", "attn.k", "attn.v", "attn.o")
VIT_BLACKLIST = ("cls_token", "patch_embed", "pos_embed", "head")


def rank_for_d(d, family):
    """按右维 d + 任务族返回分档 rank (EDA §5)。"""
    base = FAMILY_RANK.get(family, 16)
    if family == "cnn":
        if d <= 128:            # 1×1 downsample conv
            return 8
        elif d <= 576:          # 小 3×3 conv
            return 16
        else:                   # 大 3×3 conv (1152–4608)
            return 32
    if d < 8:                   # 退化边界 (太小不路由)
        return None
    return min(base, d)


def is_routable_matrix(name, param, family, min_d=8, max_d=6000, cnn_partial=False):
    """判定一个参数是否应被路由 (右维 d∈[min_d,max_d], 2D+ 矩阵)。
    cnn_partial=True: CNN 退路, 只路由集中层 (fc + 1×1 conv), 3×3 conv 留给 fallback。"""
    if param.dim() < 2:
        return False
    # ViT 白名单/黑名单
    if family == "vit":
        if any(b in name for b in VIT_BLACKLIST):
            return False
        if not any(w in name for w in VIT_WHITELIST):
            return False
    # CNN partial-routing 退路: 只路由 fc (2D) + 1×1 conv (kh=kw=1); 排除 3×3 conv
    if family == "cnn" and cnn_partial:
        if param.dim() == 4 and (param.shape[2] > 1 or param.shape[3] > 1):
            return False  # 3×3 (或更大) conv → 不路由, 留 AdamW/SGD
    d = 1
    for s in param.shape[1:]:
        d *= s
    if d < min_d or d > max_d:
        return False
    return True


def build_param_groups(model, family, base_lr=3e-3, weight_decay=0.0, cnn_partial=False):
    """把 model 参数拆成 (routed group, non-routed group), 每 routed 层带自己的 rank。
    返回 param_groups (list of dict) 供 GrassRoute/baseline 使用。
    注意: GrassRoute 用统一 group rank; 这里按 rank 值分成多个 routed group。"""
    from collections import defaultdict
    routed_by_rank = defaultdict(list)
    non_routed = []
    routed_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_routable_matrix(name, p, family, cnn_partial=cnn_partial):
            d = 1
            for s in p.shape[1:]:
                d *= s
            r = rank_for_d(d, family)
            if r is None:
                non_routed.append(p)
                continue
            routed_by_rank[r].append(p)
            routed_names.append((name, tuple(p.shape), r))
        else:
            non_routed.append(p)
    groups = []
    for r, ps in routed_by_rank.items():
        groups.append({"params": ps, "rank": r, "routed": True,
                       "lr": base_lr, "weight_decay": weight_decay})
    if non_routed:
        groups.append({"params": non_routed, "routed": False,
                       "lr": base_lr, "weight_decay": weight_decay})
    return groups, routed_names


def summarize_routing(model, family):
    """打印/返回路由清单 (anchor_3 用)。"""
    groups, names = build_param_groups(model, family)
    n_routed = sum(len(g["params"]) for g in groups if g["routed"])
    n_total = sum(1 for _ in model.parameters())
    return {"family": family, "n_routed": n_routed, "n_total": n_total,
            "routed_layers": [{"name": n, "shape": list(s), "rank": r} for n, s, r in names],
            "rank_distribution": {str(g["rank"]): len(g["params"]) for g in groups if g["routed"]}}
