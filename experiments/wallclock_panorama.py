"""
Wall-clock 全景: 每任务族 GrassRoute(B5) vs AdamW(B2) 的 per-step ms + 路由层数。
诚实呈现 G3 的"内存换计算" trade-off: 省状态内存 O((m+d)r), 花每步子空间更新时间。
关键归因: single-layer QR 便宜 (O(mdr)), total = O(mdr) × n_routed_layers。
→ 层少的族(graph 2层/sciml 4层) 开销小; ViT(48层) 开销大。
"""
import torch
import time
import json
import os
import torch.nn.functional as F

import tasks
from baselines import build_optimizer
from routing import build_param_groups

DEV = "cuda"


def per_step_ms(family, baseline, iters=15, warmup=5, **kw):
    torch.manual_seed(42)
    built = tasks.BUILDERS[family]()
    if built is None:
        return None, None
    model, train_step, eval_fn, loader, meta = built
    if baseline == "B2":
        opt = build_optimizer("B2", model.parameters(), lr=kw.get("lr", 3e-3))
        n_routed = 0
    else:
        groups, names = build_param_groups(model, family, base_lr=kw.get("lr", 3e-3))
        n_routed = sum(len(g["params"]) for g in groups if g.get("routed"))
        opt = build_optimizer(baseline, groups, rank=kw.get("rank", 16),
                              lr=kw.get("lr", 3e-3), grassmann_lr=kw.get("glr", 0.3))
    is_fb = (loader == [None])
    if family == "continual" and isinstance(loader, list):
        loader = loader[0]

    def one():
        if is_fb:
            opt.zero_grad(); loss = train_step(model, None); loss.backward(); opt.step()
        else:
            batch = next(_it)
            opt.zero_grad(); loss = train_step(model, batch); loss.backward(); opt.step()
    if not is_fb:
        _it = iter(loader)
    for _ in range(warmup):
        if is_fb:
            opt.zero_grad(); train_step(model, None).backward(); opt.step()
        else:
            b = next(_it); opt.zero_grad(); train_step(model, b).backward(); opt.step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        if is_fb:
            opt.zero_grad(); train_step(model, None).backward(); opt.step()
        else:
            b = next(_it); opt.zero_grad(); train_step(model, b).backward(); opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000, n_routed


def run():
    fams = [("graph", {"rank": 8, "glr": 3.0, "lr": 1e-2}),
            ("sciml", {"rank": 8, "glr": 0.3, "lr": 1e-2}),
            ("mlp", {"rank": 16, "glr": 0.3}),
            ("cnn", {"rank": 32, "glr": 0.3}),
            ("vit", {"rank": 16, "glr": 0.3, "lr": 1e-3}),
            ("recsys", {"rank": 16, "glr": 0.3, "lr": 1e-2})]
    res = {}
    print(f"{'family':>10} {'AdamW ms':>10} {'GrassRoute ms':>14} {'slowdown':>9} {'n_routed':>9}")
    for fam, kw in fams:
        try:
            aw, _ = per_step_ms(fam, "B2", **kw)
            gr, nr = per_step_ms(fam, "B5", **kw)
            if aw is None or gr is None:
                print(f"{fam:>10}  SKIP (data)"); continue
            res[fam] = {"adamw_ms": aw, "grassroute_ms": gr, "slowdown": gr / aw, "n_routed_layers": nr}
            print(f"{fam:>10} {aw:>10.1f} {gr:>14.1f} {gr/aw:>8.2f}x {nr:>9}", flush=True)
        except Exception as e:
            import traceback
            print(f"{fam:>10}  FAIL {type(e).__name__}: {str(e)[:60]}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "wallclock_panorama.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    run()
