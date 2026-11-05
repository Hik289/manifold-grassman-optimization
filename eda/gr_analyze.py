from pathlib import Path
import json, os, glob, math
import numpy as np
OUT=str(Path(__file__).resolve().parents[1] / "eda_out")
HALFPI=math.pi/2

def agg_task_trackers(task):
    """Return per-layer aggregated concentration + drift stats (main task json format)."""
    rows=[]
    for ln,tr in task.get("trackers",{}).items():
        s=tr["snapshots"]
        if not s: continue
        d=tr["d"]; f=s[0]; l=s[-1]
        # concentration over training: min/median/last effective rank & top-r energy
        eranks=[x["erank_participation"] for x in s]
        stbl=[x["stable_rank"] for x in s]
        E16=[x["energy_topr"].get("16") for x in s if "16" in x.get("energy_topr",{})]
        E8=[x["energy_topr"].get("8") for x in s if "8" in x.get("energy_topr",{})]
        E4=[x["energy_topr"].get("4") for x in s if "4" in x.get("energy_topr",{})]
        gds=[x["drift"]["grassmann_dist_per_step"] for x in s if "drift" in x]
        Rs=[x["drift"]["R_over_halfpi"] for x in s if "drift" in x]
        nd=task.get("null_drift",{}).get(ln,{})
        rows.append({"layer":ln,"d":d,
            "erank_med":float(np.median(eranks)),"erank_last":eranks[-1],
            "stable_rank_med":float(np.median(stbl)),
            "E_top4_last":(E4[-1] if E4 else None),
            "E_top8_last":(E8[-1] if E8 else None),
            "E_top16_last":(E16[-1] if E16 else None),
            "E_top16_med":(float(np.median(E16)) if E16 else None),
            "drift_gd_per_step_med":(float(np.median(gds)) if gds else None),
            "R_over_halfpi_med":(float(np.median(Rs)) if Rs else None),
            "R_over_halfpi_max":(float(np.max(Rs)) if Rs else None),
            "null_R_over_halfpi":(nd.get("null_R_mean",np.nan)/HALFPI if nd else None),
            "null_gd":(nd.get("null_grassmann_dist_mean") if nd else None),
        })
    return rows

def analyze_drift_probe():
    d=json.load(open(f"{OUT}/drift_probe.json"))
    res={}
    for tk,task in d.items():
        lay={}
        for ln,tr in task["trackers"].items():
            snaps=tr["snaps"]; dd=tr["d"]
            # for each rank, aggregate within_noise vs consec_drift vs to_final(last)
            perrank={}
            for r in ["1","2","4","8","16"]:
                wn=[x["within_noise"].get(r) for x in snaps if r in x.get("within_noise",{})]
                cd=[x["consec_drift"].get(r) for x in snaps if r in x.get("consec_drift",{})]
                tf=[x["to_final"].get(r) for x in snaps if r in x.get("to_final",{})]
                if not wn: continue
                perrank[r]={
                    "within_noise_med":float(np.median(wn)),
                    "consec_drift_med":(float(np.median(cd)) if cd else None),
                    "drift_over_noise":(float(np.median(cd)/ (np.median(wn)+1e-12)) if cd else None),
                    "to_final_first":(tf[0] if tf else None),
                    "to_final_last":(tf[-1] if len(tf)>1 else None),
                }
            lay[ln]={"d":dd,"perrank":perrank}
        res[tk]={"loss":[task["loss_first"],task["loss_last"]],"layers":lay}
    return res

# ---- main tasks ----
print("="*90)
print("CONCENTRATION + DRIFT (per task family, key layers)")
print("="*90)
main_files=["matreg","mlp","cnn","vit","graph","sciml","recsys","continual"]
allrows={}
for name in main_files:
    fp=f"{OUT}/{name}.json"
    if not os.path.exists(fp): continue
    d=json.load(open(fp))
    print(f"\n##### {name} #####")
    fam_rows=[]
    for tkey,task in d.items():
        if not isinstance(task,dict) or "trackers" not in task:
            if isinstance(task,dict) and "error" in task: print(f"  {tkey}: ERR {task['error']}")
            continue
        rows=agg_task_trackers(task)
        for r in rows: r["task"]=tkey
        fam_rows+=rows
        for r in rows:
            print(f"  {tkey:22s} {r['layer']:20s} d={r['d']:5d} "
                  f"erank(med/last)={r['erank_med']:.1f}/{r['erank_last']:.1f} "
                  f"Etop4/8/16={r['E_top4_last']}/{r['E_top8_last']}/{r['E_top16_last']} "
                  f"drift/step(med)={r['drift_gd_per_step_med']} "
                  f"R/(pi2) med/max={r['R_over_halfpi_med']}/{r['R_over_halfpi_max']} "
                  f"nullR={None if r['null_R_over_halfpi'] is None else round(r['null_R_over_halfpi'],3)}")
    allrows[name]=fam_rows

print("\n"+"="*90)
print("DRIFT-vs-NOISE DECOMPOSITION (split-half probe) — drift_over_noise<~1 => slow drift")
print("="*90)
dp=analyze_drift_probe()
for tk,t in dp.items():
    print(f"\n##### {tk}  loss {t['loss'][0]:.3g}->{t['loss'][1]:.3g} #####")
    for ln,L in t["layers"].items():
        print(f"  {ln:8s} d={L['d']}")
        for r,v in L["perrank"].items():
            print(f"    r={r:>2s}: within_noise={v['within_noise_med']:.3f} consec_drift={v['consec_drift_med']} "
                  f"drift/noise={None if v['drift_over_noise'] is None else round(v['drift_over_noise'],2)} "
                  f"to_final {v['to_final_first']}->{v['to_final_last']}")

json.dump({"main":allrows,"drift_probe":dp}, open(f"{OUT}/_analysis_summary.json","w"), indent=1)
print("\n[saved] _analysis_summary.json")
