from pathlib import Path
import json, os, glob, math
OUT=str(Path(__file__).resolve().parents[1] / "eda_out")

def last_snap_stats(tr):
    s=tr["snapshots"]
    if not s: return None
    l=s[-1]; f=s[0]
    d=tr["d"]
    row={"d":d,
         "erank_part_first":round(f["erank_participation"],1),
         "erank_part_last":round(l["erank_participation"],1),
         "stable_rank_last":round(l["stable_rank"],2),
         }
    for r in ["4","8","16","32"]:
        if r in l.get("energy_topr",{}):
            row[f"E_top{r}_first"]=round(f["energy_topr"][r],3)
            row[f"E_top{r}_last"]=round(l["energy_topr"][r],3)
    for r in ["16"]:
        if r in l.get("eigengap_r_norm",{}):
            row[f"ngap{r}_last"]=round(l["eigengap_r_norm"][r],4)
    # drift over snapshots (mean per-step grassmann dist, R radius)
    gds=[x["drift"]["grassmann_dist_per_step"] for x in s if "drift" in x]
    Rs=[x["drift"]["R_over_halfpi"] for x in s if "drift" in x]
    if gds:
        row["drift_gd_per_step_mean"]=round(sum(gds)/len(gds),5)
        row["R_over_halfpi_max"]=round(max(Rs),3)
        row["R_over_halfpi_mean"]=round(sum(Rs)/len(Rs),3)
    nd=None
    return row

for jf in sorted(glob.glob(f"{OUT}/*.json")):
    name=os.path.basename(jf)
    if name in ("synth_tracking_bed.json",):
        continue
    try:
        d=json.load(open(jf))
    except Exception as e:
        print(f"### {name}: LOAD ERR {e}"); continue
    print(f"\n===== {name} =====")
    for taskkey, task in d.items():
        if not isinstance(task,dict): continue
        if "trackers" not in task:
            # nested (matreg style handled below)
            if "error" in task: print(f"  {taskkey}: ERROR {task['error']}"); continue
            continue
        print(f"  [{taskkey}] loss {task.get('loss_first')}->{task.get('loss_last')} wall {task.get('wall_sec')}s")
        for ln,tr in task["trackers"].items():
            row=last_snap_stats(tr)
            nd=task.get("null_drift",{}).get(ln,{})
            ndR=round(nd.get("null_R_mean",float('nan'))/(math.pi/2),3) if nd else None
            print(f"    {ln:22s} d={row['d']:5d} Etop16 {row.get('E_top16_first','-')}->{row.get('E_top16_last','-')} "
                  f"erank {row['erank_part_first']}->{row['erank_part_last']} stblrank {row['stable_rank_last']} "
                  f"ngap16 {row.get('ngap16_last','-')} drift/step {row.get('drift_gd_per_step_mean','-')} "
                  f"Rmax {row.get('R_over_halfpi_max','-')} nullR {ndR}")
