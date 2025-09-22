from pathlib import Path
import json, re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = Path(__file__).resolve().parents[1]
RAW       = BASE / "data" / "raw"
FIGS      = BASE / "data" / "figures"
REPORTS   = BASE / "data" / "reports"
CHAMPS    = BASE / "data" / "champions"
DET_DIR   = BASE / "data" / "solutions_ortools"

FIGS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

def add_family_col(df):
    fam = df["instance"].str.extract(r"^([A-Z]+)").iloc[:,0]
    return df.assign(family=fam)

def safe_read_csv(p):
    if not Path(p).exists(): 
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def load_det_summary():
    s = safe_read_csv(DET_DIR / "summary.csv")
    if s is None:
        raise SystemExit("Missing baseline: data/solutions_ortools/summary.csv")
    # normalize
    s.columns = [c.strip().lower() for c in s.columns]
    s = s.rename(columns={"total_distance":"dist_base","vehicles":"vehicles_base","feasible":"feasible_base"})
    s = s[["instance","dist_base","vehicles_base","feasible_base"]]
    return add_family_col(s)

def load_eval():
    # step8_eval.csv contains on-time stats for every method you evaluated
    e = safe_read_csv(REPORTS/"step8_eval.csv")
    if e is None:
        raise SystemExit("Missing evaluation: data/reports/step8_eval.csv (run evaluate_plans.py)")
    e.columns = [c.strip() for c in e.columns]
    needed = {"instance","method","ontime_mean","ontime_p50","ontime_p95"}
    missing = needed - set(e.columns)
    if missing:
        raise SystemExit(f"step8_eval.csv is missing columns: {missing}")
    return add_family_col(e)

def load_champions():
    c = safe_read_csv(REPORTS/"champions.csv")
    if c is None:
        raise SystemExit("Missing champions.csv (run pick_champions.py)")
    c.columns = [c.strip() for c in c.columns]
    # make sure distance/vehicles columns exist; if not, we’ll enrich later from per-method summaries
    return add_family_col(c)

def enrich_champion_costs_from_method_summaries(champs):
    # If distance/vehicles missing, try to recover from summary.csv in the folder of each method label
    method_dirs = {
        "DET":          BASE / "data" / "solutions_ortools",
        "Q120":         BASE / "data" / "solutions_quantile" / "m1.2_a0",
        "SAA16-b0p3":   BASE / "data" / "solutions_saa" / "k16_b0p3",
        "SAA32-b0p5":   BASE / "data" / "solutions_saa" / "k32_b0p5",
        "SAA64-b0p7":   BASE / "data" / "solutions_saa" / "k64_b0p7",
        "G1":           BASE / "data" / "solutions_gamma" / "g1_q1p645_hybrid",
        "G2":           BASE / "data" / "solutions_gamma" / "g2_q1p645_hybrid",
    }
    rows = []
    for _, r in champs.iterrows():
        inst, m = r["instance"], r["method"]
        dist = r.get("distance", np.nan)
        veh  = r.get("vehicles", np.nan)
        if (pd.isna(dist) or pd.isna(veh)) and m in method_dirs:
            sumf = method_dirs[m] / "summary.csv"
            s = safe_read_csv(sumf)
            if s is not None:
                ss = s.copy()
                ss.columns = [c.strip().lower() for c in ss.columns]
                rr = ss.loc[ss["instance"]==inst]
                if not rr.empty:
                    if pd.isna(dist):
                        for cand in ["total_distance","distance","dist"]:
                            if cand in rr.columns:
                                dist = float(rr.iloc[0][cand]); break
                    if pd.isna(veh):
                        for cand in ["vehicles","veh","n_vehicles"]:
                            if cand in rr.columns:
                                veh = int(rr.iloc[0][cand]); break
        row = dict(r)
        row["distance"] = dist
        row["vehicles"] = veh
        rows.append(row)
    out = pd.DataFrame(rows)
    return out

# ---------- ROUTE PLOT HELPERS (from champions JSON) ----------
def read_instance(csv_path: Path):
    df = pd.read_csv(csv_path).rename(columns={
        "CUST NO.":"cust","XCOORD.":"x","YCOORD.":"y",
        "DEMAND":"demand","READY TIME":"ready","DUE DATE":"due","SERVICE TIME":"service"
    })
    for c in ["cust","x","y","demand","ready","due","service"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # depot first (demand=0 & service=0)
    depot_idx = int(df.index[(df["demand"]==0) & (df["service"]==0)][0])
    if depot_idx != 0:
        order = [depot_idx] + [i for i in range(len(df)) if i != depot_idx]
        df = df.iloc[order].reset_index(drop=True)
    cust_to_node = {int(df.loc[i,"cust"]): i for i in range(len(df))}
    return df, cust_to_node

def load_routes_from_json(p: Path):
    js = json.loads(p.read_text())
    return js.get("routes", [])

def plot_routes_for_champion(inst: str, outpng: Path):
    csv_path = RAW / f"{inst}.csv"
    js_path  = CHAMPS / f"{inst}.json"
    if not csv_path.exists() or not js_path.exists():
        return False
    df, custmap = read_instance(csv_path)
    routes = load_routes_from_json(js_path)

    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    plt.figure(figsize=(5.5,5.5))
    # draw customers
    plt.scatter(x[1:], y[1:], s=12, alpha=0.8)
    # depot
    plt.scatter([x[0]],[y[0]], s=60, marker="s")

    # draw routes as sequences depot->cust...->depot
    for r in routes:
        seq = [0] + [custmap[c] for c in r] + [0]
        xs = x[seq]; ys = y[seq]
        plt.plot(xs, ys, linewidth=1)

    plt.title(f"{inst} – Champion plan")
    plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout()
    plt.savefig(outpng, dpi=150); plt.close()
    return True

def main():
    det = load_det_summary()
    evaldf = load_eval()
    champs = load_champions()

    # Ensure champs have distance/vehicles
    if ("distance" not in champs.columns) or ("vehicles" not in champs.columns):
        champs = enrich_champion_costs_from_method_summaries(champs)

    # ---------- Trade-off scatter (method-level means) ----------
    # Average over instances per method from step8_eval + any cost we have on champions (for consistency, we rely on champions distance)
    # Compute per-method champion averages for distance and on-time
    champ_eval = champs.merge(evaldf[["instance","method","ontime_p95","ontime_p50","ontime_mean"]],
                              on=["instance","method"], how="left")
    method_rollup = champ_eval.groupby("method").agg(
        mean_dist=("distance","mean"),
        mean_p95=("ontime_p95","mean"),
        mean_p50=("ontime_p50","mean"),
        n=("instance","count")
    ).reset_index()

    plt.figure(figsize=(6.5,4.5))
    plt.scatter(method_rollup["mean_dist"], method_rollup["mean_p95"])
    for _, r in method_rollup.iterrows():
        plt.annotate(f'{r["method"]} (n={int(r["n"])})', (r["mean_dist"], r["mean_p95"]), fontsize=8, xytext=(4,4), textcoords="offset points")
    plt.xlabel("Mean distance (champions)"); plt.ylabel("Mean on-time p95 (champions, %)")
    plt.title("Method trade-off (champion set)")
    plt.tight_layout()
    plt.savefig(FIGS/"tradeoff_champions_scatter.png", dpi=150); plt.close()

    # ---------- Champion share by family ----------
    fam_share = champs.groupby(["family","method"]).size().reset_index(name="count")
    # Pivot to a stacked bar chart
    pivot = fam_share.pivot(index="family", columns="method", values="count").fillna(0)
    pivot.to_csv(REPORTS/"champion_share_by_family.csv")
    pivot.plot(kind="bar", stacked=True, figsize=(7,4.5))
    plt.ylabel("# champion wins"); plt.title("Champion share by family"); plt.tight_layout()
    plt.savefig(FIGS/"champion_share_by_family.png", dpi=150); plt.close()

    # ---------- p95 on-time by family & method (from eval, all runs) ----------
    # Use all evaldf (not only champions) to show method reliability by family
    box = evaldf.copy()
    # Keep only methods present in your current champion/eval setup (optional)
    keep_methods = method_rollup["method"].unique().tolist()
    box = box[box["method"].isin(keep_methods)]
    plt.figure(figsize=(7.5,4.5))
    # manual grouped box: one boxplot per (family, method)
    groups = []
    labels = []
    for fam in ["C","R","RC"]:
        for m in sorted(keep_methods):
            vals = box[(box["family"]==fam) & (box["method"]==m)]["ontime_p95"].dropna().to_numpy()
            if len(vals) > 0:
                groups.append(vals); labels.append(f"{fam}-{m}")
    if groups:
        plt.boxplot(groups, labels=labels, vert=True, showfliers=False)
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.ylabel("On-time p95 (%)")
        plt.title("On-time p95 by family & method (all instances)")
        plt.tight_layout()
        plt.savefig(FIGS/"p95_by_family_method_box.png", dpi=150); plt.close()

    # ---------- Improvement vs Deterministic (per champion) ----------
    det_small = det[["instance","dist_base","vehicles_base"]]
    comp = champs.merge(det_small, on="instance", how="left")
    comp["delta_dist"] = comp["distance"] - comp["dist_base"]
    comp["pct_improve"] = 100.0 * (comp["dist_base"] - comp["distance"]) / comp["dist_base"]
    comp.to_csv(REPORTS/"champion_vs_det_improvement.csv", index=False)

    plt.figure(figsize=(6.5,4))
    vals = comp["pct_improve"].dropna().to_numpy()
    plt.hist(vals, bins=15)
    plt.xlabel("% distance improvement vs DET (champions)"); plt.ylabel("Count")
    plt.title("Distribution of champion improvements vs deterministic")
    plt.tight_layout()
    plt.savefig(FIGS/"champion_improvement_hist.png", dpi=150); plt.close()

    # ---------- Top-10 hardest instances under DET ----------
    # Hard = lowest p95 on-time for DET in the full evaluation set
    det_eval = evaldf[evaldf["method"]=="DET"].copy()
    if not det_eval.empty:
        hardest = det_eval.sort_values("ontime_p95").head(10)
        hardest.to_csv(REPORTS/"top10_hard_instances_under_det.csv", index=False)

    # ---------- Route plots for a few champions ----------
    shortlist = []
    # one per family if possible
    for fam in ["C","R","RC"]:
        group = champs[champs["family"]==fam].sort_values("distance")
        if not group.empty:
            shortlist.append(group.iloc[0]["instance"])
    # add two more best overall
    best2 = champs.sort_values("distance").head(5)["instance"].tolist()
    for inst in best2:
        if inst not in shortlist:
            shortlist.append(inst)
    made = []
    for inst in shortlist[:6]:
        outpng = FIGS / f"champion_route_{inst}.png"
        ok = plot_routes_for_champion(inst, outpng)
        if ok:
            made.append(inst)
    if made:
        (REPORTS/"champion_route_plots.txt").write_text("Made route plots for: " + ", ".join(made), encoding="utf-8")

    print("Wrote:")
    print(f" - {FIGS/'tradeoff_champions_scatter.png'}")
    print(f" - {FIGS/'champion_share_by_family.png'}")
    print(f" - {FIGS/'p95_by_family_method_box.png'} (if groups existed)")
    print(f" - {FIGS/'champion_improvement_hist.png'}")
    print(f" - {REPORTS/'champion_share_by_family.csv'}")
    print(f" - {REPORTS/'champion_vs_det_improvement.csv'}")
    print(f" - {REPORTS/'top10_hard_instances_under_det.csv'} (if DET present)")
    print(f" - Route plots under {FIGS} starting with champion_route_*.png")

if __name__ == "__main__":
    main()
