from pathlib import Path
import pandas as pd
import numpy as np
import re

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "data" / "reports"
CHAMPIONS = REPORTS / "champions.csv"            # produced in Step 11
EVAL = REPORTS / "step8_eval.csv"                # produced in Steps 8–10/11
OUT1 = REPORTS / "final_per_instance_table.csv"
OUT2 = REPORTS / "final_overall_table.csv"
OUTMD = REPORTS / "final_per_instance_table.md"

# Map your method labels -> where their summary.csv lives
METHOD_DIRS = {
    "DET":          BASE / "data" / "solutions_ortools",
    "Q120":         BASE / "data" / "solutions_quantile" / "m1.2_a0",
    "SAA16-b0p3":   BASE / "data" / "solutions_saa" / "k16_b0p3",
    "SAA32-b0p5":   BASE / "data" / "solutions_saa" / "k32_b0p5",
    "SAA64-b0p7":   BASE / "data" / "solutions_saa" / "k64_b0p7",
    "G1":           BASE / "data" / "solutions_gamma" / "g1_q1p645_hybrid",
    "G2":           BASE / "data" / "solutions_gamma" / "g2_q1p645_hybrid",
}

def add_family_col(df):
    fam = df["instance"].str.extract(r"^([A-Z]+)").iloc[:,0]
    return df.assign(family=fam)

def load_runtime_for(method_label, instance):
    """Look up runtime for (method, instance) from that method's summary.csv if present."""
    p = METHOD_DIRS.get(method_label, None)
    if p is None: 
        return np.nan
    sumf = p / "summary.csv"
    if not sumf.exists():
        return np.nan
    s = pd.read_csv(sumf)
    # be defensive on column names
    s.columns = [c.strip().lower() for c in s.columns]
    row = s.loc[s["instance"] == instance]
    if row.empty:
        return np.nan
    for cand in ["runtime","runtime_s","time_s","solve_time_s"]:
        if cand in row.columns:
            return float(row.iloc[0][cand])
    return np.nan

def main():
    if not CHAMPIONS.exists():
        raise SystemExit(f"Missing {CHAMPIONS}. Run Step 11 champion picker first.")

    champs = pd.read_csv(CHAMPIONS)
    # Normalize column names
    champs.columns = [c.strip() for c in champs.columns]
    need = {"instance","method"}
    if not need.issubset(set(champs.columns)):
        raise SystemExit("champions.csv must have columns: instance, method (plus distance/vehicles if available).")

    # If distance/vehicles are missing in champions.csv, pull them from the method summaries
    have_distance = "distance" in champs.columns or "total_distance" in champs.columns
    have_vehicles = "vehicles" in champs.columns
    if "total_distance" in champs.columns and "distance" not in champs.columns:
        champs = champs.rename(columns={"total_distance":"distance"})

    if not (have_distance and have_vehicles):
        # attempt to enrich from summary.csv of each method
        rows = []
        for _, r in champs.iterrows():
            inst, m = r["instance"], r["method"]
            p = METHOD_DIRS.get(m, None)
            dist = r.get("distance", np.nan)
            veh  = r.get("vehicles", np.nan)
            if p and (np.isnan(dist) or np.isnan(veh)):
                sumf = p / "summary.csv"
                if sumf.exists():
                    s = pd.read_csv(sumf)
                    s.columns = [c.strip().lower() for c in s.columns]
                    rr = s.loc[s["instance"]==inst]
                    if (not rr.empty):
                        if np.isnan(dist):
                            for cand in ["total_distance","distance","dist"]:
                                if cand in rr.columns:
                                    dist = float(rr.iloc[0][cand]); break
                        if np.isnan(veh):
                            for cand in ["vehicles","veh","n_vehicles"]:
                                if cand in rr.columns:
                                    veh = int(rr.iloc[0][cand]); break
            rr = dict(r)
            rr["distance"] = dist
            rr["vehicles"] = veh
            rows.append(rr)
        champs = pd.DataFrame(rows)

    # Attach family
    champs = add_family_col(champs)

    # Attach stochastic evaluation stats from step8_eval.csv (p50/p95/mean)
    if not EVAL.exists():
        raise SystemExit(f"Missing {EVAL}. Run evaluate_plans.py first.")
    ev = pd.read_csv(EVAL)
    ev.columns = [c.strip() for c in ev.columns]
    keepcols = ["instance","method","ontime_mean","ontime_p50","ontime_p95"]
    ev = ev[keepcols].drop_duplicates()

    tbl = champs.merge(ev, on=["instance","method"], how="left")

    # Attach runtimes per instance from each method's summary.csv
    runtimes = []
    for _, r in tbl.iterrows():
        runtimes.append(load_runtime_for(r["method"], r["instance"]))
    tbl["runtime_s"] = runtimes

    # Order columns & sort
    cols = ["instance","family","method","vehicles","distance","ontime_p50","ontime_p95","ontime_mean","runtime_s"]
    for c in cols:
        if c not in tbl.columns: tbl[c] = np.nan
    tbl = tbl[cols].sort_values(["family","instance"]).reset_index(drop=True)

    # Save
    tbl.to_csv(OUT1, index=False)

    # Overall summary by method (means over champions)
    overall = tbl.groupby("method").agg(
        mean_distance=("distance","mean"),
        mean_vehicles=("vehicles","mean"),
        mean_ontime_p50=("ontime_p50","mean"),
        mean_ontime_p95=("ontime_p95","mean"),
        mean_runtime_s=("runtime_s","mean"),
        n_instances=("instance","count"),
    ).sort_values("mean_distance")
    overall.to_csv(OUT2)

    # Optional Markdown view
    md = tbl.to_markdown(index=False, floatfmt=".3f")
    OUTMD.write_text(md, encoding="utf-8")

    print("Wrote:")
    print(f" - {OUT1}")
    print(f" - {OUT2}")
    print(f" - {OUTMD}")

if __name__ == "__main__":
    main()
