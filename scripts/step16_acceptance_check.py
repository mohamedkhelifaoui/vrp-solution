from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE = Path(__file__).resolve().parents[1]
R = BASE / "data" / "reports"
RAW = BASE / "data" / "raw"

CHAMPIONS   = R / "champions.csv"                                   # Step 11
EVAL        = R / "step8_eval.csv"                                   # evaluate_plans.py (common scenarios)
DET_SUMMARY = BASE / "data" / "solutions_ortools" / "summary.csv"

METHOD_SUMMARY = {
    "det":          BASE / "data" / "solutions_ortools" / "summary.csv",
    "q120":         BASE / "data" / "solutions_quantile" / "m1.2_a0" / "summary.csv",
    "saa16-b0p3":   BASE / "data" / "solutions_saa" / "k16_b0p3" / "summary.csv",
    "saa32-b0p5":   BASE / "data" / "solutions_saa" / "k32_b0p5" / "summary.csv",
    "saa64-b0p7":   BASE / "data" / "solutions_saa" / "k64_b0p7" / "summary.csv",
    "g1":           BASE / "data" / "solutions_gamma" / "g1_q1p645_hybrid" / "summary.csv",
    "g2":           BASE / "data" / "solutions_gamma" / "g2_q1p645_hybrid" / "summary.csv",
    "ch95":         BASE / "data" / "solutions_chance" / "ch95" / "summary.csv",  # present only if you ran Step 11 chance
}

OUT_MD  = R / "step16_acceptance_report.md"
OUT_CSV = R / "step16_acceptance_details.csv"

def normcols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Missing required column; tried {candidates}.\nAvailable: {list(df.columns)}")
    return None

def load_det_summary():
    s = pd.read_csv(DET_SUMMARY)
    s = normcols(s)
    # unify distance/vehicles
    if "det_distance" not in s.columns:
        if "total_distance" in s.columns: s = s.rename(columns={"total_distance":"det_distance"})
        elif "distance" in s.columns:     s = s.rename(columns={"distance":"det_distance"})
        elif "dist" in s.columns:         s = s.rename(columns={"dist":"det_distance"})
    if "det_vehicles" not in s.columns:
        if "vehicles" in s.columns:       s = s.rename(columns={"vehicles":"det_vehicles"})
    return s[["instance","det_distance","det_vehicles"]]

def load_method_summary(label):
    # label is already normalized to lower
    p = METHOD_SUMMARY.get(label, None)
    if not p or not p.exists():
        return None
    s = pd.read_csv(p)
    s = normcols(s)
    # unify columns
    if "distance" not in s.columns:
        for cand in ["total_distance","dist"]:
            if cand in s.columns:
                s = s.rename(columns={cand:"distance"}); break
    if "vehicles" not in s.columns:
        for cand in ["veh","n_vehicles"]:
            if cand in s.columns:
                s = s.rename(columns={cand:"vehicles"}); break
    for rt in ["runtime","runtime_s","time_s","solve_time_s"]:
        if rt in s.columns:
            s = s.rename(columns={rt:"runtime_s"}); break
    keep = [c for c in ["instance","distance","vehicles","runtime_s"] if c in s.columns]
    return s[keep] if keep else None

def main():
    # ---- champions
    champs = pd.read_csv(CHAMPIONS)
    champs = normcols(champs)
    need = {"instance","method"}
    if not need.issubset(champs.columns):
        raise SystemExit("champions.csv must have columns: instance, method (plus distance/vehicles if available).")
    # normalize labels (case-insensitive)
    champs["method"] = champs["method"].astype(str).str.strip().str.lower()
    champs["family"] = champs["instance"].str.extract(r"^([A-Z]+)", expand=False)

    # ---- load DET baseline summary
    det = load_det_summary()

    # ---- evaluation table (common scenarios)
    ev = pd.read_csv(EVAL)
    ev = normcols(ev)
    # normalize labels for merge
    if "method" in ev.columns:
        ev["method"] = ev["method"].astype(str).str.strip().str.lower()

    # pick evaluation columns robustly
    on_mean = pick_col(ev, ["ontime_mean","on_time_mean","ontime","on_time_avg","on_time"])
    on_p50  = pick_col(ev, ["ontime_p50","on_time_p50","p50","median_ontime","median_on_time"])
    on_p95  = pick_col(ev, ["ontime_p95","on_time_p95","p95","p95_ontime","p95_on_time"])

    # det evaluation slice
    det_ev = ev[ev["method"]=="det"][["instance",on_mean,on_p50,on_p95]].rename(
        columns={on_mean:"det_on_mean", on_p50:"det_on_p50", on_p95:"det_on_p95"}
    )

    # champion evaluation rows
    ch_ev = champs.merge(
        ev[["instance","method",on_mean,on_p50,on_p95]],
        on=["instance","method"], how="left"
    ).rename(columns={on_mean:"ch_on_mean", on_p50:"ch_on_p50", on_p95:"ch_on_p95"})

    # ---- ensure champion distances/vehicles/runtime
    if "distance" not in ch_ev.columns: ch_ev["distance"] = np.nan
    if "vehicles" not in ch_ev.columns: ch_ev["vehicles"] = np.nan

    rows = []
    for _, r in ch_ev.iterrows():
        rr = r.copy()
        need_fill = pd.isna(rr["distance"]) or pd.isna(rr["vehicles"]) or ("runtime_s" not in ch_ev.columns) or pd.isna(rr.get("runtime_s", np.nan))
        if need_fill:
            s = load_method_summary(rr["method"])
            if s is not None:
                row = s[s["instance"]==rr["instance"]]
                if not row.empty:
                    if pd.isna(rr["distance"]) and "distance" in row.columns:
                        rr["distance"] = float(row.iloc[0]["distance"])
                    if pd.isna(rr["vehicles"]) and "vehicles" in row.columns:
                        rr["vehicles"] = float(row.iloc[0]["vehicles"])
                    if "runtime_s" in row.columns:
                        rr["runtime_s"] = float(row.iloc[0]["runtime_s"])
        rows.append(rr)
    ch_ev = pd.DataFrame(rows)

    # ---- merge baseline cost & evaluation
    full = ch_ev.merge(det, on="instance", how="left").merge(det_ev, on="instance", how="left")

    # Safety: if any on-time is still missing for champions, mark as NaN and continue
    if "ch_on_mean" not in full.columns:
        full["ch_on_mean"] = np.nan
    if "ch_on_p50" not in full.columns:
        full["ch_on_p50"] = np.nan
    if "ch_on_p95" not in full.columns:
        full["ch_on_p95"] = np.nan

    # ---- deltas & acceptance
    full["delta_on_pp"]   = full["ch_on_mean"] - full["det_on_mean"]
    full["delta_dist_pct"] = 100.0 * (full["distance"] - full["det_distance"]) / full["det_distance"]

    condA = (full["delta_on_pp"] >= 8.0) & (full["delta_dist_pct"] <= 3.0)           # +8 pp on-time, ≤ +3% dist
    condB = (full["distance"] <= 0.95 * full["det_distance"]) & (full["ch_on_mean"] >= full["det_on_mean"] - 1e-9)
    full["meets_A"] = condA
    full["meets_B"] = condB
    full["meets_any"] = condA | condB

    # runtime flag > 1800s (if available)
    if "runtime_s" not in full.columns:
        full["runtime_s"] = np.nan
    full["runtime_flag"] = full["runtime_s"] > 1800

    # save details
    keep = ["instance","family","method",
            "det_distance","distance","delta_dist_pct",
            "det_on_mean","ch_on_mean","delta_on_pp",
            "ch_on_p50","ch_on_p95","vehicles","runtime_s","runtime_flag",
            "meets_A","meets_B","meets_any"]
    for c in keep:
        if c not in full.columns: full[c] = np.nan
    full[keep].to_csv(OUT_CSV, index=False)

    # summary stats
    n_total = len(full)
    n_ok = int(full["meets_any"].fillna(False).sum())
    pct_ok = 100.0 * n_ok / max(1,n_total)

    fam_acc = full.groupby("family")["meets_any"].mean().mul(100).round(1).fillna(0.0).to_dict()
    med_on   = float(np.nanmedian(full["delta_on_pp"]))
    med_dist = float(np.nanmedian(full["delta_dist_pct"]))
    n_rtflag = int(full["runtime_flag"].fillna(False).sum())

    # top-5 & worst-5
    top = full.sort_values(["delta_on_pp","delta_dist_pct"], ascending=[False, True]).head(5)
    worst = full.sort_values("delta_dist_pct", ascending=False).head(5)

    lines = []
    lines.append("# Step 16 — Acceptance Report\n")
    lines.append(f"- Champions evaluated: **{n_total}**")
    lines.append(f"- Instances meeting acceptance (A or B): **{n_ok}/{n_total} = {pct_ok:.1f}%**\n")
    lines.append("**Acceptance rate by family (%):**")
    for k in ["C","R","RC"]:
        if k in fam_acc: lines.append(f"- {k}: {fam_acc[k]:.1f}%")
    lines.append("")
    lines.append(f"**Median improvements** — on-time mean: **{med_on:+.2f} pp**, distance: **{med_dist:+.2f}%**.\n")
    lines.append(f"**Runtime flags (>1800 s)**: {n_rtflag} instance(s)\n")
    lines.append("**Top-5 reliability gains (Δon-time, with lowest Δdistance):**")
    for _,r in top.iterrows():
        lines.append(f"- {r['instance']} ({r['method']}): +{r['delta_on_pp']:.2f} pp, Δdist {r['delta_dist_pct']:.2f}%")
    lines.append("")
    lines.append("**Top-5 distance increases (to review):**")
    for _,r in worst.iterrows():
        lines.append(f"- {r['instance']} ({r['method']}): Δdist {r['delta_dist_pct']:.2f}%, Δon-time {r['delta_on_pp']:+.2f} pp")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:")
    print(f" - {OUT_CSV}")
    print(f" - {OUT_MD}")

if __name__ == "__main__":
    main()
