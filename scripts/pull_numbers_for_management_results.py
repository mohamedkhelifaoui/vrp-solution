from pathlib import Path
import pandas as pd
import json

BASE = Path(__file__).resolve().parents[1]
R = BASE / "data" / "reports"

# Required inputs (already produced in your pipeline)
EVAL_METHOD = R / "step8_eval_by_method.csv"            # overall eval means by method
BEST_VS_BASE = R / "best_vs_baseline.csv"               # step6 or step8-10 “best-of-sweep vs baseline” with family
CHAMPS = R / "champions.csv"                            # step11 champions
CHAMPS_STATS = R / "champions_stats_by_method.csv"      # step11 champions stats
FINAL_OVERALL = R / "final_overall_table.csv"           # step12 overall means over champions
FINAL_PER_INSTANCE = R / "final_per_instance_table.csv" # step12 per-instance champions

OUT_JSON = R / "management_numbers.json"
OUT_MD   = R / "management_results_filled.md"

def read_optional_csv(p):
    if p.exists():
        df = pd.read_csv(p)
        # normalize lower headers for robustness
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return None

def safe_fmt(x, fmt="{:.2f}"):
    try:
        if pd.isna(x): return "—"
        return fmt.format(float(x))
    except Exception:
        return "—"

eval_m = read_optional_csv(EVAL_METHOD)
best = read_optional_csv(BEST_VS_BASE)
champs = read_optional_csv(CHAMPS)
chstats = read_optional_csv(CHAMPS_STATS)
overall = read_optional_csv(FINAL_OVERALL)
final_pi = read_optional_csv(FINAL_PER_INSTANCE)

numbers = {}

# 1) Overall on-time means by method (from step8_eval_by_method.csv)
if eval_m is not None and {"method","ontime_mean"}.issubset(eval_m.columns):
    # sort highest on-time mean
    e_sorted = eval_m.sort_values("ontime_mean", ascending=False)
    numbers["overall_ontime_by_method"] = (
        e_sorted[["method","ontime_mean"]]
        .to_dict(orient="records")
    )

# 2) Champions stats by method
if chstats is not None:
    # Expect columns like: method, mean_distance, mean_vehicles, mean_p95 (or ontime_p95), n_instances
    # Normalize potential names
    rename_map = {
        "mean_p95":"p95",
        "ontime_p95":"p95",
        "mean_distance":"dist",
        "mean_vehicles":"veh",
        "n_instances":"n"
    }
    for k,v in list(rename_map.items()):
        if k in chstats.columns:
            chstats.rename(columns={k:v}, inplace=True)
    # keep what we have
    keep = [c for c in ["method","dist","veh","p95","n"] if c in chstats.columns]
    numbers["champions_stats_by_method"] = chstats[keep].to_dict(orient="records")

# 3) Overall means over champions (final_overall_table.csv)
if overall is not None:
    # expected: method, mean_distance, mean_vehicles, mean_ontime_p50, mean_ontime_p95, mean_runtime_s, n_instances
    ov = overall.copy()
    ov.columns = [c.replace("mean_","") for c in ov.columns]
    numbers["final_overall_means"] = ov.to_dict(orient="records")

# 4) Median % distance change by family (best_vs_baseline.csv)
if best is not None and {"family","pct_improve"}.issubset(best.columns):
    fam_med = best.groupby("family")["pct_improve"].median().to_dict()
    # rename: pct_improve is positive when robust is better (shorter); distance change vs baseline
    numbers["median_pct_improve_by_family"] = fam_med

# 5) Per-instance champions table counts (for a headline)
if final_pi is not None:
    nrows = len(final_pi)
    methods = final_pi["method"].value_counts().to_dict() if "method" in final_pi.columns else {}
    numbers["final_per_instance_counts"] = {"n_instances": nrows, "by_method": methods}

# Save numbers
OUT_JSON.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

# Build a filled markdown management section (short—edit/expand as you like)
lines = []
lines.append("# Management Results – Filled Numbers\n")

# Headline from champions overall (if available)
if numbers.get("final_overall_means"):
    # pick the best method by p95 (or ontime_p95)
    df = pd.DataFrame(numbers["final_overall_means"])
    # normalize
    for col in ["ontime_p95","p95"]:
        if col in df.columns:
            df["p95"] = df.get("p95", df.get(col))
    best_row = df.sort_values("p95", ascending=False).iloc[0]
    lines.append(f"- **Best reliability overall**: {best_row['method']} with p95≈{safe_fmt(best_row['p95'])}%, "
                 f"distance≈{safe_fmt(best_row['distance'],'{:.1f}')}, vehicles≈{safe_fmt(best_row['vehicles'],'{:.1f}')} "
                 f"(means over champions).")
    lines.append("")

# Overall on-time means by method
if numbers.get("overall_ontime_by_method"):
    lines.append("**Overall on-time (ex-post, common scenarios)**")
    for r in numbers["overall_ontime_by_method"]:
        lines.append(f"- {r['method']}: on-time mean ≈ {safe_fmt(r['ontime_mean'])}%")
    lines.append("")

# Champions stats by method
if numbers.get("champions_stats_by_method"):
    lines.append("**Champion (per-instance) averages by method**")
    for r in numbers["champions_stats_by_method"]:
        meth = r.get("method","?")
        dist = safe_fmt(r.get("dist"), "{:.1f}")
        veh  = safe_fmt(r.get("veh"),  "{:.2f}")
        p95  = safe_fmt(r.get("p95"),  "{:.2f}")
        n    = int(r.get("n",0)) if r.get("n") is not None else 0
        lines.append(f"- {meth}: p95≈{p95}% | distance≈{dist} | vehicles≈{veh} | instances={n}")
    lines.append("")

# Median % improvement by family
if numbers.get("median_pct_improve_by_family"):
    lines.append("**Median % distance improvement (best-of-sweep vs baseline) by family**")
    for fam, val in numbers["median_pct_improve_by_family"].items():
        lines.append(f"- {fam}: {safe_fmt(val, '{:+.2f}%')}")
    lines.append("")

# Counts
if numbers.get("final_per_instance_counts"):
    n = numbers["final_per_instance_counts"]["n_instances"]
    bym = numbers["final_per_instance_counts"]["by_method"]
    lines.append(f"**Per-instance champions:** {n} instances total.")
    if bym:
        lines.append("By method:")
        for m, c in bym.items():
            lines.append(f"- {m}: {c}")
    lines.append("")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("Wrote:")
print(f" - {OUT_JSON}")
print(f" - {OUT_MD}")
