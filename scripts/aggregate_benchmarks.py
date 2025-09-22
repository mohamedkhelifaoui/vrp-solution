# scripts/aggregate_benchmarks.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import re

BASE     = Path(__file__).resolve().parents[1]
BENCH    = BASE / "data" / "benchmarks"
REPORTS  = BASE / "data" / "reports"
FIGS     = BASE / "data" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def parse_tag(tag: str):
    """Parse a tag like TL30_MGLS_V10000 -> (30, 'GLS', 10000)"""
    m = re.match(r"TL(\d+)_M([A-Z]+)_V(\d+)", tag)
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))

# ---------- Load baseline (greedy) ----------
base = pd.read_csv(BASE / "data/solutions/summary.csv")

# Defensive types
for c in ["vehicles", "total_distance", "horizon"]:
    if c in base.columns:
        base[c] = pd.to_numeric(base[c], errors="coerce")

# Always derive family from instance (C/R/RC)
base["family"] = base["instance"].astype(str).str.extract(r"^([A-Z]+)")
base = base.rename(columns={
    "vehicles": "vehicles_base",
    "total_distance": "dist_base",
    "feasible": "feasible_base",
})

# ---------- Load all benchmark summaries ----------
rows = []
if BENCH.exists():
    for d in sorted(BENCH.iterdir()):
        if not d.is_dir():
            continue
        tag = d.name
        tl, meta, vcost = parse_tag(tag)
        if tl is None:
            # Skip any folder not matching our tag format
            continue
        sumf = d / "summary.csv"
        if not sumf.exists():
            continue
        df = pd.read_csv(sumf)
        # Robust types
        for c in ["vehicles", "total_distance", "horizon"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["tag"] = tag
        df["time_limit"] = tl
        df["meta"] = meta
        df["vehicle_cost"] = vcost
        # Derive family here as well
        df["family"] = df["instance"].astype(str).str.extract(r"^([A-Z]+)")
        rows.append(df)

if not rows:
    raise SystemExit("No benchmark summaries found in data/benchmarks/<TAG>/summary.csv")

allruns = pd.concat(rows, ignore_index=True)

# Standardize OR-Tools column names
allruns = allruns.rename(columns={
    "vehicles": "vehicles_ort",
    "total_distance": "dist_ort",
    "feasible": "feasible_ort",
})

# Unified objective (distance + vehicle_cost * vehicles)
allruns["objective_ort"] = allruns["dist_ort"] + allruns["vehicle_cost"] * allruns["vehicles_ort"]

# ---------- Save combined table ----------
all_csv = REPORTS / "benchmarks_all_runs.csv"
allruns.to_csv(all_csv, index=False)

# ---------- Success rates ----------
succ_by_tag = (allruns.groupby("tag")["feasible_ort"]
               .apply(lambda s: (s == True).mean() * 100)
               .sort_values(ascending=False))
succ_by_tag.to_csv(REPORTS / "success_rate_by_tag.csv", header=["success_rate_%"])

succ_by_family_tag = (allruns.groupby(["family", "tag"])["feasible_ort"]
                      .apply(lambda s: (s == True).mean() * 100)
                      .reset_index(name="success_rate_%"))
succ_by_family_tag.to_csv(REPORTS / "success_rate_by_family_tag.csv", index=False)

# ---------- Means per tag (feasible only) ----------
feas = allruns[allruns["feasible_ort"] == True].copy()
if feas.empty:
    # Write minimal outputs and stop cleanly
    (REPORTS / "means_by_tag.csv").write_text("")
    (REPORTS / "best_method_per_instance.csv").write_text("")
    (REPORTS / "best_vs_baseline.csv").write_text("")
    print("No feasible solutions found in benchmarks. Wrote empty placeholders.")
    raise SystemExit(0)

means_by_tag = feas.groupby("tag").agg(
    mean_dist=("dist_ort", "mean"),
    mean_veh=("vehicles_ort", "mean"),
    count=("instance", "count"),
).sort_values("mean_dist")
means_by_tag.to_csv(REPORTS / "means_by_tag.csv")

# ---------- Best config per instance (min objective_ort) ----------
best_idx = feas.groupby("instance")["objective_ort"].idxmin()
best = feas.loc[best_idx].copy()
best.to_csv(REPORTS / "best_method_per_instance.csv", index=False)

# ---------- Compare best-of-sweep vs baseline ----------
comp = best.merge(
    base[["instance", "family", "vehicles_base", "dist_base"]],
    on="instance", how="left"
)

# Fallback: (re)derive family if merge didn’t bring it (prevents KeyError)
if "family" not in comp.columns or comp["family"].isna().all():
    comp["family"] = comp["instance"].astype(str).str.extract(r"^([A-Z]+)")

# Drop rows with missing or zero baseline distance to avoid div-by-zero
comp = comp.dropna(subset=["dist_base", "dist_ort"])
comp = comp[comp["dist_base"] > 0].copy()

comp["delta_dist"] = comp["dist_ort"] - comp["dist_base"]
comp["pct_improve"] = 100 * (comp["dist_base"] - comp["dist_ort"]) / comp["dist_base"]
comp["delta_veh"] = comp["vehicles_ort"] - comp["vehicles_base"]
comp.to_csv(REPORTS / "best_vs_baseline.csv", index=False)

# ---------- Figures ----------
# 1) Success rate by tag
if not succ_by_tag.empty:
    plt.figure(figsize=(10, 5))
    succ_by_tag.plot(kind="bar")
    plt.ylabel("Success rate (%)")
    plt.title("Feasible-solution rate by configuration (tag)")
    plt.tight_layout()
    plt.savefig(FIGS / "success_rate_by_tag.png", dpi=150)
    plt.close()

# 2) Feasible count by tag
feasible_counts = (allruns.groupby("tag")["feasible_ort"]
                   .sum().astype(int).sort_values(ascending=False))
if not feasible_counts.empty:
    plt.figure(figsize=(10, 5))
    feasible_counts.plot(kind="bar")
    plt.ylabel("# feasible instances")
    plt.title("Feasible count by configuration (tag)")
    plt.tight_layout()
    plt.savefig(FIGS / "feasible_count_by_tag.png", dpi=150)
    plt.close()

# 3) Boxplot: best vs baseline improvement by family
comp_for_plot = comp.dropna(subset=["family", "pct_improve"])
if not comp_for_plot.empty and comp_for_plot["family"].notna().any():
    plt.figure(figsize=(8, 5))
    comp_for_plot.boxplot(column="pct_improve", by="family", grid=False)
    plt.suptitle("")
    plt.title("Best-of-sweep vs Baseline: % distance improvement (positive = better)")
    plt.xlabel("Family")
    plt.ylabel("% improvement")
    plt.tight_layout()
    plt.savefig(FIGS / "best_vs_base_box_by_family.png", dpi=150)
    plt.close()

# ---------- Text summary ----------
summary_txt = REPORTS / "step6_summary.txt"
with summary_txt.open("w", encoding="utf-8") as f:
    f.write("=== Step 6 Tuning Summary ===\n")
    f.write(f"Combined runs table: {all_csv}\n\n")

    f.write("-- Success rate by tag (%) --\n")
    f.write(succ_by_tag.to_string(float_format=lambda x: f"{x:.1f}") + "\n\n")

    f.write("-- Success rate by family & tag (%) --\n")
    f.write(succ_by_family_tag.to_string(index=False, float_format=lambda x: f"{x:.1f}") + "\n\n")

    f.write("-- Means by tag (feasible only) --\n")
    f.write(means_by_tag.to_string(float_format=lambda x: f"{x:.3f}") + "\n\n")

    f.write("-- Top 10 best improvements (best-of-sweep vs baseline) --\n")
    top10 = comp.sort_values("pct_improve", ascending=False).head(10)
    cols = ["instance","family","vehicles_base","vehicles_ort","dist_base","dist_ort","pct_improve","tag","time_limit","meta","vehicle_cost"]
    f.write(top10[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")

print("Wrote:")
print(f" - {REPORTS/'benchmarks_all_runs.csv'}")
print(f" - {REPORTS/'success_rate_by_tag.csv'}")
print(f" - {REPORTS/'success_rate_by_family_tag.csv'}")
print(f" - {REPORTS/'means_by_tag.csv'}")
print(f" - {REPORTS/'best_method_per_instance.csv'}")
print(f" - {REPORTS/'best_vs_baseline.csv'}")
print(f" - {FIGS/'success_rate_by_tag.png'}")
print(f" - {FIGS/'feasible_count_by_tag.png'}")
print(f" - {FIGS/'best_vs_base_box_by_family.png'}")
print(f" - {summary_txt}")
