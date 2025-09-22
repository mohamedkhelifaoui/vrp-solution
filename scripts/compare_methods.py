from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")             # non-interactive backend for saving files
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
reports = BASE / "data" / "reports"
figdir = BASE / "data" / "figures"
reports.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)

# Load summaries
base = pd.read_csv(BASE / "data/solutions/summary.csv")
ort  = pd.read_csv(BASE / "data/solutions_ortools/summary.csv")

# Ensure numeric types (robust if CSVs were saved with strings)
for c in ["vehicles", "total_distance", "horizon"]:
    base[c] = pd.to_numeric(base[c], errors="coerce")
    ort[c]  = pd.to_numeric(ort[c], errors="coerce")

b = base.rename(columns={
    "vehicles": "vehicles_base",
    "total_distance": "dist_base",
    "feasible": "feasible_base"
})
o = ort.rename(columns={
    "vehicles": "vehicles_ort",
    "total_distance": "dist_ort",
    "feasible": "feasible_ort"
})

# Merge & compute deltas
df = b.merge(o[["instance","vehicles_ort","dist_ort","feasible_ort"]], on="instance", how="left")
df["family"] = df["instance"].str.extract(r"^([A-Z]+)")  # C, R, or RC
df["delta_veh"] = df["vehicles_ort"] - df["vehicles_base"]
df["delta_dist"] = df["dist_ort"] - df["dist_base"]
df["pct_distance_change"] = 100 * df["delta_dist"] / df["dist_base"]

# Save comparison table
out_csv = reports / "method_comparison.csv"
df.to_csv(out_csv, index=False)

# Plot: % distance change by family (negative is better)
valid = df.dropna(subset=["pct_distance_change"])
ax = valid.boxplot(column="pct_distance_change", by="family", grid=False)
plt.title("OR-Tools vs Baseline: % distance change by family (negative is better)")
plt.suptitle("")
plt.xlabel("Family")
plt.ylabel("% change")
plt.tight_layout()
plt.savefig(figdir / "pct_distance_change_by_family.png", dpi=150)
plt.close()

# Plot: count of NO SOLUTION by family
order = ["C", "R", "RC"]
ns = df[df["feasible_ort"] != True].groupby("family").size().reindex(order).fillna(0).astype(int)
ns.to_csv(reports / "no_solution_counts.csv", header=["count"])
ns.plot(kind="bar")
plt.title("OR-Tools NO SOLUTION counts by family")
plt.xlabel("Family")
plt.ylabel("# instances")
plt.tight_layout()
plt.savefig(figdir / "no_solution_by_family.png", dpi=150)
plt.close()

print("Wrote:")
print(f" - {out_csv}")
print(f" - {figdir/'pct_distance_change_by_family.png'}")
print(f" - {figdir/'no_solution_by_family.png'}")
print(f" - {reports/'no_solution_counts.csv'}")

# Show top wins (negative % = OR-Tools shorter distance)
wins = valid.sort_values("pct_distance_change").head(8)[
    ["instance","family","vehicles_base","vehicles_ort","dist_base","dist_ort","pct_distance_change"]
]
print("\nTop distance improvements (negative %):")
print(wins.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
