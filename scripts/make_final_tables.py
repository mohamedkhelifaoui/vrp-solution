# scripts/make_final_tables.py
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = Path(__file__).resolve().parents[1]
REPORTS  = BASE / "data" / "reports"
FIGS     = BASE / "data" / "figures"
FINAL    = BASE / "data" / "final" / "current"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# load baseline (your greedy/first deterministic) and the chosen final tag
base = pd.read_csv(BASE / "data/solutions/summary.csv")
final = pd.read_csv(FINAL / "summary.csv")

# be defensive on types
for df in (base, final):
    for c in ["vehicles", "total_distance", "horizon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

base = base.rename(columns={
    "vehicles":"vehicles_base",
    "total_distance":"dist_base",
    "feasible":"feasible_base"
})
final = final.rename(columns={
    "vehicles":"vehicles_final",
    "total_distance":"dist_final",
    "feasible":"feasible_final"
})

df = base.merge(final[["instance","vehicles_final","dist_final","feasible_final"]],
                on="instance", how="inner")
df["family"] = df["instance"].str.extract(r"^([A-Z]+)")
df["delta_dist"] = df["dist_final"] - df["dist_base"]
df["pct_improve"] = 100 * (df["dist_base"] - df["dist_final"]) / df["dist_base"]
df["delta_veh"] = df["vehicles_final"] - df["vehicles_base"]

# save table
out_csv = REPORTS / "final_vs_baseline.csv"
df.to_csv(out_csv, index=False)

# quick text summary
summary_txt = REPORTS / "step7_summary.txt"
with summary_txt.open("w", encoding="utf-8") as f:
    f.write("=== Step 7: Final deterministic configuration vs. Baseline ===\n")
    f.write(f"Final folder: {FINAL}\n")
    f.write(f"Table: {out_csv}\n\n")
    f.write("-- Overall stats --\n")
    f.write(df[["pct_improve","delta_dist","delta_veh"]]
            .describe().to_string(float_format=lambda x: f"{x:.3f}") + "\n\n")
    f.write("-- Per family mean improvements --\n")
    fam = df.groupby("family").agg(
        mean_pct_improve=("pct_improve","mean"),
        mean_delta_dist=("delta_dist","mean"),
        mean_delta_veh=("delta_veh","mean"),
        n=("instance","count")
    )
    f.write(fam.to_string(float_format=lambda x: f"{x:.3f}") + "\n\n")
    top10 = df.sort_values("pct_improve", ascending=False).head(10)
    f.write("-- Top 10 instances by % improvement --\n")
    f.write(top10[["instance","family","vehicles_base","vehicles_final",
                   "dist_base","dist_final","pct_improve"]]
            .to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")

# figures
plt.figure(figsize=(8,5))
df.boxplot(column="pct_improve", by="family", grid=False)
plt.suptitle("")
plt.title("Final vs Baseline: % distance improvement (positive is better)")
plt.xlabel("Family"); plt.ylabel("% improvement")
plt.tight_layout()
plt.savefig(FIGS / "final_vs_base_box_by_family.png", dpi=150)
plt.close()

plt.figure(figsize=(8,5))
veh = df.groupby("family")["delta_veh"].mean().reindex(["C","R","RC"])
veh.plot(kind="bar")
plt.title("Final vs Baseline: mean vehicle change by family")
plt.xlabel("Family"); plt.ylabel("Δ vehicles (final - base)")
plt.tight_layout()
plt.savefig(FIGS / "final_vs_base_delta_vehicles_by_family.png", dpi=150)
plt.close()

print("Wrote:")
print(f" - {out_csv}")
print(f" - {summary_txt}")
print(f" - {FIGS/'final_vs_base_box_by_family.png'}")
print(f" - {FIGS/'final_vs_base_delta_vehicles_by_family.png'}")
