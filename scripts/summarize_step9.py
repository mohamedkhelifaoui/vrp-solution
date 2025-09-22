from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
REP  = BASE / "data" / "reports"
FIG  = BASE / "data" / "figures"
REP.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---- Load solver summaries (distance/vehicles) ----
det = pd.read_csv(BASE / "data/solutions_ortools/summary.csv")
q12 = pd.read_csv(BASE / "data/solutions_quantile/m1.2_a0/summary.csv")
saa = pd.read_csv(BASE / "data/solutions_saa/k32_b0p5/summary.csv")

def tidy(df, label):
    df = df.copy()
    for c in ["vehicles","total_distance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["method"] = label
    return df[["instance","method","vehicles","total_distance","feasible"]]

det = tidy(det, "DET")
q12 = tidy(q12, "Q120")
saa = tidy(saa, "SAA32-b0p5")

wide = (pd.concat([det,q12,saa], ignore_index=True)
          .pivot(index="instance", columns="method", values=["vehicles","total_distance","feasible"]))

# Flatten columns like ('total_distance','DET') -> 'dist_DET'
wide.columns = [f"{a[:4]}_{b}" if a=="total_distance" else f"{a}_{b}" for a,b in wide.columns]
wide = wide.reset_index()

# Family label (C/R/RC)
wide["family"] = wide["instance"].str.extract(r"^([A-Z]+)")

# % distance change vs DET (negative = better)
for m in ["Q120","SAA32-b0p5"]:
    wide[f"pct_dist_vs_DET_{m}"] = 100 * (wide[f"tota_{m}"] - wide["tota_DET"]) / wide["tota_DET"]

# Save per-instance comparison
out_perinst = REP / "step9_method_table.csv"
wide.to_csv(out_perinst, index=False)

# ---- Load simulation evaluation (on-time %, tardiness) ----
eval_df = pd.read_csv(REP / "step9_eval.csv")  # created earlier via evaluate_plans.py + Copy-Item
# eval_df columns: instance, method, ontime_mean, ontime_p05, ontime_p50, ontime_p95, tard_mean

# Family on evaluation
eval_df["family"] = eval_df["instance"].str.extract(r"^([A-Z]+)")

# Averages by method
avg_by_method = (eval_df.groupby("method")[["ontime_mean","tard_mean"]]
                 .mean().sort_values("ontime_mean", ascending=False))
avg_by_method.to_csv(REP / "step9_eval_by_method_avg.csv")

# Averages by family & method
avg_by_fam = (eval_df.groupby(["family","method"])[["ontime_mean","tard_mean"]]
              .mean().reset_index())
avg_by_fam.to_csv(REP / "step9_eval_by_family_method.csv", index=False)

# ---- Plots ----
# 1) On-time vs. average distance (method-level)
#    Use distance means from solver summaries; align with eval_df methods
dist_means = (pd.concat([det,q12,saa])
                .groupby("method")["total_distance"].mean())
df_plot = avg_by_method.join(dist_means).reset_index()

plt.figure(figsize=(6,5))
for _, r in df_plot.iterrows():
    plt.scatter(r["total_distance"], r["ontime_mean"], s=80)
    plt.text(r["total_distance"]*1.002, r["ontime_mean"]*1.002, r["method"])
plt.xlabel("Mean total distance")
plt.ylabel("Mean on-time (%)")
plt.title("Step 9: Cost vs On-time (method level)")
plt.tight_layout()
plt.savefig(FIG / "step9_cost_vs_ontime_method.png", dpi=150)
plt.close()

# 2) On-time % by family (boxplot from per-instance metrics)
plt.figure(figsize=(8,5))
# keep only methods we compare
keep = eval_df[eval_df["method"].isin(["DET","Q120","SAA32-b0p5"])].copy()
# Combine family & method for grouping on one axis
keep["fam_m"] = keep["family"] + " | " + keep["method"]
order = sorted(keep["fam_m"].unique(), key=lambda x: (x.split(" | ")[0], x.split(" | ")[1]))
keep.boxplot(column="ontime_mean", by="fam_m", grid=False)
plt.suptitle("")
plt.xlabel("Family | Method")
plt.ylabel("On-time (%)")
plt.title("Step 9: On-time by family & method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIG / "step9_ontime_box_by_family_method.png", dpi=150)
plt.close()

print("Wrote:")
print(f" - {out_perinst}")
print(f" - {REP/'step9_eval_by_method_avg.csv'}")
print(f" - {REP/'step9_eval_by_family_method.csv'}")
print(f" - {FIG/'step9_cost_vs_ontime_method.png'}")
print(f" - {FIG/'step9_ontime_box_by_family_method.png'}")
