from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")           # save-to-file backend
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "data" / "reports"
FIGS = BASE / "data" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# --- methods we’ll show on the slide (label -> folder) ---
METHODS = {
    "DET":  BASE / "data" / "solutions_ortools",
    "Q120": BASE / "data" / "solutions_quantile" / "m1.2_a0",
    "G1":   BASE / "data" / "solutions_gamma" / "g1_q1p645_hybrid",
    "G2":   BASE / "data" / "solutions_gamma" / "g2_q1p645_hybrid",
    "SAA16-b0p3": BASE / "data" / "solutions_saa" / "k16_b0p3",
}

# --- read on-time metrics (average across instances) ---
eval_by_method = REPORTS / "step8_eval_by_method.csv"
if eval_by_method.exists():
    em = pd.read_csv(eval_by_method)
else:
    # fallback: aggregate from instance-level eval
    inst_eval = pd.read_csv(REPORTS / "step8_eval.csv")
    em = inst_eval.groupby("method", as_index=False)[["ontime_mean","tard_mean"]].mean()

em = em.rename(columns={"ontime_mean":"ontime_avg", "tard_mean":"tard_avg"})

# --- read per-method distance & vehicles from each summary.csv ---
rows = []
for label, folder in METHODS.items():
    sumf = folder / "summary.csv"
    if sumf.exists():
        df = pd.read_csv(sumf)
        # be defensive on types
        for c in ("total_distance","vehicles"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        mean_dist = float(np.nanmean(df["total_distance"])) if "total_distance" in df else np.nan
        mean_veh  = float(np.nanmean(df["vehicles"])) if "vehicles" in df else np.nan
    else:
        mean_dist, mean_veh = np.nan, np.nan
    rows.append({"method": label, "mean_distance": mean_dist, "mean_vehicles": mean_veh})

costs = pd.DataFrame(rows)

# --- combine on-time with cost info ---
combo = costs.merge(em, left_on="method", right_on=em.columns[0], how="left")
combo = combo.rename(columns={combo.columns[3]: "ontime_avg", combo.columns[4]: "tard_avg"})
combo = combo[["method","mean_distance","mean_vehicles","ontime_avg","tard_avg"]]

# keep only the five, in nice display order
order = ["DET","Q120","SAA16-b0p3","G1","G2"]
combo = combo.set_index("method").reindex(order).reset_index()

# --- save a tidy CSV (handy for appendix) ---
out_csv = REPORTS / "tradeoff_summary_five_methods.csv"
combo.to_csv(out_csv, index=False)

# --- make the slide: cost (x) vs on-time (y) ---
plt.figure(figsize=(8,6))
x = combo["mean_distance"].to_numpy()
y = combo["ontime_avg"].to_numpy()

plt.scatter(x, y, s=120, alpha=0.9)

# annotate each point with method + vehicles
for i, row in combo.iterrows():
    label = row["method"]
    veh = row["mean_vehicles"]
    txt = f"{label}\n({veh:.1f} veh)"
    # slight offset to avoid overlap
    plt.annotate(txt, (row["mean_distance"], row["ontime_avg"]),
                 xytext=(6, 6), textcoords="offset points", fontsize=10)

plt.xlabel("Average total distance (cost)")
plt.ylabel("Average on-time service (%)")
plt.title("VRPTW Trade-off: Cost vs On-time (DET, Q120, SAA16-b0p3, G1, G2)")
plt.grid(True, alpha=0.25)
plt.tight_layout()

out_png = FIGS / "tradeoff_slide_cost_vs_ontime.png"
plt.savefig(out_png, dpi=180)
plt.close()

print("Wrote:")
print(f" - {out_csv}")
print(f" - {out_png}")
