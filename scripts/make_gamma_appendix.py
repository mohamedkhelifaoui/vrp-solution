from pathlib import Path
import pandas as pd
import numpy as np

# Optional figs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
SOL = BASE / "data"
REPORTS = SOL / "reports"
FIGS = SOL / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ---- where each method’s summary.csv lives ----
METHODS = {
    "DET":          SOL / "solutions_ortools" / "summary.csv",
    "Q120":         SOL / "solutions_quantile" / "m1.2_a0" / "summary.csv",
    "SAA16-b0p3":   SOL / "solutions_saa" / "k16_b0p3" / "summary.csv",
    "SAA32-b0p5":   SOL / "solutions_saa" / "k32_b0p5" / "summary.csv",
    "SAA64-b0p7":   SOL / "solutions_saa" / "k64_b0p7" / "summary.csv",
    "G1":           SOL / "solutions_gamma" / "g1_q1p645_hybrid" / "summary.csv",
    "G2":           SOL / "solutions_gamma" / "g2_q1p645_hybrid" / "summary.csv",
}

def load_one(method, path):
    if not path.exists():
        print(f"!! Missing: {method} -> {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize expected columns
    # we expect: instance, vehicles, total_distance, feasible
    # coerce types
    for c in ["vehicles", "total_distance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "feasible" in df.columns:
        # unify to bool
        df["feasible"] = df["feasible"].astype(str).str.strip().str.lower().isin(["true","1","yes"])
    else:
        df["feasible"] = True  # assume feasible if column missing
    # family: C, R, or RC (from instance name)
    df["family"] = df["instance"].str.extract(r"^([A-Z]+)")
    df["method"] = method
    return df[["instance","family","method","vehicles","total_distance","feasible"]]

# load & stack
frames = [load_one(m, p) for m,p in METHODS.items()]
runs = pd.concat([f for f in frames if not f.empty], ignore_index=True)
if runs.empty:
    raise SystemExit("No summaries found. Check paths in METHODS dict.")

# per-family aggregates
def agg_table(df):
    grp = df.groupby(["family","method"], dropna=False)
    out = grp.agg(
        instances=("instance","nunique"),
        feasible_count=("feasible", lambda s: int((s==True).sum())),
        feasible_rate_pct=("feasible", lambda s: 100.0 * (s==True).mean()),
        avg_vehicles=("vehicles", "mean"),
        avg_distance=("total_distance", "mean"),
    ).reset_index()
    # nice ordering
    fam_order = ["C","R","RC"]
    meth_order = ["DET","Q120","SAA16-b0p3","SAA32-b0p5","SAA64-b0p7","G1","G2"]
    out["family"] = pd.Categorical(out["family"], fam_order, ordered=True)
    out["method"] = pd.Categorical(out["method"], meth_order, ordered=True)
    out = out.sort_values(["family","method"]).reset_index(drop=True)
    # round for presentation
    out["feasible_rate_pct"] = out["feasible_rate_pct"].round(1)
    out["avg_vehicles"] = out["avg_vehicles"].round(2)
    out["avg_distance"] = out["avg_distance"].round(1)
    return out

tbl_family = agg_table(runs)
tbl_overall = agg_table(runs.assign(family="ALL")).rename(columns={"family":"scope"})

# save CSVs
csv_family = REPORTS / "appendix_family_method_table.csv"
csv_overall = REPORTS / "appendix_overall_method_table.csv"
tbl_family.to_csv(csv_family, index=False)
tbl_overall.to_csv(csv_overall, index=False)

# write a markdown table for quick copy-paste in the report
md_path = REPORTS / "appendix_family_method_table.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# Appendix – Feasibility and Averages by Family & Method\n\n")
    f.write("**Per Family**\n\n")
    f.write("| Family | Method | #Inst | Feasible | Feasible % | Avg Veh | Avg Dist |\n")
    f.write("|:------:|:------:|------:|---------:|-----------:|--------:|---------:|\n")
    for _, r in tbl_family.iterrows():
        f.write(f"| {r['family']} | {r['method']} | {r['instances']} | "
                f"{r['feasible_count']} | {r['feasible_rate_pct']:.1f} | "
                f"{r['avg_vehicles']:.2f} | {r['avg_distance']:.1f} |\n")
    f.write("\n**Overall (ALL families combined)**\n\n")
    f.write("| Scope | Method | #Inst | Feasible | Feasible % | Avg Veh | Avg Dist |\n")
    f.write("|:-----:|:------:|------:|---------:|-----------:|--------:|---------:|\n")
    for _, r in tbl_overall.rename(columns={"scope":"family"}).iterrows():
        f.write(f"| {r['family']} | {r['method']} | {r['instances']} | "
                f"{r['feasible_count']} | {r['feasible_rate_pct']:.1f} | "
                f"{r['avg_vehicles']:.2f} | {r['avg_distance']:.1f} |\n")

print("Wrote:")
print(f" - {csv_family}")
print(f" - {csv_overall}")
print(f" - {md_path}")

# --------- Optional small figures for the appendix ----------
# Feasible % by family & method
pivot = tbl_family.pivot(index="method", columns="family", values="feasible_rate_pct")
pivot.plot(kind="bar")
plt.ylabel("Feasible rate (%)")
plt.title("Feasible rate by family & method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGS/"feasible_rate_by_family_method.png", dpi=150)
plt.close()

# Avg distance by family & method
pivot_d = tbl_family.pivot(index="method", columns="family", values="avg_distance")
pivot_d.plot(kind="bar")
plt.ylabel("Average distance")
plt.title("Avg distance by family & method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGS/"avg_distance_by_family_method.png", dpi=150)
plt.close()

# Avg vehicles by family & method
pivot_v = tbl_family.pivot(index="method", columns="family", values="avg_vehicles")
pivot_v.plot(kind="bar")
plt.ylabel("Average vehicles")
plt.title("Avg vehicles by family & method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGS/"avg_vehicles_by_family_method.png", dpi=150)
plt.close()

print("Also wrote figures to data/figures.")
