from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RPTS = BASE / "data" / "reports"
FIGS = BASE / "data" / "figures"
RPTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Inputs produced earlier
champions_csv = RPTS / "ml_champions.csv"                 # chosen champion per instance
eval_csv       = RPTS / "step8_eval.csv"                  # eval for Champions + baselines (labels you used)

# ---- Load ----
ch = pd.read_csv(champions_csv)
ev = pd.read_csv(eval_csv)

# Normalize columns we rely on
need_cols = {"instance","method","ontime_p50","ontime_p95"}
if not need_cols.issubset(set(ch.columns)):
    # For safety if columns differ, fall back to eval’s p50/p95 later
    pass

# Which labels are baselines in step8_eval?
baseline_labels = {"SAA16-b0p3","Gamma1","Q120"}
champ_label     = "Champions"  # you used this when you ran evaluate_plans

# ---- Champion share by method ----
share = (ch["method"].value_counts().rename_axis("method").reset_index(name="count"))
share["share"] = (share["count"] / len(ch)).round(4)
share.to_csv(RPTS / "final_champion_share_by_method.csv", index=False)

# ---- Champion stats by family (from ml_champions.csv) ----
# If a 'family' column exists we’ll use it; if not, infer: first char(s) before digits (C/R/RC).
def infer_family(inst: str) -> str:
    inst = str(inst)
    if inst.startswith("RC"): return "RC"
    if inst.startswith("R"):  return "R"
    return "C"

if "family" not in ch.columns:
    ch["family"] = ch["instance"].map(infer_family)

by_family = (ch.groupby("family")[["distance","vehicles","ontime_p50","ontime_p95"]]
               .mean(numeric_only=True)
               .reset_index())
by_family.to_csv(RPTS / "final_champion_by_family.csv", index=False)

# ---- Aggregate eval: method means from step8_eval ----
# step8_eval columns include: instance, method(label), ontime_p05, ontime_p50, ontime_p95, ontime_mean, tard_mean, ...
agg_cols = ["ontime_p50","ontime_p95","ontime_mean","tard_mean"]
ev_means = (ev.groupby("method")[agg_cols].mean(numeric_only=True)
              .sort_values("ontime_p50", ascending=False)
              .reset_index())
ev_means.to_csv(RPTS / "final_eval_method_means.csv", index=False)

# ---- Champions vs best-baseline per instance (delta) ----
ev_pivot = (ev.pivot_table(index="instance", columns="method", values="ontime_p50", aggfunc="mean"))
# Keep only baselines for best-of comparison
best_baseline = ev_pivot[list(baseline_labels.intersection(ev_pivot.columns))].max(axis=1)
champ_p50     = ev_pivot.get(champ_label)

cmp = pd.DataFrame({
    "instance": best_baseline.index,
    "champion_p50": champ_p50.values,
    "best_baseline_p50": best_baseline.values
})
cmp["delta_p50"] = (cmp["champion_p50"] - cmp["best_baseline_p50"]).round(3)
cmp.sort_values("delta_p50", ascending=False, inplace=True)
cmp.to_csv(RPTS / "final_delta_champ_vs_best_baseline.csv", index=False)

# ---- Quick markdown summary ----
md = []
md += ["# Champion Pack — Summary\n"]
md += [f"- #instances: {len(ch)}"]
md += [f"- Champion share by method:\n"]
for _,r in share.iterrows():
    md += [f"  - {r['method']}: {int(r['count'])} ({r['share']*100:.1f}%)"]
md += ["\n## Averages by family (from champions.csv)"]
md += [by_family.to_markdown(index=False)]
md += ["\n## Method means from evaluation (step8_eval)"]
md += [ev_means.to_markdown(index=False)]
md += ["\n## Champion vs best baseline per instance (Δ p50 = champ - best baseline) — top 10"]
md += [cmp.head(10).to_markdown(index=False)]
(Path(RPTS / "final_summary.md")).write_text("\n".join(md), encoding="utf-8")

# ---- Optional: quick plots (distance vs ontime_p50 for champions; share by method) ----
try:
    import matplotlib.pyplot as plt

    # Scatter: distance vs ontime_p50 for champions
    plt.figure()
    ch.plot.scatter(x="distance", y="ontime_p50")
    plt.title("Champions: Distance vs On-Time p50")
    plt.savefig(FIGS / "champions_scatter_distance_vs_p50.png", bbox_inches="tight")
    plt.close()

    # Bar: champion share by method
    plt.figure()
    share.set_index("method")["count"].plot.bar()
    plt.title("Champion Count by Method")
    plt.ylabel("count")
    plt.savefig(FIGS / "champions_share_by_method.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    # plotting is optional; keep going
    pass

print("[OK] Wrote:")
print(f" - {RPTS/'final_champion_share_by_method.csv'}")
print(f" - {RPTS/'final_champion_by_family.csv'}")
print(f" - {RPTS/'final_eval_method_means.csv'}")
print(f" - {RPTS/'final_delta_champ_vs_best_baseline.csv'}")
print(f" - {RPTS/'final_summary.md'}")
print(f" - {FIGS/'champions_scatter_distance_vs_p50.png'} (optional)")
print(f" - {FIGS/'champions_share_by_method.png'} (optional)")
