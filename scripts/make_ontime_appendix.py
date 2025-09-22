from pathlib import Path
import pandas as pd

# Optional figs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
REPORTS = DATA / "reports"
FIGS = DATA / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ---- Where each method summary.csv lives (same labels as you used in evaluate_plans) ----
METHODS = {
    "DET":          DATA / "solutions_ortools" / "summary.csv",
    "Q120":         DATA / "solutions_quantile" / "m1.2_a0" / "summary.csv",
    "SAA16-b0p3":   DATA / "solutions_saa" / "k16_b0p3" / "summary.csv",
    "SAA32-b0p5":   DATA / "solutions_saa" / "k32_b0p5" / "summary.csv",
    "SAA64-b0p7":   DATA / "solutions_saa" / "k64_b0p7" / "summary.csv",
    "G1":           DATA / "solutions_gamma" / "g1_q1p645_hybrid" / "summary.csv",
    "G2":           DATA / "solutions_gamma" / "g2_q1p645_hybrid" / "summary.csv",
}

EVAL = REPORTS / "step8_eval.csv"   # built by evaluate_plans.py

def load_method_summary(method, path):
    if not path.exists():
        print(f"!! Missing: {method} -> {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # expected columns: instance, vehicles, total_distance, feasible
    for c in ["vehicles", "total_distance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].astype(str).str.strip().str.lower().isin(["true","1","yes"])
    else:
        df["feasible"] = True
    df["family"] = df["instance"].str.extract(r"^([A-Z]+)")
    df["method"] = method
    return df[["instance","family","method","vehicles","total_distance","feasible"]]

def build_feas_table():
    frames = [load_method_summary(m, p) for m, p in METHODS.items()]
    runs = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if runs.empty:
        raise SystemExit("No summaries found. Check METHODS paths.")

    g = runs.groupby(["family","method"], dropna=False)
    feas = g.agg(
        instances=("instance","nunique"),
        feasible_count=("feasible", lambda s: int((s==True).sum())),
        feasible_rate_pct=("feasible", lambda s: 100.0 * (s==True).mean()),
        avg_vehicles=("vehicles","mean"),
        avg_distance=("total_distance","mean"),
    ).reset_index()

    # nice ordering
    fam_order = ["C","R","RC"]
    meth_order = ["DET","Q120","SAA16-b0p3","SAA32-b0p5","SAA64-b0p7","G1","G2"]
    feas["family"] = pd.Categorical(feas["family"], fam_order, ordered=True)
    feas["method"] = pd.Categorical(feas["method"], meth_order, ordered=True)
    feas = feas.sort_values(["family","method"]).reset_index(drop=True)

    # round
    feas["feasible_rate_pct"] = feas["feasible_rate_pct"].round(1)
    feas["avg_vehicles"] = feas["avg_vehicles"].round(2)
    feas["avg_distance"] = feas["avg_distance"].round(1)
    return feas

def build_ontime_table():
    if not EVAL.exists():
        raise SystemExit(f"Missing evaluation file: {EVAL}")
    ev = pd.read_csv(EVAL)
    # expected cols: instance, method, ontime_mean, ontime_p05, ontime_p50, ontime_p95, tard_mean
    ev["family"] = ev["instance"].astype(str).str.extract(r"^([A-Z]+)")
    # aggregate per family & method (mean across instances)
    ot = ev.groupby(["family","method"]).agg(
        ontime_mean=("ontime_mean","mean"),
        ontime_p50=("ontime_p50","mean"),
        ontime_p95=("ontime_p95","mean"),
        tard_mean=("tard_mean","mean"),
        n_inst=("instance","nunique"),
    ).reset_index()
    for c in ["ontime_mean","ontime_p50","ontime_p95"]:
        ot[c] = ot[c].round(2)
    ot["tard_mean"] = ot["tard_mean"].round(3)
    # order
    fam_order = ["C","R","RC"]
    meth_order = ["DET","Q120","SAA16-b0p3","SAA32-b0p5","SAA64-b0p7","G1","G2"]
    ot["family"] = pd.Categorical(ot["family"], fam_order, ordered=True)
    ot["method"] = pd.Categorical(ot["method"], meth_order, ordered=True)
    ot = ot.sort_values(["family","method"]).reset_index(drop=True)
    return ot

def main():
    feas = build_feas_table()
    ot   = build_ontime_table()

    # merge side-by-side
    combo = feas.merge(ot, on=["family","method"], how="outer")

    # save CSV
    out_csv = REPORTS / "appendix_family_method_with_ontime.csv"
    combo.to_csv(out_csv, index=False)

    # also write a Markdown table to paste into the appendix
    md = REPORTS / "appendix_family_method_with_ontime.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Appendix — Family × Method (Feasibility, Cost, Vehicles, On-time)\n\n")
        f.write("| Family | Method | #Inst | Feasible | Feas% | AvgVeh | AvgDist | OnTime-Mean | OnTime-p50 | OnTime-p95 | Tard-Mean |\n")
        f.write("|:------:|:------:|-----:|--------:|-----:|------:|-------:|-----------:|-----------:|-----------:|---------:|\n")
        for _, r in combo.iterrows():
            f.write(
                f"| {r['family']} | {r['method']} | "
                f"{int(r['instances']) if pd.notna(r['instances']) else ''} | "
                f"{int(r['feasible_count']) if pd.notna(r['feasible_count']) else ''} | "
                f"{r['feasible_rate_pct'] if pd.notna(r['feasible_rate_pct']) else ''} | "
                f"{r['avg_vehicles'] if pd.notna(r['avg_vehicles']) else ''} | "
                f"{r['avg_distance'] if pd.notna(r['avg_distance']) else ''} | "
                f"{r['ontime_mean'] if pd.notna(r['ontime_mean']) else ''} | "
                f"{r['ontime_p50'] if pd.notna(r['ontime_p50']) else ''} | "
                f"{r['ontime_p95'] if pd.notna(r['ontime_p95']) else ''} | "
                f"{r['tard_mean'] if pd.notna(r['tard_mean']) else ''} |\n"
            )

    print("Wrote:")
    print(f" - {out_csv}")
    print(f" - {md}")

    # quick figures (optional)
    piv = combo.pivot(index="method", columns="family", values="ontime_p50")
    piv.plot(kind="bar")
    plt.ylabel("On-time p50 (%)")
    plt.title("On-time p50 by family & method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS/"ontime_p50_by_family_method.png", dpi=150)
    plt.close()

    piv95 = combo.pivot(index="method", columns="family", values="ontime_p95")
    piv95.plot(kind="bar")
    plt.ylabel("On-time p95 (%)")
    plt.title("On-time p95 by family & method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS/"ontime_p95_by_family_method.png", dpi=150)
    plt.close()

    print("Also wrote figures to data/figures.")
    
if __name__ == "__main__":
    main()
