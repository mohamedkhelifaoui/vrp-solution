from pathlib import Path
import argparse, shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "data" / "reports"
FIGS = BASE / "data" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def read_summary(dir_path: Path, label: str) -> pd.DataFrame:
    s = pd.read_csv(dir_path / "summary.csv")
    # be robust on types
    for c in ("vehicles","total_distance"):
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    s["method"] = label
    return s[["instance","method","vehicles","total_distance"]]

def collect_plans(dir_path: Path) -> dict:
    """Return set of available plan JSONs by instance."""
    out = {}
    for p in dir_path.glob("*.json"):
        out[p.stem] = p
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="Solution directories")
    ap.add_argument("--labels", nargs="+", required=True, help="Method labels (same order as --dirs)")
    ap.add_argument("--target", type=float, default=99.0, help="Target on-time %")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    labels = args.labels
    if len(dirs) != len(labels):
        raise SystemExit("len(--dirs) must equal len(--labels)")

    # 1) Load common-scenario evaluation (already produced by evaluate_plans.py)
    eval_df = pd.read_csv(REPORTS / "step8_eval.csv")
    # Keep the metric we'll use for SLA (mean on-time across 200 scenarios)
    eval_df = eval_df[["instance","method","ontime_mean"]]

    # 2) Load each method's deterministic summary (distance, vehicles)
    rows = []
    plan_paths = {}
    for d, lab in zip(dirs, labels):
        rows.append(read_summary(d, lab))
        plan_paths[lab] = collect_plans(d)
    sums = pd.concat(rows, ignore_index=True)

    # 3) Merge eval + summaries
    merged = eval_df.merge(sums, on=["instance","method"], how="inner")

    # 4) Pick champion per instance:
    #    - filter methods with ontime_mean >= target
    #    - choose minimal total_distance; tie-breaker: vehicles
    champions = []
    for inst, g in merged.groupby("instance"):
        g_ok = g[g["ontime_mean"] >= args.target]
        if g_ok.empty:
            # no method meets SLA: pick best ontime, break ties by distance
            g2 = g.sort_values(["ontime_mean","total_distance","vehicles"],
                               ascending=[False, True, True]).head(1)
            status = "NO_METHOD_MET_TARGET"
        else:
            g2 = g_ok.sort_values(["total_distance","vehicles"], ascending=[True, True]).head(1)
            status = "OK"
        row = g2.iloc[0].to_dict()
        row["status"] = status
        champions.append(row)
    ch = pd.DataFrame(champions)

    # 5) Save champions table
    out_csv = REPORTS / "champions.csv"
    ch.to_csv(out_csv, index=False)

    # 6) Copy chosen JSON plans to data/champions/
    dest_dir = BASE / "data" / "champions"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for _, r in ch.iterrows():
        inst = r["instance"]; lab = r["method"]
        src = plan_paths.get(lab, {}).get(inst)
        if src and src.exists():
            shutil.copy2(src, dest_dir / f"{inst}.json")
        else:
            missing.append((inst, lab))

    if missing:
        (REPORTS / "champions_missing.txt").write_text(
            "\n".join([f"{i}  {m}" for i,m in missing]), encoding="utf-8"
        )

    # 7) Simple plots
    by_method = ch.groupby("method").agg(
        mean_ontime=("ontime_mean","mean"),
        mean_dist=("total_distance","mean"),
        n=("instance","count")
    ).sort_values("mean_dist")
    by_method.to_csv(REPORTS / "champions_stats_by_method.csv")

    plt.figure(figsize=(8,6))
    plt.scatter(by_method["mean_dist"], by_method["mean_ontime"], s=140)
    for m, r in by_method.iterrows():
        plt.text(r["mean_dist"]+5, r["mean_ontime"]-0.8, m)
    plt.xlabel("Mean total distance (champions)")
    plt.ylabel("Mean on-time % (champions)")
    plt.title("Champion set: cost vs on-time by method")
    plt.tight_layout()
    plt.savefig(FIGS / "champions_cost_vs_ontime.png", dpi=150)
    plt.close()

    print("Wrote:")
    print(f" - {out_csv}")
    print(f" - {REPORTS/'champions_stats_by_method.csv'}")
    print(f" - {FIGS/'champions_cost_vs_ontime.png'}")
    print(f" - Copied final JSON plans to {dest_dir}")

if __name__ == "__main__":
    main()
