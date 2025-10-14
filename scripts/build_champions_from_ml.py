import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_all_snapshots(evals_dir: Path):
    rows = []
    for f in sorted(evals_dir.glob("*_eval.csv")):
        df = pd.read_csv(f)
        # Heuristic column normalization
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("instance","inst","name"): rename_map[c] = "instance"
            if lc.startswith("method"): rename_map[c] = "method"
            if lc in ("distance","total_distance","mean_distance"): rename_map[c] = "distance"
            if lc in ("vehicles","n_vehicles","mean_vehicles"): rename_map[c] = "vehicles"
            if lc in ("ontime_p50","p50_ontime","ontime50"): rename_map[c] = "ontime_p50"
            if lc in ("ontime_p95","p95_ontime","ontime95"): rename_map[c] = "ontime_p95"
        df = df.rename(columns=rename_map)
        keep = [c for c in ["instance","method","distance","vehicles","ontime_p50","ontime_p95"] if c in df.columns]
        rows.append(df[keep])
    if not rows:
        raise SystemExit("No *_eval.csv snapshots found. Run ml_autorun first.")
    return pd.concat(rows, ignore_index=True)

def select_champions(df, p50_target=95.0, prefer_p95=True):
    # keep best per-instance meeting target; tie-break distance -> vehicles
    champs = []
    for inst, g in df.groupby("instance"):
        cand = g[g["ontime_p50"] >= p50_target] if "ontime_p50" in g else g.copy()
        if cand.empty:
            # fallback: pick best by ontime_p50, then distance
            g2 = g.copy()
            g2 = g2.sort_values(
                by=[("ontime_p50" if "ontime_p50" in g2 else "distance"),
                    ("distance" if "ontime_p50" in g2 else "vehicles"),
                    "vehicles"],
                ascending=[False, True, True]
            )
            champs.append(g2.iloc[0])
            continue
        # optional tie-break by ontime_p95 first if available
        sort_keys = []
        sort_orders = []
        if prefer_p95 and "ontime_p95" in cand:
            sort_keys += ["ontime_p95"]; sort_orders += [False]
        sort_keys += ["distance","vehicles"]
        sort_orders += [True, True]
        cand = cand.sort_values(by=sort_keys, ascending=sort_orders)
        champs.append(cand.iloc[0])
    return pd.DataFrame(champs).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", default="data/ml_runs/evals")
    ap.add_argument("--p50_target", type=float, default=95.0)
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_all_snapshots(Path(args.evals_dir))

    df["family"] = df["instance"].str.extract(r'^(C|R|RC)', expand=False)
    df.to_csv(outdir / "ml_all_candidates.csv", index=False)

    champs = select_champions(df, p50_target=args.p50_target, prefer_p95=True)
    champs.to_csv(outdir / "ml_champions.csv", index=False)

    # coverage by method
    cov = (champs.groupby("method")
                 .agg(n_instances=("instance","count"),
                      mean_distance=("distance","mean"),
                      mean_vehicles=("vehicles","mean"),
                      mean_ontime_p50=("ontime_p50","mean"),
                      mean_ontime_p95=("ontime_p95","mean") if "ontime_p95" in champs else ("distance","mean"))
                 .reset_index()
          )
    cov.to_csv(outdir / "ml_champions_stats_by_method.csv", index=False)

    # family summary
    fam = (champs.groupby("family")
                 .agg(mean_distance=("distance","mean"),
                      mean_vehicles=("vehicles","mean"),
                      mean_ontime_p50=("ontime_p50","mean"),
                      mean_ontime_p95=("ontime_p95","mean") if "ontime_p95" in champs else ("distance","mean"),
                      n_instances=("instance","count"))
                 .reset_index())
    fam.to_csv(outdir / "ml_champions_by_family.csv", index=False)

    print("[OK] Wrote:")
    print(" -", outdir / "ml_all_candidates.csv")
    print(" -", outdir / "ml_champions.csv")
    print(" -", outdir / "ml_champions_stats_by_method.csv")
    print(" -", outdir / "ml_champions_by_family.csv")

if __name__ == "__main__":
    main()
