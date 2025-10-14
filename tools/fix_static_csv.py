from pathlib import Path
import pandas as pd
import numpy as np

SRC  = Path("data/reports/static_master_results.csv")
OUT  = Path("data/reports/static_master_results_clean.csv")

PRIMARY_KEY = ["instance","method"]

def pick_champion(df_inst):
    # tie-breaks: p50 desc, p95 desc, distance asc, vehicles asc
    s = df_inst.sort_values(
        by=["ontime_p50","ontime_p95","distance","vehicles"],
        ascending=[False, False, True, True]
    ).copy()
    s["is_champion"] = False
    if len(s):
        s.iloc[0, s.columns.get_loc("is_champion")] = True
    return s

def fix_block(df):
    # numeric casts
    num_cols = [
        "n_customers","time_limit_s","vehicle_cost","K_eval","seed_eval","cv_global_eval","cv_link_eval",
        "vehicles","distance","runtime_s","ontime_mean","ontime_p50","ontime_p95","tard_mean",
        "sla_threshold","seed_build","gap_to_det_pct","gap_to_best_pct"
    ]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # booleans
    for c in ["feasible","meets_SLA","is_champion"]:
        if c in df.columns: df[c] = df[c].astype(bool)

    # 1) enforce percentile order & bounds
    df["ontime_p50"] = df["ontime_p50"].clip(lower=0, upper=100)
    df["ontime_p95"] = df["ontime_p95"].clip(lower=0, upper=100)
    swap = df["ontime_p95"] < df["ontime_p50"]
    df.loc[swap, ["ontime_p50","ontime_p95"]] = df.loc[swap, ["ontime_p95","ontime_p50"]].to_numpy()

    # 2) tard_mean ~ 0 if p50 or p95 are 100
    mask_full = (df["ontime_p50"] >= 100.0) | (df["ontime_p95"] >= 100.0)
    df.loc[mask_full, "tard_mean"] = 0.0

    # 3) recompute meets_SLA strictly
    df["meets_SLA"] = (df["ontime_p50"] >= df["sla_threshold"])

    # 4) recompute gaps per-instance
    out = []
    for inst, g in df.groupby("instance", as_index=False):
        # distance DET (baseline)
        det = g[g["method"]=="DET"]["distance"]
        dist_det = det.iloc[0] if len(det) else np.nan
        min_dist = g["distance"].min()

        g = g.copy()
        g["gap_to_det_pct"]  = 100.0 * (g["distance"] - dist_det) / dist_det if pd.notna(dist_det) else np.nan
        g["gap_to_best_pct"] = 100.0 * (g["distance"] - min_dist) / min_dist

        # 5) set unique champion with the tie-break rule
        g["is_champion"] = False
        g = pick_champion(g)

        out.append(g)
    df2 = pd.concat(out, ignore_index=True)

    # 6) round pretty
    for c in ["distance","runtime_s","ontime_mean","ontime_p50","ontime_p95","tard_mean","gap_to_det_pct","gap_to_best_pct"]:
        if c in df2.columns:
            df2[c] = df2[c].astype(float).round(2)

    return df2

def main():
    if not SRC.exists():
        raise SystemExit(f"Missing input: {SRC}")
    raw = pd.read_csv(SRC)
    if raw.duplicated(PRIMARY_KEY).any():
        dups = raw[raw.duplicated(PRIMARY_KEY, keep=False)].sort_values(PRIMARY_KEY)
        print("[WARN] Duplicate (instance, method) rows:\n", dups[PRIMARY_KEY])
        raw = raw.drop_duplicates(PRIMARY_KEY, keep="first")
    clean = fix_block(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT} with {len(clean)} rows")

if __name__ == "__main__":
    main()
