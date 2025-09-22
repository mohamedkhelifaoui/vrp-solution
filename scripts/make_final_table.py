# scripts/make_final_table.py
from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE = Path(__file__).resolve().parents[1]
REP  = BASE / "data" / "reports"
REP.mkdir(parents=True, exist_ok=True)

# Inputs produced in Steps 8–9
EVAL_CSV   = REP / "step8_eval.csv"               # on-time stats for every method per instance
CHAMPS_CSV = REP / "champions.csv"                # winners per instance from pick_champions.py
CHAMP_DIR  = BASE / "data" / "champions"          # copied JSON plans (one per instance)

DET_SUM    = BASE / "data/solutions_ortools/summary.csv"

def find_q120_summary():
    """Return the quantile (×1.2) summary file path."""
    p = BASE / "data/solutions_quantile/m1.2_a0/summary.csv"
    if p.exists():
        return p
    root = BASE / "data/solutions_quantile"
    if root.exists():
        for child in sorted(root.iterdir()):
            cand = child / "summary.csv"
            if child.is_dir() and cand.exists():
                return cand
    raise SystemExit("Could not find a quantile summary (expected data/solutions_quantile/.../summary.csv)")

Q120_SUM = find_q120_summary()

def read_method_summary(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        "instance": "instance",
        "vehicles": "vehicles",
        "total_distance": "distance",
        "runtime_sec": "runtime_sec",
        "feasible": "feasible",
    })
    if "runtime_sec" not in df.columns:
        df["runtime_sec"] = np.nan
    df["method"] = label
    return df[["instance","method","vehicles","distance","runtime_sec"]]

def read_champ_json(inst: str):
    """Read vehicles/distance (and runtime if present) from data/champions/<inst>.json."""
    jp = CHAMP_DIR / f"{inst}.json"
    if not jp.exists():
        return np.nan, np.nan, np.nan
    try:
        js = json.loads(jp.read_text(encoding="utf-8"))
        veh = js.get("vehicles", np.nan)
        dist = js.get("total_distance", np.nan)
        rt   = js.get("runtime_sec", np.nan)
        return veh, dist, rt
    except Exception:
        return np.nan, np.nan, np.nan

def main():
    # ---- load on-time stats (from evaluate_plans.py) ----
    if not EVAL_CSV.exists():
        raise SystemExit(f"Missing {EVAL_CSV}. Run evaluate_plans.py first.")
    evalv = pd.read_csv(EVAL_CSV)
    need_cols = {"instance","method","ontime_p50","ontime_p95"}
    if not need_cols.issubset(evalv.columns):
        raise SystemExit(f"{EVAL_CSV} must contain {need_cols}")
    eval_keep = evalv[["instance","method","ontime_p50","ontime_p95"]]

    # ---- load deterministic & quantile summaries ----
    det  = read_method_summary(DET_SUM,  "DET")
    q120 = read_method_summary(Q120_SUM, "Q120")

    # ---- load champions, fill vehicles/distance/runtime from JSON if missing ----
    if not CHAMPS_CSV.exists():
        raise SystemExit(f"Missing {CHAMPS_CSV}. Run pick_champions.py first.")
    ch = pd.read_csv(CHAMPS_CSV).copy()

    if "method" not in ch.columns:
        raise SystemExit("champions.csv is missing column: method")

    # create placeholder cols if absent
    for c in ("vehicles","distance","runtime_sec"):
        if c not in ch.columns:
            ch[c] = np.nan

    # fill from JSON where needed
    if not CHAMP_DIR.exists():
        raise SystemExit(f"Champions folder not found: {CHAMP_DIR}. Re-run pick_champions.py.")
    fill_rows = []
    for inst in ch["instance"].astype(str):
        veh, dist, rt = read_champ_json(inst)
        fill_rows.append((veh, dist, rt))
    fill_df = pd.DataFrame(fill_rows, columns=["veh_json","dist_json","rt_json"])
    ch = pd.concat([ch.reset_index(drop=True), fill_df], axis=1)
    ch["vehicles"]    = ch["vehicles"].fillna(ch["veh_json"])
    ch["distance"]    = ch["distance"].fillna(ch["dist_json"])
    if "runtime_sec" in ch.columns:
        ch["runtime_sec"] = ch["runtime_sec"].fillna(ch["rt_json"])
    else:
        ch["runtime_sec"] = ch["rt_json"]
    ch = ch.drop(columns=["veh_json","dist_json","rt_json"])

    # if still missing, at least proceed (will show NaNs)
    missing = ch[ch[["vehicles","distance"]].isna().any(axis=1)]
    if len(missing):
        print(f"NOTE: {len(missing)} champion rows still missing vehicles/distance (no JSON fields).")

    # rename for merge with eval
    ch = ch[["instance","method","vehicles","distance","runtime_sec"]] \
           .rename(columns={"method":"champ_method"})

    # merge with eval using the underlying champ method label
    ch_eval = ch.merge(
        eval_keep.rename(columns={"method":"champ_method"}),
        on=["instance","champ_method"],
        how="left"
    )
    ch_eval["method"] = "CHAMPION(" + ch_eval["champ_method"].astype(str) + ")"
    ch_eval = ch_eval.drop(columns=["champ_method"])

    # ---- join eval to DET/Q120 ----
    det  = det.merge(eval_keep, on=["instance","method"], how="left")
    q120 = q120.merge(eval_keep, on=["instance","method"], how="left")

    # ---- stack and finalize ----
    out = pd.concat([det, q120, ch_eval], ignore_index=True)
    out["family"] = out["instance"].str.extract(r"^([A-Z]+)")
    out = out[[
        "instance","family","method",
        "vehicles","distance","runtime_sec",
        "ontime_p50","ontime_p95"
    ]].sort_values(["family","instance","method"])

    csv_path  = REP / "final_comparison_table.csv"
    out.to_csv(csv_path, index=False)

    # optional XLSX
    try:
        import openpyxl  # pip install openpyxl (optional)
        xlsx_path = REP / "final_comparison_table.xlsx"
        out.to_excel(xlsx_path, index=False)
    except Exception:
        xlsx_path = None

    # per-family quick summary (nice to cite)
    fam = out.groupby(["family","method"]).agg(
        mean_distance=("distance","mean"),
        mean_vehicles=("vehicles","mean"),
        mean_p50=("ontime_p50","mean"),
        mean_p95=("ontime_p95","mean"),
        n_instances=("instance","nunique")
    ).reset_index()
    fam_path = REP / "final_comparison_family_summary.csv"
    fam.to_csv(fam_path, index=False)

    print("Wrote:")
    print(f" - {csv_path}")
    if xlsx_path:
        print(f" - {xlsx_path}")
    print(f" - {fam_path}")

if __name__ == "__main__":
    main()
