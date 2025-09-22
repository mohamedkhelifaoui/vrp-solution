#!/usr/bin/env python3
"""
Quick QA: check headers, try to detect depot, basic time-window sanity.
Outputs JSON report even if configs/data_paths.yaml is missing.
"""
import json
from pathlib import Path
import pandas as pd

HEADER_MAP = {
    "id":      ["CUST_NO", "CUST NO.", "ID", "CustNo", "Customer"],
    "x":       ["XCOORD", "XCOORD.", "X", "X_COORD"],
    "y":       ["YCOORD", "YCOORD.", "Y", "Y_COORD"],
    "demand":  ["DEMAND", "Demand"],
    "a":       ["READY_TIME", "READY TIME", "READY", "earliest", "a"],
    "b":       ["DUE_DATE", "DUE DATE", "DUE", "latest", "b"],
    "service": ["SERVICE_TIME", "SERVICE TIME", "SERVICETIME", "service"],
}

def load_paths():
    # Try YAML if present; otherwise use defaults
    cfg_path = Path("configs/data_paths.yaml")
    raw_dir = Path("data/raw")
    out_json = Path("data/qa_report.json")
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            raw_dir = Path(cfg.get("raw_dir", raw_dir))
            out_json = Path(cfg.get("qa_report", out_json))
        except Exception as e:
            print(f"[warn] Could not read YAML: {e}. Using defaults.")
    return raw_dir, out_json

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    raw_dir, out_json = load_paths()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {"files": []}

    for p in sorted(raw_dir.glob("*.csv")):
        entry = {"file": p.name, "issues": [], "summary": {}}
        try:
            df = pd.read_csv(p)
            cols = {k: find_col(df, v) for k, v in HEADER_MAP.items()}
            missing = [k for k, v in cols.items() if v is None]
            if missing:
                entry["issues"].append(f"Missing expected columns: {missing}")
            else:
                tmp = df.rename(columns={
                    cols["id"]: "id",
                    cols["x"]: "x",
                    cols["y"]: "y",
                    cols["demand"]: "demand",
                    cols["a"]: "a",
                    cols["b"]: "b",
                    cols["service"]: "service",
                }).copy()
                for c in ["id","x","y","demand","a","b","service"]:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                nrows_start = len(tmp)
                tmp = tmp.dropna(subset=["id","x","y","demand","a","b","service"])
                n_nan = nrows_start - len(tmp)

                depot_rows = tmp[(tmp["demand"]==0) & (tmp["service"]==0)]
                if len(depot_rows) != 1:
                    entry["issues"].append(f"Depot rows != 1 (found {len(depot_rows)})")

                if (tmp["a"] > tmp["b"]).any():
                    entry["issues"].append("Some rows have READY_TIME > DUE_DATE")

                entry["summary"] = {
                    "n_rows": int(len(tmp)),
                    "n_nan_rows_dropped": int(n_nan),
                    "n_depot_candidates": int(len(depot_rows)),
                    "a_min": float(tmp["a"].min()),
                    "a_max": float(tmp["a"].max()),
                    "b_min": float(tmp["b"].min()),
                    "b_max": float(tmp["b"].max()),
                    "total_demand": float(tmp["demand"].sum()),
                }
        except Exception as e:
            entry["issues"].append(f"Exception while reading: {e}")

        report["files"].append(entry)

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"QA report written: {out_json}")

if __name__ == "__main__":
    main()
