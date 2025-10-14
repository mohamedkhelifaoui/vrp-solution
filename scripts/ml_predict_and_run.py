from __future__ import annotations
import argparse, subprocess, json, shutil, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
TMP_DIR = BASE / "data" / "ml_runs" / "tmp"
REPORTS = BASE / "data" / "reports"

def run_cmd(cmd: list[str]):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def safe_methods_for_instance(inst: str) -> list[str]:
    d = TMP_DIR / inst
    if not d.exists():
        return []
    return [p.name for p in d.iterdir() if p.is_dir()]

def pick_champion(inst: str, prefer_method: str | None) -> tuple[str, Path]:
    """Return (chosen_method, json_path). Prefer `prefer_method` if its JSON exists,
       otherwise pick the method with max ontime_p50 from step8_eval.csv."""
    # try ML-predicted first
    if prefer_method:
        cand = next((p for p in (TMP_DIR/inst/prefer_method).glob("*.json")), None)
        if cand:
            return prefer_method, cand

    # fall back to best p50 from the fresh eval file
    eval_csv = REPORTS / "step8_eval.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing {eval_csv} (expected after ml_autorun).")

    df = pd.read_csv(eval_csv)
    sub = df[(df["instance"] == inst)].copy()
    if sub.empty:
        raise RuntimeError(f"No evaluation rows for instance {inst} in {eval_csv}")

    # keep only methods that actually produced a JSON
    have = set(safe_methods_for_instance(inst))
    sub = sub[sub["method"].isin(have)]
    if sub.empty:
        raise RuntimeError(f"No JSON plans found under {TMP_DIR/inst} for {inst}")

    sub = sub.sort_values(["ontime_p50", "ontime_p95"], ascending=[False, False])
    best_method = sub["method"].iloc[0]
    best_json = next((p for p in (TMP_DIR/inst/best_method).glob("*.json")), None)
    if not best_json:
        raise RuntimeError(f"Could not find JSON for chosen method {best_method} ({TMP_DIR/inst/best_method})")
    return best_method, best_json

def _rel_to_base(p: Path) -> str:
    """Return path relative to project BASE when possible; else absolute."""
    try:
        return str(p.resolve().relative_to(BASE))
    except Exception:
        return str(p.resolve())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to a single Solomon-style CSV (e.g., data/raw/NEW.csv)")
    ap.add_argument("--fallbacks", nargs="+", default=["Q120", "Gamma1-q1p645"],
                    help="Extra methods to run/evaluate alongside the ML pick")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_global", type=float, default=0.20)
    ap.add_argument("--cv_link", type=float, default=0.10)
    ap.add_argument("--outdir", default=str(BASE / "data" / "solutions_champions" / "new"))
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    inst = csv_path.stem  # instance name from filename
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Run ml_autorun for this single CSV (this both predicts and evaluates)
    cmd = [
        sys.executable, str(BASE / "scripts" / "ml_autorun.py"),
        "--csv", str(csv_path),
        "--K", str(args.K),
        "--seed", str(args.seed),
        "--cv_global", str(args.cv_global),
        "--cv_link", str(args.cv_link),
        "--fallbacks", *args.fallbacks
    ]
    run_cmd(cmd)

    # 2) Read prediction from ml_autorun_summary.csv if present
    pred_method = None
    summ_csv = BASE / "data" / "ml_runs" / "ml_autorun_summary.csv"
    if summ_csv.exists():
        try:
            summ = pd.read_csv(summ_csv)
            row = summ[summ["instance"] == inst]
            if not row.empty and "pred" in row.columns:
                pred_method = str(row.iloc[0]["pred"])
        except Exception as e:
            print(f"[WARN] Could not read {summ_csv}: {e}")

    # 3) Choose champion (prefer ML prediction; otherwise best p50 from eval)
    chosen_method, json_src = pick_champion(inst, pred_method)

    # 4) Copy winning plan + write a tiny summary
    out_json = outdir / f"{inst}__champion_{chosen_method}.json"
    shutil.copy2(json_src, out_json)

    # collect candidate metrics for a tiny per-instance summary
    eval_csv = REPORTS / "step8_eval.csv"
    df = pd.read_csv(eval_csv)
    cand = df[df["instance"] == inst].copy()
    cand = cand[["instance","method","distance","vehicles","ontime_p50","ontime_p95","tard_mean"]]

    summary = {
        "instance": inst,
        "predicted_method": pred_method,
        "chosen_method": chosen_method,
        "json": _rel_to_base(out_json),
    }
    pd.DataFrame([summary]).to_csv(outdir / f"{inst}__summary.csv", index=False)
    cand.to_csv(outdir / f"{inst}__candidates_eval.csv", index=False)

    print("\n[OK] Champion ready")
    print(" - instance       :", inst)
    print(" - predicted      :", pred_method)
    print(" - chosen         :", chosen_method)
    print(" - plan           :", out_json)
    print(" - candidates eval:", outdir / f"{inst}__candidates_eval.csv")
    print(" - summary        :", outdir / f"{inst}__summary.csv")

if __name__ == "__main__":
    main()
