# scripts/ml_pick_config.py
import sys, pickle, argparse, re
from pathlib import Path
import numpy as np, pandas as pd

DEFAULT_MODEL_PATH = Path("models/meta_selector.pkl")
RAW_DIR = Path("data/raw")
BASELINE_SUMMARY = Path("data/solutions_ortools/summary.csv")  # optional

def fam_horiz(inst: str):
    inst = inst.upper()
    if inst.startswith("RC"):
        fam = "RC"; digits = re.sub(r"\D", "", inst)
    else:
        fam = inst[0] if inst and inst[0] in ["C","R"] else "?"
        digits = re.sub(r"\D", "", inst)
    horiz = 2 if (digits and digits[0]=="2") else 1
    return fam, horiz

def load_solomon(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {}
    for k in list(df.columns):
        ku = k.strip().upper()
        if ku.startswith("CUST"): rename[k]="cust"
        elif ku.startswith("XCOORD"): rename[k]="x"
        elif ku.startswith("YCOORD"): rename[k]="y"
        elif ku=="DEMAND": rename[k]="demand"
        elif ku.startswith("READY"): rename[k]="ready"
        elif ku.startswith("DUE"): rename[k]="due"
        elif ku.startswith("SERVICE"): rename[k]="service"
    df = df.rename(columns=rename)
    for c in ["cust","x","y","demand","ready","due","service"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

def features_from_df(df: pd.DataFrame):
    mask_cust = ~((df["demand"]==0) & (df["service"]==0))
    x, y = df.loc[mask_cust,"x"].to_numpy(), df.loc[mask_cust,"y"].to_numpy()
    n = len(x)
    cx, cy = (x.mean() if n else 0.0), (y.mean() if n else 0.0)
    r2 = ((x-cx)**2+(y-cy)**2).mean() if n>0 else 0.0
    if n>1:
        dmat = np.sqrt((x[:,None]-x[None,:])**2+(y[:,None]-y[None,:])**2)
        dmat[dmat==0]=np.inf
        nn_mean = np.min(dmat,axis=1).mean()
        iu = np.triu_indices(n,1)
        pair_mean = dmat[iu].mean()
        cluster_ratio = nn_mean/(pair_mean+1e-9)
    else:
        nn_mean=pair_mean=cluster_ratio=0.0
    w = (df["due"]-df["ready"]).clip(lower=0)[mask_cust]
    serv = df.loc[mask_cust,"service"]
    return {
        "n_customers": n,
        "mean_sq_radius": r2,
        "nn_mean": nn_mean,
        "pair_mean": pair_mean,
        "cluster_ratio": cluster_ratio,
        "win_mean": float(w.mean()) if len(w) else 0.0,
        "win_med": float(w.median()) if len(w) else 0.0,
        "win_std": float(w.std(ddof=0)) if len(w) else 0.0,
        "win_p10": float(w.quantile(0.10)) if len(w) else 0.0,
        "win_p25": float(w.quantile(0.25)) if len(w) else 0.0,
        "win_p75": float(w.quantile(0.75)) if len(w) else 0.0,
        "service_mean": float(serv.mean()) if len(serv) else 0.0,
        "service_p75": float(serv.quantile(0.75)) if len(serv) else 0.0,
        "tight_share": float((w<=w.quantile(0.25)).mean()) if len(w) else 0.0,
    }

def baseline_lookup(instance: str):
    if not BASELINE_SUMMARY.exists(): return {}
    df = pd.read_csv(BASELINE_SUMMARY)
    cols = {c.lower(): c for c in df.columns}
    inst_col = cols.get("instance") or cols.get("name") or list(df.columns)[0]
    row = df[df[inst_col].astype(str).str.upper()==instance.upper()]
    if row.empty: return {}
    out={}
    for k in row.columns:
        lk=k.lower()
        if "distance" in lk: out["baseline_distance"]=float(row.iloc[0][k])
        if "vehicle" in lk: out["baseline_vehicles"]=float(row.iloc[0][k])
    return out

def label_to_dir(method_label: str) -> str:
    m = (method_label or "").upper()
    if m=="DET": return "data/solutions_ortools"
    if m.startswith("Q"):
        fac = m[1:]
        try:
            j = float(fac)/100.0
            return f"data/solutions_quantile/m{j:.1f}_a0".replace(".0_","_")
        except Exception:
            return "data/solutions_quantile/m1.2_a0"
    if m.startswith("SAA"):
        return "data/solutions_saa/" + m.replace("SAA","k").replace("B","b")
    if m in ("G1","G2"):
        qtag = "q1p645"
        return f"data/solutions_gamma/g{m[-1]}_{qtag}_hybrid"
    if m.startswith("GAMMA1"):
        return "data/solutions_gamma/g1_q1p645_hybrid"
    if m.startswith("GAMMA2"):
        return "data/solutions_gamma/g2_q1p645_hybrid"
    return "data/solutions_ortools"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=str, help="Instance id like RC104")
    ap.add_argument("--csv", type=str, help="Path to raw Solomon CSV")
    ap.add_argument("--model", type=str, help="Path to meta_selector.pkl (optional)")
    args = ap.parse_args()

    # Locate model
    model_path = Path(args.model) if args.model else DEFAULT_MODEL_PATH
    if not model_path.exists():
        # Graceful fallback so callers can proceed
        print(f"[WARN] Missing model {model_path}. Falling back to DET.", file=sys.stderr)
        print("Predicted method : DET")
        return

    # Locate CSV & derive instance name
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[ERROR] {csv_path} not found", file=sys.stderr)
            print("Predicted method : DET")
            return
        instance = csv_path.stem.upper()
    elif args.instance:
        instance = args.instance.upper()
        csv_path = RAW_DIR / f"{instance}.csv"
        if not csv_path.exists():
            alts = list(RAW_DIR.glob(f"{instance}.*"))
            if not alts:
                print(f"[ERROR] Raw CSV not found for instance {instance}", file=sys.stderr)
                print("Predicted method : DET")
                return
            csv_path = alts[0]
    else:
        print("Use --instance C101 or --csv data/raw/C101.csv", file=sys.stderr)
        print("Predicted method : DET")
        return

    # Build features row
    fam, horiz = fam_horiz(instance)
    df = load_solomon(csv_path)
    feats = {"family": fam, "horizon": int(horiz)}
    feats.update(features_from_df(df))
    feats.update(baseline_lookup(instance))
    X = pd.DataFrame([feats])

    # Load model pipeline
    try:
        with open(model_path, "rb") as f:
            pack = pickle.load(f)
        pipe = pack["pipeline"]
        feature_cols = pack["feature_cols"]
        classes = pack["classes_"]
    except ModuleNotFoundError as e:
        # e.g., lightgbm not installed; degrade gracefully
        print(f"[WARN] {e}. Falling back to DET.", file=sys.stderr)
        print("Predicted method : DET")
        return
    except Exception as e:
        # Any other unpickle error (e.g., sklearn version mismatch)
        print(f"[WARN] Could not load model: {e}. Falling back to DET.", file=sys.stderr)
        print("Predicted method : DET")
        return

    # align columns (fill any missing numeric cols with 0)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    # Predict
    yhat = pipe.predict(X)[0]
    method = classes[yhat]
    sol_dir = label_to_dir(method)

    print(f"Instance         : {instance}")
    print(f"Predicted method : {method}")
    print(f"Solutions dir    : {sol_dir}")
    print("\n# Suggested commands:")
    print(f"python scripts/evaluate_plans.py --dirs \"{sol_dir}\" "
          f"--labels {method} --K 200 --seed 42 --cv_global 0.20 --cv_link 0.10")

if __name__ == "__main__":
    main()
