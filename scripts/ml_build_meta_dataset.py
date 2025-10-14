import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/ml"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CHAMPIONS_CSV = Path("data/reports/champions.csv")         # labels
BASELINE_SUMMARY = Path("data/solutions_ortools/summary.csv")  # optional features
OUT_CSV = OUT_DIR / "meta_train.csv"

# --- utilities ---------------------------------------------------------------
def parse_instance_name(p: Path) -> str:
    # assume filenames like C101.csv, R211.csv, RC107.csv (case-insensitive ok)
    base = p.stem.upper()
    # handle RCxxx vs Cxxx/Rxxx
    if base.startswith("RC"):
        return base
    if base.startswith("C") or base.startswith("R"):
        return base
    return base  # fallback

def fam_horiz(inst: str):
    inst = inst.upper()
    if inst.startswith("RC"):
        fam = "RC"; num = re.sub(r"\D", "", inst)  # digits
    else:
        fam = inst[0] if inst[0] in ["C","R"] else "?"
        num = re.sub(r"\D", "", inst)
    horiz = "2" if num and num[0] == "2" else "1"
    return fam, horiz

def load_solomon(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize headers
    cols = {c.strip().upper(): c for c in df.columns}
    # remap to canonical short names
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

def euclid_features(df: pd.DataFrame):
    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    # depot assumed unique row with demand=0 and service=0; exclude from stats
    mask_cust = ~( (df["demand"]==0) & (df["service"]==0) )
    xs_c, ys_c = xs[mask_cust], ys[mask_cust]
    n_cust = len(xs_c)
    cx, cy = xs_c.mean(), ys_c.mean()
    r2 = ((xs_c-cx)**2 + (ys_c-cy)**2).mean()  # mean squared radius
    # nearest neighbor (cheap) on customers only
    if n_cust <= 1:
        nn_mean = 0.0; pair_mean = 0.0
    else:
        dmat = np.sqrt( (xs_c[:,None]-xs_c[None,:])**2 + (ys_c[:,None]-ys_c[None,:])**2 )
        dmat[dmat==0] = np.inf
        nn_mean = np.min(dmat, axis=1).mean()
        iu = np.triu_indices(n_cust, k=1)
        pair_mean = dmat[iu].mean()
    return dict(n_customers=n_cust, mean_sq_radius=r2, nn_mean=nn_mean, pair_mean=pair_mean,
                cluster_ratio=(nn_mean/(pair_mean+1e-9) if pair_mean>0 else 0.0))

def window_features(df: pd.DataFrame):
    width = (df["due"] - df["ready"]).clip(lower=0)
    # exclude depot if due==ready==0
    mask_cust = ~( (df["demand"]==0) & (df["service"]==0) )
    w = width[mask_cust]
    serv = df.loc[mask_cust, "service"]
    return dict(
        win_mean=w.mean(), win_med=w.median(), win_std=w.std(ddof=0),
        win_p10=w.quantile(0.10), win_p25=w.quantile(0.25), win_p75=w.quantile(0.75),
        service_mean=serv.mean(), service_p75=serv.quantile(0.75),
        tight_share=float((w<=w.quantile(0.25)).mean())
    )

def baseline_lookup(instance: str, baseline_df: pd.DataFrame):
    # Try to find a row for this instance; column names may vary
    # accept case-insensitive matches on 'instance'
    if baseline_df is None: return {}
    cols = {c.lower(): c for c in baseline_df.columns}
    inst_col = None
    for cand in ["instance", "name", "id"]:
        if cand in cols: inst_col = cols[cand]; break
    if inst_col is None: return {}
    row = baseline_df[baseline_df[inst_col].astype(str).str.upper()==instance.upper()]
    if row.empty: return {}
    out = {}
    for k in row.columns:
        lk = k.lower()
        if "distance" in lk: out["baseline_distance"] = float(row.iloc[0][k])
        if "vehicle" in lk:  out["baseline_vehicles"] = float(row.iloc[0][k])
    return out

def read_champions(path: Path):
    if not path.exists():
        sys.exit(f"[ERROR] Missing labels file: {path}")
    df = pd.read_csv(path)
    # try to find columns
    cols = {c.lower(): c for c in df.columns}
    inst_col = cols.get("instance") or cols.get("name") or list(df.columns)[0]
    method_col = cols.get("method") or cols.get("label") or cols.get("algo") or list(df.columns)[1]
    out = df[[inst_col, method_col]].copy()
    out.columns = ["instance", "method"]
    out["instance"] = out["instance"].astype(str).str.upper()
    out["method"] = out["method"].astype(str)
    return out

# --- main -------------------------------------------------------------------
def main():
    champs = read_champions(CHAMPIONS_CSV)
    baseline_df = pd.read_csv(BASELINE_SUMMARY) if BASELINE_SUMMARY.exists() else None

    rows = []
    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        inst = parse_instance_name(csv_path)
        df = load_solomon(csv_path)
        fam, horiz = fam_horiz(inst)
        feats = {"instance": inst, "family": fam, "horizon": int(horiz)}
        feats.update(euclid_features(df))
        feats.update(window_features(df))
        feats.update(baseline_lookup(inst, baseline_df))
        rows.append(feats)

    X = pd.DataFrame(rows)
    data = champs.merge(X, on="instance", how="inner")
    if data.empty:
        sys.exit("[ERROR] No overlap between champions.csv and raw instances.")
    data.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_CSV} with shape {data.shape}")

if __name__ == "__main__":
    main()
