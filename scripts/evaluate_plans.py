from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parents[1]
REPORTS = BASE / "data" / "reports"
FIGS    = BASE / "data" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ----------------- Helpers -----------------

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Make the Solomon headers robust to dots/spaces/case."""
    def norm(s: str) -> str:
        return s.strip().upper().replace(" ", "").replace(".", "")
    name_map = {c: norm(c) for c in df.columns}

    # find columns by normalized name
    def pick(*cands):
        for c in df.columns:
            if name_map[c] in cands:
                return c
        return None

    colmap = {
        "cust":   pick("CUSTNO", "CUSTOMER", "CUSTOMERID", "ID"),
        "x":      pick("XCOORD", "X", "LON", "LONGITUDE"),
        "y":      pick("YCOORD", "Y", "LAT", "LATITUDE"),
        "demand": pick("DEMAND", "Q", "DEMANDQTY"),
        "ready":  pick("READYTIME", "READY", "EARLY", "TWSTART"),
        "due":    pick("DUEDATE", "DUE", "LATE", "TWEND"),
        "service":pick("SERVICETIME", "SERVICE", "SERV")
    }
    missing = [k for k, v in colmap.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. "
                         f"Got columns: {list(df.columns)}")

    df = df.rename(columns={
        colmap["cust"]: "cust",
        colmap["x"]: "x",
        colmap["y"]: "y",
        colmap["demand"]: "demand",
        colmap["ready"]: "ready",
        colmap["due"]: "due",
        colmap["service"]: "service",
    })
    return df

def read_instance(csv_path: Path):
    """Read a Solomon-style instance and return:
       df (with depot row at index 0), Euclidean base_time matrix, and cust_id->node mapping."""
    raw = pd.read_csv(csv_path)
    df = _normalize_headers(raw)

    # robust numeric conversion
    for c in ["cust","x","y","demand","ready","due","service"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # depot detection: demand==0 & service==0
    depot_rows = df.index[(df["demand"] == 0) & (df["service"] == 0)].tolist()
    if not depot_rows:
        raise ValueError(f"No depot row found in {csv_path.name} (demand=0 & service=0).")
    depot_idx = int(depot_rows[0])

    # move depot to index 0 if needed
    if depot_idx != 0:
        order = [depot_idx] + [i for i in range(len(df)) if i != depot_idx]
        df = df.iloc[order].reset_index(drop=True)

    # distances = times (Solomon convention)
    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    base_time = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = xs - xs[i]; dy = ys - ys[i]
        base_time[i, :] = np.hypot(dx, dy)

    # robust cust-id mapping (ints)
    cust_ids = pd.to_numeric(df["cust"], errors="coerce").fillna(-1).astype(int).tolist()
    cust_to_node = {int(cid): i for i, cid in enumerate(cust_ids)}
    return df, base_time, cust_to_node

def lognormal_params_for_cv(cv: float):
    """For a lognormal with E=1 and given CV, sigma^2 = ln(1+cv^2), mu = -0.5*sigma^2."""
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = -0.5 * sigma2
    return mu, sigma

def make_scenarios(n: int, K: int, seed: int, cv_global=0.20, cv_link=0.10):
    rng = np.random.default_rng(seed)
    muG, sG = lognormal_params_for_cv(cv_global)
    muL, sL = lognormal_params_for_cv(cv_link)
    G = rng.lognormal(mean=muG, sigma=sG, size=K)           # (K,)
    E = rng.lognormal(mean=muL, sigma=sL, size=(K, n, n))   # (K,n,n)
    return G, E

def _map_id_to_node(cust_id: int, cust_id_to_node: dict, n: int, inst: str) -> int:
    """Safe mapping with fallbacks and clear error messages."""
    cid = int(cust_id)
    if cid in cust_id_to_node:
        return cust_id_to_node[cid]
    # fallback: sometimes routes use 1..N row positions
    if 1 <= cid <= n:
        return cid - 1
    raise KeyError(f"Unknown customer id {cust_id} in routes for instance {inst}")

def simulate_on_time(df, base_time, routes, cust_id_to_node, instance_name: str,
                     K=100, seed=42, cv_global=0.20, cv_link=0.10):
    """Simulate on-time performance under multiplicative global/link noise."""
    n = len(df)
    ready   = df["ready"].to_numpy()
    due     = df["due"].to_numpy()
    service = df["service"].to_numpy()

    # Edge case: empty or None routes (no solution)
    if not routes or all(len(r) == 0 for r in routes):
        return {
            "ontime_mean": np.nan, "ontime_p05": np.nan, "ontime_p50": np.nan, "ontime_p95": np.nan,
            "tard_mean": np.nan, "n_customers": 0, "K": K
        }

    G, E = make_scenarios(n, K, seed, cv_global, cv_link)
    ontime_ratio = []
    mean_tard = []

    for s in range(K):
        T = base_time * G[s] * E[s]
        ontime = 0
        tard_acc = []

        for r in routes:
            if not r:
                continue
            t = 0.0
            prev = 0  # depot index
            for cust_id in r:
                j = _map_id_to_node(cust_id, cust_id_to_node, n, instance_name)
                t += T[prev, j]
                if t < ready[j]:
                    t = ready[j]
                tard = max(0.0, t - due[j])
                if tard == 0.0:
                    ontime += 1
                tard_acc.append(tard)
                t += service[j]
                prev = j
            # back to depot (travel only)
            t += T[prev, 0]

        total = sum(len(r) for r in routes)
        if total > 0:
            ontime_ratio.append(100.0 * ontime / total)
            mean_tard.append(np.mean(tard_acc) if tard_acc else 0.0)
        else:
            ontime_ratio.append(np.nan)
            mean_tard.append(np.nan)

    ontime_arr = np.array(ontime_ratio, dtype=float)
    tard_arr   = np.array(mean_tard, dtype=float)

    return {
        "ontime_mean": float(np.nanmean(ontime_arr)),
        "ontime_p05":  float(np.nanpercentile(ontime_arr, 5)),
        "ontime_p50":  float(np.nanpercentile(ontime_arr, 50)),
        "ontime_p95":  float(np.nanpercentile(ontime_arr, 95)),
        "tard_mean":   float(np.nanmean(tard_arr)),
        "n_customers": int(sum(len(r) for r in routes)),
        "K": int(K),
    }

def load_plan_json(p: Path):
    """Return (instance, routes, feasible_flag)."""
    js = json.loads(p.read_text(encoding="utf-8"))
    inst    = js.get("instance", p.stem)
    routes  = js.get("routes", [])
    feas    = js.get("feasible", True)
    return inst, routes, bool(feas)

def evaluate_dir(dir_path: Path, label: str, K: int, seed: int,
                 cv_global: float, cv_link: float) -> pd.DataFrame:
    rows = []
    for p in sorted(dir_path.glob("*.json")):
        inst, routes, feas = load_plan_json(p)

        # Skip infeasible/empty plans gracefully
        if not feas or not routes:
            rows.append({
                "instance": inst, "method": label,
                "ontime_mean": np.nan, "ontime_p05": np.nan, "ontime_p50": np.nan, "ontime_p95": np.nan,
                "tard_mean": np.nan, "n_customers": 0, "K": K
            })
            continue

        csv_path = BASE / "data" / "raw" / f"{inst}.csv"
        if not csv_path.exists():
            # no matching instance, skip
            continue

        df, base_time, custmap = read_instance(csv_path)
        metrics = simulate_on_time(
            df, base_time, routes, custmap, inst,
            K=K, seed=seed, cv_global=cv_global, cv_link=cv_link
        )
        rows.append({"instance": inst, "method": label, **metrics})
    return pd.DataFrame(rows)

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="One or more solution directories to evaluate")
    ap.add_argument("--labels", nargs="+",
                    help="Optional method labels (same length/order as --dirs)")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_global", type=float, default=0.20)
    ap.add_argument("--cv_link", type=float, default=0.10)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    labels = args.labels if args.labels and len(args.labels) == len(dirs) else [d.name for d in dirs]

    dfs = []
    for d, lab in zip(dirs, labels):
        dfs.append(evaluate_dir(d, lab, args.K, args.seed, args.cv_global, args.cv_link))
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    out_csv = REPORTS / "step8_eval.csv"
    out.to_csv(out_csv, index=False)

    # Overall mean by method (skip all-NaN safely)
    if not out.empty:
        agg_cols = ["ontime_mean", "tard_mean"]
        bym = (out.groupby("method")[agg_cols]
                   .mean(numeric_only=True)
                   .sort_values("ontime_mean", ascending=False))
        bym.to_csv(REPORTS / "step8_eval_by_method.csv")
    else:
        (REPORTS / "step8_eval_by_method.csv").write_text("")

    print("Wrote:")
    print(f" - {out_csv}")
    print(f" - {REPORTS/'step8_eval_by_method.csv'}")

if __name__ == "__main__":
    main()
