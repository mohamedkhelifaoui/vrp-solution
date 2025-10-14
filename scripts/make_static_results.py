# scripts/make_static_results.py
from __future__ import annotations
from pathlib import Path
import json, math, argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RAW  = DATA / "raw"
FIG  = DATA / "figures"
REP  = DATA / "reports"
FIG.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

# Methods to include (static)
METHOD_DIRS: Dict[str, Path] = {
    "DET":          DATA / "solutions_ortools",
    "Q120":         DATA / "solutions_quantile" / "m1.2_a0",
    "SAA16-b0p3":   DATA / "solutions_saa" / "k16_b0p3",
    "SAA32-b0p5":   DATA / "solutions_saa" / "k32_b0p5",
    "SAA64-b0p7":   DATA / "solutions_saa" / "k64_b0p7",
    "Gamma1":       DATA / "solutions_gamma" / "g1_q1p645_hybrid",
    "Gamma2":       DATA / "solutions_gamma" / "g2_q1p645_hybrid",
}

def list_instances() -> List[str]:
    return sorted([p.stem for p in RAW.glob("*.csv")])

def read_instance(name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    csv_path = RAW / f"{name}.csv"
    df = pd.read_csv(csv_path).rename(columns={
        "CUST NO.":"cust","XCOORD.":"x","YCOORD.":"y",
        "DEMAND":"demand","READY TIME":"ready","DUE DATE":"due","SERVICE TIME":"service"
    })
    for c in ["cust","x","y","demand","ready","due","service"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    depot_idx = int(df.index[(df["demand"]==0) & (df["service"]==0)][0])
    if depot_idx != 0:
        order = [depot_idx] + [i for i in range(len(df)) if i != depot_idx]
        df = df.iloc[order].reset_index(drop=True)
    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = xs - xs[i]; dy = ys - ys[i]
        T[i, :] = np.hypot(dx, dy)
    return df, T

def find_method_json(method_label: str, instance: str) -> Optional[Path]:
    d = METHOD_DIRS.get(method_label)
    if not d: return None
    cand = d / f"{instance}.json"
    if cand.exists(): return cand
    hits = sorted(d.glob(f"{instance}*.json"))
    return hits[0] if hits else None

def load_plan(json_path: Path):
    js = json.loads(json_path.read_text(encoding="utf-8"))
    routes = js.get("routes") or js.get("routes_by_vehicle") or js.get("solution",{}).get("routes") or []
    vehicles = js.get("vehicles")
    distance = js.get("total_distance", js.get("distance", js.get("dist")))
    feasible = js.get("feasible", True)
    return routes, vehicles, distance, feasible

def _coerce_routes_to_nodes(routes, df: pd.DataFrame):
    cust_to_node = {int(df.loc[i, "cust"]): i for i in range(len(df))}
    n = len(df); out=[]
    for r in routes:
        rr=[]
        for v in r:
            try: v_int = int(v)
            except: continue
            if v_int == 0: continue
            if 0 <= v_int < n: rr.append(v_int)
            else:
                j = cust_to_node.get(v_int)
                if j is not None: rr.append(j)
        out.append(rr)
    return out

def _route_distance(base_time: np.ndarray, routes_nodes) -> float:
    dist = 0.0
    for r in routes_nodes:
        prev = 0
        for j in r:
            dist += base_time[prev, j]
            prev = j
        dist += base_time[prev, 0]
    return float(dist)

def _lognormal_params_for_cv(cv: float):
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = -0.5 * sigma2
    return mu, sigma

def simulate_ontime(df: pd.DataFrame, base_time: np.ndarray, routes, K=200, seed=42, cv_global=0.20, cv_link=0.10):
    routes_nodes = _coerce_routes_to_nodes(routes, df)
    n = len(df)
    ready = df["ready"].to_numpy(); due = df["due"].to_numpy(); service = df["service"].to_numpy()
    rng = np.random.default_rng(seed)
    muG, sG = _lognormal_params_for_cv(cv_global)
    muL, sL = _lognormal_params_for_cv(cv_link)
    G = rng.lognormal(mean=muG, sigma=sG, size=K)
    E = rng.lognormal(mean=muL, sigma=sL, size=(K, n, n))
    ontime_ratio=[]
    for s in range(K):
        T = base_time * G[s] * E[s]
        ontime=0; total=0
        for r in routes_nodes:
            t=0.0; prev=0
            for j in r:
                t += T[prev, j]
                if t < ready[j]: t = ready[j]
                tard = max(0.0, t - due[j])
                if tard == 0: ontime += 1
                total += 1
                t += service[j]
                prev = j
            t += T[prev, 0]
        ontime_ratio.append(0.0 if total==0 else 100.0*ontime/total)
    return dict(
        ontime_p50 = float(np.percentile(ontime_ratio, 50)),
        ontime_p95 = float(np.percentile(ontime_ratio, 95)),
        ontime_mean = float(np.mean(ontime_ratio)),
        n_customers = int(sum(len(r) for r in routes_nodes)),
    )

def family_of(inst: str) -> str:
    s = inst.upper()
    if s.startswith("RC"): return "RC"
    if s.startswith("R"):  return "R"
    return "C"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_global", type=float, default=0.20)
    ap.add_argument("--cv_link", type=float, default=0.10)
    args = ap.parse_args()

    rows=[]
    instances = list_instances()
    for inst in instances:
        df, T = read_instance(inst)
        for meth, mdir in METHOD_DIRS.items():
            jp = find_method_json(meth, inst)
            if jp is None or not jp.exists():
                rows.append(dict(instance=inst, family=family_of(inst), method=meth,
                                 feasible=False, distance=np.nan, vehicles=np.nan,
                                 ontime_p50=np.nan, ontime_p95=np.nan, ontime_mean=np.nan))
                continue
            routes, veh, dist, feas = load_plan(jp)
            rnodes = _coerce_routes_to_nodes(routes, df)
            if dist is None:
                dist = _route_distance(T, rnodes)
            if veh is None:
                veh = len([r for r in rnodes if len(r)>0])
            if not feas or len(rnodes)==0:
                rows.append(dict(instance=inst, family=family_of(inst), method=meth,
                                 feasible=False, distance=np.nan, vehicles=veh,
                                 ontime_p50=np.nan, ontime_p95=np.nan, ontime_mean=np.nan))
                continue
            stats = simulate_ontime(df, T, routes, K=args.K, seed=args.seed,
                                    cv_global=args.cv_global, cv_link=args.cv_link)
            rows.append(dict(instance=inst, family=family_of(inst), method=meth,
                             feasible=True, distance=float(dist), vehicles=int(veh),
                             ontime_p50=stats["ontime_p50"], ontime_p95=stats["ontime_p95"],
                             ontime_mean=stats["ontime_mean"]))

    step8 = pd.DataFrame(rows)
    step8.to_csv(REP/"step8_eval.csv", index=False)

    # Success rate by tag (=method)
    sr = (step8.groupby("method")["feasible"]
                .agg(["mean","sum","count"])
                .rename(columns={"mean":"success_rate","sum":"feasible_count","count":"total"}))
    sr["success_rate"] = (sr["success_rate"]*100).round(1)
    sr.reset_index().to_csv(REP/"success_rate_by_tag.csv", index=False)

    # Means by tag (feasible only)
    feas = step8[step8["feasible"]==True]
    means = (feas.groupby("method")[["distance","vehicles","ontime_p50","ontime_p95","ontime_mean"]]
                  .mean().round(2).reset_index())
    means.to_csv(REP/"means_by_tag.csv", index=False)

    # Best-of-sweep vs DET per instance (distance improvement), shown by family
    # pick best (min distance) among methods for each instance
    det = feas[feas["method"]=="DET"][["instance","distance"]].rename(columns={"distance":"det"})
    best = (feas.sort_values(["instance","distance"])
                 .drop_duplicates("instance", keep="first")
                 [["instance","family","distance"]]
                 .rename(columns={"distance":"best"}))
    comp = best.merge(det, on="instance", how="inner")
    comp["impr_pct"] = 100.0*(comp["det"] - comp["best"]) / comp["det"]
    # Boxplot by family
    plt.figure(figsize=(10,5))
    data = [comp[comp["family"]==fam]["impr_pct"].dropna() for fam in ["C","R","RC"]]
    plt.boxplot(data, labels=["C","R","RC"])
    plt.title("Best-of-sweep vs Baseline: % distance improvement (positive = better)")
    plt.xlabel("Family"); plt.ylabel("% improvement")
    plt.tight_layout()
    plt.savefig(FIG/"best_vs_base_box_by_family.png", dpi=150)

    print("[OK] Wrote:")
    print(" -", REP/"step8_eval.csv")
    print(" -", REP/"success_rate_by_tag.csv")
    print(" -", REP/"means_by_tag.csv")
    print(" -", FIG/"best_vs_base_box_by_family.png")

if __name__ == "__main__":
    main()
