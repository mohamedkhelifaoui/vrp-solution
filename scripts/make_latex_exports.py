#!/usr/bin/env python3
# make_latex_exports.py — Generate LaTeX-aligned CSVs & PNGs from VRPTW results
# Usage examples (at bottom) or: python make_latex_exports.py -h

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import argparse, io, zipfile, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================ Config / Conventions ============================

EXPECTED_COLS = [
    "instance","family","n_customers","method","method_family","tag","metaheuristic",
    "time_limit_s","vehicle_cost","K_eval","seed_eval","cv_global_eval","cv_link_eval",
    "feasible","vehicles","distance","runtime_s","ontime_mean","ontime_p50","ontime_p95",
    "tard_mean","sla_metric","sla_threshold","meets_SLA","is_champion","status",
    "version_tag","seed_build","gap_to_det_pct","gap_to_best_pct"
]

PREF_METHOD_ORDER = [
    "DET","Q110","Q120","Q130",
    "SAA16-b0p3","SAA32-b0p5","SAA64-b0p7",
    "Gamma1-q1p645","Gamma2-q1p645"
]


# ============================== I/O & Utilities ==============================

def method_sort_key(m: str) -> tuple:
    try:
        return (0, PREF_METHOD_ORDER.index(m))
    except ValueError:
        return (1, m)

def robust_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV and coerce known types; warn if schema deviates."""
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        # Fallback tolerant parser
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        print("[WARN] CSV had malformed rows; used tolerant parser (skipped bad lines).", file=sys.stderr)

    # If expected columns exist, great. If not, we’ll proceed but warn.
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing expected columns: {missing}", file=sys.stderr)

    # Coerce dtypes
    num_cols = [
        "n_customers","time_limit_s","vehicle_cost","K_eval","seed_eval",
        "cv_global_eval","cv_link_eval","vehicles","distance","runtime_s",
        "ontime_mean","ontime_p50","ontime_p95","tard_mean","sla_threshold",
        "seed_build","gap_to_det_pct","gap_to_best_pct"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["feasible","meets_SLA","is_champion"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true","1","t","yes"])

    # Duplicate key heads-up (not fatal)
    if set(["instance","method"]).issubset(df.columns) and df.duplicated(["instance","method"]).any():
        print("[WARN] Duplicate (instance, method) rows detected.", file=sys.stderr)

    return df

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_csv(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)

def _buf_png(fig, dpi=200) -> bytes:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    b.seek(0)
    return b.read()

def _save_png_bytes(png_bytes: bytes, out_path: Path) -> None:
    with open(out_path, "wb") as f:
        f.write(png_bytes)


# ============================== Filtering (CRN) ==============================

def apply_scenario_filters(df: pd.DataFrame,
                           K: Optional[int],
                           cvg: Optional[float],
                           cvl: Optional[float],
                           seed: Optional[int],
                           tol: float = 1e-9) -> pd.DataFrame:
    """Filter dataset by CRN controls if provided (K_eval, cv_global_eval, cv_link_eval, seed_eval)."""
    out = df.copy()
    if K is not None and "K_eval" in out:             out = out[np.isclose(out["K_eval"], K, atol=tol, rtol=0)]
    if cvg is not None and "cv_global_eval" in out:   out = out[np.isclose(out["cv_global_eval"], cvg, atol=1e-8, rtol=0)]
    if cvl is not None and "cv_link_eval" in out:     out = out[np.isclose(out["cv_link_eval"], cvl, atol=1e-8, rtol=0)]
    if seed is not None and "seed_eval" in out:       out = out[np.isclose(out["seed_eval"], seed, atol=0, rtol=0)]
    return out


# ============================== Champion Selection ==============================

def pick_instance_champion(sub: pd.DataFrame,
                           metric: str,
                           threshold: float,
                           feasible_only: bool = False) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Exact logic (slides/demo/CSV aligned):
      1) meets_SLA_local := ontime_p50 >= T  (or p95)
      2) candidates := rows passing SLA; else least violator(s) by max on-time metric
      3) tie-break: distance asc → vehicles asc → runtime_s asc
      4) champion := first after sort
    Optionally filter to feasible rows before (1).
    Returns: (scored rows with flags, champion row, rationale string)
    """
    work = sub.copy()
    if feasible_only and "feasible" in work:
        work = work[work["feasible"] == True].copy()

    if metric == "p50":
        work["meets_SLA_local"] = work["ontime_p50"] >= threshold
        metric_label = "On-time p50"
        metric_col = "ontime_p50"
    else:
        work["meets_SLA_local"] = work["ontime_p95"] >= threshold
        metric_label = "On-time p95"
        metric_col = "ontime_p95"

    rationale = [f"SLA target: {metric_label} ≥ {threshold:.1f}%."]

    candidates = work[work["meets_SLA_local"]]
    if candidates.empty:
        # least violator(s): maximize on-time metric
        max_metric = work[metric_col].max()
        candidates = work[work[metric_col] == max_metric].copy()
        rationale.append("No plan meets the SLA; select the least violating (highest on-time), then minimum distance/vehicles/runtime.")
    else:
        rationale.append("Among SLA-feasible plans, choose minimum distance, then vehicles, then runtime.")

    candidates = candidates.sort_values(["distance","vehicles","runtime_s"], ascending=True)
    champ_row = candidates.iloc[0]

    work["is_champion_local"] = (work["method"] == champ_row["method"])
    return work, champ_row, " ".join(rationale)


# ============================== Tables (CSVs) ==============================

def make_tables_for_latex(adf: pd.DataFrame,
                          inst_for_screen: str,
                          metric: str = "p50",
                          threshold: float = 95.0,
                          feasible_only_for_screen: bool = False) -> Dict[str, pd.DataFrame]:
    # champions.csv — rows with is_champion==True in original CSV (not re-picked)
    champs = adf[adf.get("is_champion", False) == True].copy()
    champions_rows = champs[[
        "instance","method","method_family","vehicles","distance",
        "ontime_p50","ontime_p95","meets_SLA"
    ]].sort_values("instance")

    # champions_stats_by_method.csv — grouped over champions only
    champions_by_method = (
        champs.groupby(["method","method_family"], as_index=False)
             .agg(champions=("instance","count"),
                  avg_distance=("distance","mean"),
                  avg_vehicles=("vehicles","mean"),
                  avg_p50=("ontime_p50","mean"))
             .sort_values(["champions","avg_distance"], ascending=[False, True])
    )

    # summary_by_family.csv — grouped over all rows in the active filter
    summary_by_family = (
        adf.groupby("method_family", as_index=False)
           .agg(rows=("instance","count"),
                feasible_rate=("feasible", lambda s: 100*np.mean(s)),
                sla_rate=("meets_SLA", lambda s: 100*np.mean(s)),
                dist_mean=("distance","mean"),
                veh_mean=("vehicles","mean"),
                p50_mean=("ontime_p50","mean"))
           .sort_values("method_family")
    )

    # summary_by_method.csv — preferred method order then alpha
    summary_by_method = (
        adf.groupby(["method","method_family"], as_index=False)
           .agg(rows=("instance","count"),
                feasible_rate=("feasible", lambda s: 100*np.mean(s)),
                sla_rate=("meets_SLA", lambda s: 100*np.mean(s)),
                dist_mean=("distance","mean"),
                veh_mean=("vehicles","mean"),
                p50_mean=("ontime_p50","mean"))
           .sort_values("method", key=lambda s: s.map(method_sort_key))
    )

    # [INST]_scored_[metric].csv — instance screen using champion logic
    sub = adf[adf["instance"] == inst_for_screen].copy()
    if sub.empty:
        raise ValueError(f"Instance '{inst_for_screen}' not found in filtered dataset.")

    scored, champ, _ = pick_instance_champion(
        sub=sub, metric=metric, threshold=threshold, feasible_only=feasible_only_for_screen
    )
    instance_screen = (
        scored[[
            "instance","method","method_family","feasible",
            "meets_SLA_local","is_champion_local",
            "vehicles","distance","runtime_s",
            "ontime_p50","ontime_p95","gap_to_det_pct","gap_to_best_pct",
            "tag","metaheuristic"
        ]]
        .rename(columns={
            "meets_SLA_local": "meets_SLA(target)",
            "is_champion_local":"is_champion(target)"
        })
        .sort_values(["meets_SLA(target)","distance"], ascending=[False, True])
    )

    return {
        "champions.csv": champions_rows,
        "champions_stats_by_method.csv": champions_by_method,
        "summary_by_family.csv": summary_by_family,
        "summary_by_method.csv": summary_by_method,
        f"{inst_for_screen}_scored_{metric}.csv": instance_screen,
    }


# ============================== Figures (PNGs) ==============================

def fig_frontier_p50(adf: pd.DataFrame, inst: str) -> bytes:
    sub = adf[(adf["instance"]==inst)].copy()
    fig, ax = plt.subplots()
    if sub.empty:
        ax.text(0.5,0.5,"No data", ha="center")
        return _buf_png(fig)

    methods = sub["method"].unique().tolist()
    veh_vals = sorted(sub["vehicles"].astype(float).dropna().astype(int).unique().tolist())
    marker_pool = ["o","s","^","D","P","X","v","<",">"]
    veh_to_marker = {v: marker_pool[i % len(marker_pool)] for i, v in enumerate(veh_vals)}
    cmap = plt.cm.get_cmap("tab10", max(len(methods), 1))
    meth_to_color = {m: cmap(i % 10) for i, m in enumerate(methods)}

    for _, r in sub.iterrows():
        veh = int(r["vehicles"]) if not pd.isna(r["vehicles"]) else veh_vals[0]
        ax.scatter(r["distance"], r["ontime_p50"],
                   marker=veh_to_marker.get(veh, "o"),
                   color=meth_to_color.get(r["method"], "C0"), alpha=0.9)

    mhandles = [plt.Line2D([0],[0], marker='o', ls='None', color=meth_to_color[m], label=m) for m in methods]
    vhandles = [plt.Line2D([0],[0], marker=veh_to_marker[v], ls='None', color='gray', label=f'{v} veh') for v in veh_vals]
    if mhandles:
        leg1 = ax.legend(handles=mhandles, title="Method", loc="lower right")
        ax.add_artist(leg1)
    if vhandles:
        ax.legend(handles=vhandles, title="Vehicles", loc="lower left")

    ax.set_xlabel("Distance"); ax.set_ylabel("On-time p50 (%)"); ax.grid(alpha=0.3)
    return _buf_png(fig)

def fig_distance_by_method(adf: pd.DataFrame, inst: str) -> bytes:
    sub = adf[(adf["instance"]==inst)].copy()
    sub = sub.sort_values("method", key=lambda s: s.map(method_sort_key))
    fig, ax = plt.subplots()
    if sub.empty:
        ax.text(0.5,0.5,"No data", ha="center")
        return _buf_png(fig)
    x = np.arange(len(sub))
    ax.bar(x, sub["distance"])
    ax.set_xticks(x); ax.set_xticklabels(sub["method"], rotation=30, ha="right")
    ax.set_xlabel("Method"); ax.set_ylabel("Distance")
    return _buf_png(fig)

def fig_improve_vs_det_box(adf: pd.DataFrame) -> bytes:
    det = adf[adf["method"]=="DET"][["instance","distance"]].rename(columns={"distance":"distance_DET"})
    joined = adf.merge(det, on="instance", how="left")
    joined["improv_pct_vs_DET"] = 100.0 * (joined["distance_DET"] - joined["distance"]) / joined["distance_DET"]
    imp = joined[joined["method"]!="DET"]
    groups = sorted(imp["method_family"].dropna().unique())
    data = [imp[imp["method_family"]==fam]["improv_pct_vs_DET"].dropna() for fam in groups]

    fig, ax = plt.subplots()
    if any(len(x) for x in data):
        ax.boxplot(data, labels=groups)
    else:
        ax.text(0.5,0.5,"No data", ha="center")
    ax.set_ylabel("Improvement vs DET (%)"); ax.grid(alpha=0.3)
    return _buf_png(fig)

def fig_sla_sweep(adf: pd.DataFrame, inst: str, threshold: float = 95.0) -> bytes:
    sub = adf[(adf["instance"]==inst)].copy()
    sub = sub.sort_values("method", key=lambda s: s.map(method_sort_key))
    fig, ax = plt.subplots()
    if sub.empty:
        ax.text(0.5,0.5,"No data", ha="center")
        return _buf_png(fig)
    ax.scatter(sub["method"], sub["ontime_p50"])
    ax.axhline(threshold, linestyle="--")
    ax.set_xticklabels(sub["method"], rotation=30, ha="right")
    ax.set_ylabel("On-time p50 (%)"); ax.set_title(f"SLA sweep @ {threshold:.1f}%")
    return _buf_png(fig)


# ============================== Build / Write ==============================

def build_all(input_csv: Path,
              out_dir: Path,
              inst_for_figs: str = "R110",
              metric_for_screen: str = "p50",
              threshold: float = 95.0,
              include_sla_sweep: bool = False,
              feasible_only_for_screen: bool = False,
              K: Optional[int] = None,
              cvg: Optional[float] = None,
              cvl: Optional[float] = None,
              seed: Optional[int] = None,
              also_zip: bool = False) -> Path:
    """
    Generate all CSV tables and PNG figures with LaTeX-aligned names.
    Returns: path to the ZIP if also_zip=True else the output directory.
    """
    ensure_dir(out_dir)
    df = robust_read_csv(input_csv)

    # Apply CRN filters if provided
    adf = apply_scenario_filters(df, K=K, cvg=cvg, cvl=cvl, seed=seed)
    if adf.empty:
        raise ValueError("Filtered dataset is empty. Adjust K/cvg/cvl/seed or omit filters.")

    # ---- Tables ----
    tables = make_tables_for_latex(
        adf, inst_for_screen=inst_for_figs, metric=metric_for_screen,
        threshold=threshold, feasible_only_for_screen=feasible_only_for_screen
    )
    for name, tdf in tables.items():
        _save_csv(tdf, out_dir / name)

    # ---- Figures ----
    figs = {
        f"frontier_{inst_for_figs}_p50.png": fig_frontier_p50(adf, inst_for_figs),
        f"distance_by_method_{inst_for_figs}.png": fig_distance_by_method(adf, inst_for_figs),
        "improvement_vs_DET_box.png": fig_improve_vs_det_box(adf),
    }
    if include_sla_sweep:
        figs[f"sla_sweep_{inst_for_figs}.png"] = fig_sla_sweep(adf, inst_for_figs, threshold)

    for name, png_bytes in figs.items():
        _save_png_bytes(png_bytes, out_dir / name)

    # ---- ZIP (optional) ----
    if also_zip:
        zip_path = out_dir.with_suffix("")  # strip trailing slash
        zip_path = Path(f"{zip_path}.zip") if zip_path.suffix != ".zip" else zip_path
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.iterdir()):
                zf.write(p, arcname=p.name)
        return zip_path

    return out_dir


# ============================== CLI ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LaTeX-aligned CSV tables and PNG figures from VRPTW results CSV."
    )
    p.add_argument("--input", "-i", type=Path, required=True,
                   help="Path to results CSV (prefer cleaned CSV).")
    p.add_argument("--out", "-o", type=Path, default=Path("latex_exports"),
                   help="Output directory (default: ./latex_exports).")
    p.add_argument("--instance", "-n", type=str, default="R110",
                   help="Instance name for the figures/screens (default: R110).")
    p.add_argument("--metric", "-m", choices=["p50","p95"], default="p50",
                   help="SLA metric for the instance screen (default: p50).")
    p.add_argument("--threshold", "-t", type=float, default=95.0,
                   help="SLA threshold percentage (default: 95.0).")
    p.add_argument("--include-sla-sweep", action="store_true",
                   help="Also produce sla_sweep_[INST].png.")
    p.add_argument("--feasible-only", action="store_true",
                   help="For the instance screen champion, restrict to feasible==True before SLA check.")
    # CRN filters (optional)
    p.add_argument("--K", type=int, default=None, help="Filter K_eval to this value.")
    p.add_argument("--cvg", type=float, default=None, help="Filter cv_global_eval to this value (e.g., 0.20).")
    p.add_argument("--cvl", type=float, default=None, help="Filter cv_link_eval to this value (e.g., 0.10).")
    p.add_argument("--seed", type=int, default=None, help="Filter seed_eval to this value (e.g., 42).")

    p.add_argument("--zip", action="store_true",
                   help="Also write a ZIP next to the output directory.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    dest = build_all(
        input_csv=args.input,
        out_dir=args.out,
        inst_for_figs=args.instance,
        metric_for_screen=args.metric,
        threshold=args.threshold,
        include_sla_sweep=args.include_sla_sweep,
        feasible_only_for_screen=args.feasible_only,
        K=args.K, cvg=args.cvg, cvl=args.cvl, seed=args.seed,
        also_zip=args.zip
    )
    print(f"✅ Done. Wrote outputs to: {dest}")

if __name__ == "__main__":
    main()
