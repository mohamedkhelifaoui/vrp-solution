# latex_exports.py — CSV+PNG exports with LaTeX-aligned names
from __future__ import annotations
from typing import Dict
import io, zipfile, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Preferred method order ----------
PREF_METHOD_ORDER = [
    "DET","Q110","Q120","Q130",
    "SAA16-b0p3","SAA32-b0p5","SAA64-b0p7",
    "Gamma1-q1p645","Gamma2-q1p645"
]

def _method_sort_key(m: str) -> tuple:
    try:
        return (0, PREF_METHOD_ORDER.index(m))
    except ValueError:
        return (1, m)

# ---------- Tables ----------
def make_tables_for_latex(adf: pd.DataFrame,
                          inst_for_screen: str,
                          metric: str = "p50",
                          threshold: float = 95.0) -> Dict[str, pd.DataFrame]:
    """Build all CSV tables using the current filtered dataset (ADF)."""

    # tab:champions_rows → champions.csv
    champs = adf[adf.get("is_champion", False) == True].copy()
    champions_rows = champs[[
        "instance","method","method_family","vehicles","distance",
        "ontime_p50","ontime_p95","meets_SLA"
    ]].sort_values("instance")

    # tab:champions_by_method → champions_stats_by_method.csv
    champions_by_method = (
        champs.groupby(["method","method_family"], as_index=False)
              .agg(champions=("instance","count"),
                   avg_distance=("distance","mean"),
                   avg_vehicles=("vehicles","mean"),
                   avg_p50=("ontime_p50","mean"))
              .sort_values(["champions","avg_distance"], ascending=[False, True])
    )

    # tab:summary_by_family → summary_by_family.csv
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

    # tab:summary_by_method → summary_by_method.csv
    summary_by_method = (
        adf.groupby(["method","method_family"], as_index=False)
           .agg(rows=("instance","count"),
                feasible_rate=("feasible", lambda s: 100*np.mean(s)),
                sla_rate=("meets_SLA", lambda s: 100*np.mean(s)),
                dist_mean=("distance","mean"),
                veh_mean=("vehicles","mean"),
                p50_mean=("ontime_p50","mean"))
           .sort_values("method", key=lambda s: s.map(_method_sort_key))
    )

    # tab:instance_screen_[INST] → [INST]_scored_[p50|p95].csv
    sub = adf[adf["instance"] == inst_for_screen].copy()
    if metric == "p50":
        sub["meets_SLA(target)"] = sub["ontime_p50"] >= threshold
    else:
        sub["meets_SLA(target)"] = sub["ontime_p95"] >= threshold

    cand = sub[sub["meets_SLA(target)"]]
    if cand.empty:
        # least violator
        best = sub.loc[[sub["ontime_p50"].idxmax()]] if metric=="p50" else sub.loc[[sub["ontime_p95"].idxmax()]]
    else:
        # strict, deterministic tie-break
        best = cand.sort_values(["distance","vehicles","runtime_s","method"], ascending=True).head(1)


    sub["is_champion(target)"] = sub["method"].isin(best["method"])
    instance_screen = (
        sub[[
            "instance","method","method_family","feasible",
            "meets_SLA(target)","is_champion(target)",
            "vehicles","distance","runtime_s",
            "ontime_p50","ontime_p95","gap_to_det_pct","gap_to_best_pct",
            "tag","metaheuristic"
        ]].sort_values(["meets_SLA(target)","distance"], ascending=[False, True])
    )

    return {
        "champions.csv": champions_rows,
        "champions_stats_by_method.csv": champions_by_method,
        "summary_by_family.csv": summary_by_family,
        "summary_by_method.csv": summary_by_method,
        f"{inst_for_screen}_scored_{metric}.csv": instance_screen,
    }

# ---------- Figures (PNG bytes) ----------
def _buf_png(fig, dpi=200) -> bytes:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    b.seek(0)
    return b.read()

def fig_frontier_p50(adf: pd.DataFrame, inst: str) -> bytes:
    sub = adf[(adf["instance"]==inst)].copy()
    if sub.empty:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data", ha="center"); return _buf_png(fig)
    methods = sub["method"].unique().tolist()
    veh_vals = sorted(sub["vehicles"].astype(int).unique().tolist())
    marker_pool = ["o","s","^","D","P","X","v","<",">"]
    veh_to_marker = {v: marker_pool[i % len(marker_pool)] for i, v in enumerate(veh_vals)}
    cmap = plt.cm.get_cmap("tab10", len(methods))
    meth_to_color = {m: cmap(i) for i, m in enumerate(methods)}
    fig, ax = plt.subplots()
    for _, r in sub.iterrows():
        ax.scatter(r["distance"], r["ontime_p50"],
                   marker=veh_to_marker[int(r["vehicles"])],
                   color=meth_to_color[r["method"]], alpha=0.9)
    mhandles = [plt.Line2D([0],[0], marker='o', ls='None', color=meth_to_color[m], label=m) for m in methods]
    vhandles = [plt.Line2D([0],[0], marker=veh_to_marker[v], ls='None', color='gray', label=f'{v} veh') for v in veh_vals]
    leg1 = ax.legend(handles=mhandles, title="Method", loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=vhandles, title="Vehicles", loc="lower left")
    ax.set_xlabel("Distance"); ax.set_ylabel("On-time p50 (%)"); ax.grid(alpha=0.3)
    return _buf_png(fig)

def fig_distance_by_method(adf: pd.DataFrame, inst: str) -> bytes:
    sub = adf[(adf["instance"]==inst)].copy().sort_values("method", key=lambda s: s.map(_method_sort_key))
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
    sub = adf[(adf["instance"]==inst)].copy().sort_values("method", key=lambda s: s.map(_method_sort_key))
    fig, ax = plt.subplots()
    if sub.empty:
        ax.text(0.5,0.5,"No data", ha="center")
        return _buf_png(fig)
    ax.scatter(sub["method"], sub["ontime_p50"])
    ax.axhline(threshold, linestyle="--")
    ax.set_xticklabels(sub["method"], rotation=30, ha="right")
    ax.set_ylabel("On-time p50 (%)"); ax.set_title(f"SLA sweep @ {threshold:.1f}%")
    return _buf_png(fig)

# ---------- ZIP pack ----------
def build_latex_zip(tables: Dict[str, pd.DataFrame],
                    figs: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, d in tables.items():
            zf.writestr(fname, d.to_csv(index=False))
        for fname, png_bytes in figs.items():
            zf.writestr(fname, png_bytes)
    buf.seek(0)
    return buf.read()

# ---------- Streamlit UI section ----------
def render_latex_exports_ui(adf: pd.DataFrame, default_inst: str = "R110") -> None:
    st.markdown("---")
    st.subheader("LaTeX exports (CSV + PNG)")

    inst_list = sorted(adf["instance"].unique().tolist())
    if not inst_list:
        st.info("No instances in current filter.")
        return
    inst_default = inst_list.index(default_inst) if default_inst in inst_list else 0
    inst_for_figs = st.selectbox("Instance for figures", inst_list, index=inst_default, key="latex_inst")

    colA, colB, colC = st.columns(3)
    with colA:
        sla_metric_export = st.radio("Metric for instance screen", ["p50","p95"], index=0, horizontal=True, key="latex_metric")
    with colB:
        sla_thr_export = st.number_input("SLA threshold (%)", min_value=80.0, max_value=99.9, value=95.0, step=0.1, key="latex_thr")
    with colC:
        include_sla_sweep = st.checkbox("Include SLA sweep figure", value=False, key="latex_sweep")

    if st.button("Build LaTeX pack (CSV+PNGs)", key="build_latex"):
        tables = make_tables_for_latex(adf, inst_for_figs, sla_metric_export, sla_thr_export)
        figs = {
            f"frontier_{inst_for_figs}_p50.png": fig_frontier_p50(adf, inst_for_figs),
            f"distance_by_method_{inst_for_figs}.png": fig_distance_by_method(adf, inst_for_figs),
            "improvement_vs_DET_box.png": fig_improve_vs_det_box(adf),
        }
        if include_sla_sweep:
            figs[f"sla_sweep_{inst_for_figs}.png"] = fig_sla_sweep(adf, inst_for_figs, sla_thr_export)

        zip_bytes = build_latex_zip(tables, figs)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="Download latex_exports.zip",
            data=zip_bytes,
            file_name=f"latex_exports_{stamp}.zip",
            mime="application/zip"
        )

        st.caption("Included CSVs:"); st.write(sorted(tables.keys()))
        st.caption("Included PNGs:"); st.write(sorted(figs.keys()))
    else:
        st.info("Configure options above, then click **Build LaTeX pack (CSV+PNGs)**.")
