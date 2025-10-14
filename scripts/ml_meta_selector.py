#!/usr/bin/env python3
# ml_meta_selector.py (robust v3.2)
# - Normalizes/aliases columns
# - Computes missing targets (meets_SLA, gap_to_best_pct, is_champion)
# - Picks a UNIQUE champion per instance (ties broken deterministically)
# - Trains only when labels have >=2 classes; otherwise degrades gracefully
# - Works with sklearn old/new (OneHotEncoder sparse vs sparse_output)

import argparse, json, os, re
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error,
)
from sklearn.base import clone

RANDOM_STATE = 42

FULL_CATEGORICAL = [
    "instance","family","method","method_family","metaheuristic","tag","sla_metric","version_tag",
]
FULL_NUMERIC = [
    "n_customers","time_limit_s","vehicle_cost","K_eval","cv_global_eval","cv_link_eval","sla_threshold",
]

ALIASES = {
    "algo":"method","algorithm":"method","methodname":"method","method_name":"method","solver":"method",
    "strategy":"method","policy":"method","methodfamily":"method_family","method-family":"method_family",
    "metafamily":"method_family","metaheur":"metaheuristic","meta_heuristic":"metaheuristic",
    "veh_cost":"vehicle_cost","vehiclecost":"vehicle_cost","customers":"n_customers","nclients":"n_customers",
    "k_eval":"K_eval","cv_global":"cv_global_eval","cv_link":"cv_link_eval","runtime":"runtime_s",
    "run_time_s":"runtime_s","run_time":"runtime_s","distance_total":"distance","dist":"distance",
    "family_name":"family","version":"version_tag","meets_sla":"meets_SLA","is_champ":"is_champion",
    "champion":"is_champion","sla_thresh":"sla_threshold",
}

@dataclass
class RecoConfig:
    sla_prob_threshold: float = 0.90

# ---------------------------- Utilities ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        cc = re.sub(r"[^a-z0-9_]", "_", c.strip().lower())
        cc = re.sub(r"_+", "_", cc).strip("_")
        cols.append(ALIASES.get(cc, cc))
    out = df.copy()
    out.columns = cols
    return out

def pick_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def coerce_boolish(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        return s.astype(str).str.strip().str.lower().map(
            {"true":1,"false":0,"1":1,"0":0,"yes":1,"no":0}
        )
    return pd.to_numeric(s, errors="coerce")

def ensure_instance(df: pd.DataFrame, path: str, assume_from_filename: bool) -> pd.DataFrame:
    out = df.copy()
    if "instance" not in out.columns and assume_from_filename:
        out["instance"] = os.path.splitext(os.path.basename(path))[0]
    if "instance" not in out.columns:
        out["instance"] = "GLOBAL"
    return out

def ensure_method(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "method" not in out.columns:
        fallback = pick_existing(out, ["tag","method_family"])
        if fallback:
            out["method"] = out[fallback].astype(str)
        else:
            out["method"] = [f"m{i}" for i in range(len(out))]
    return out

def infer_meets_sla(df: pd.DataFrame, default_metric: str, default_thresh: float) -> pd.DataFrame:
    if "meets_SLA" in df.columns:
        return df
    metric_col = df["sla_metric"].iloc[0] if "sla_metric" in df.columns and df["sla_metric"].notna().any() else default_metric
    if metric_col not in df.columns:
        metric_col = pick_existing(df, ["ontime_p50","ontime_mean","ontime_p95","tard_mean"])
        if not metric_col:
            return df  # cannot compute
    threshold = df["sla_threshold"].iloc[0] if "sla_threshold" in df.columns and df["sla_threshold"].notna().any() else default_thresh

    vals = pd.to_numeric(df[metric_col], errors="coerce")
    thr = float(threshold)
    # Auto-scale thresholds if data is 0–1 but threshold looks 0–100
    if vals.dropna().between(0, 1.5).mean() > 0.7 and thr > 1.5:
        thr = thr / 100.0

    higher_is_better = not metric_col.startswith("tard")
    meets = (vals >= thr) if higher_is_better else (vals <= thr)

    out = df.copy()
    out["meets_SLA"] = meets.fillna(False).astype(int)
    return out

def infer_gap_and_champion(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "gap_to_best_pct" not in out.columns:
        if "distance" in out.columns:
            out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
            out["gap_to_best_pct"] = out.groupby("instance")["distance"].transform(
                lambda s: (s - s.min()) / s.min() * 100.0
            )
        else:
            out["gap_to_best_pct"] = 0.0

    # Ensure numeric gaps
    out["gap_to_best_pct"] = pd.to_numeric(out["gap_to_best_pct"], errors="coerce").fillna(np.inf)

    # UNIQUE champion per instance (break ties deterministically by first occurrence)
    is_champ = pd.Series(0, index=out.index, dtype=int)
    for inst, g in out.groupby("instance"):
        if g["gap_to_best_pct"].isna().all():
            idx = g.index[0]
        else:
            idx = g["gap_to_best_pct"].idxmin()
        is_champ.loc[idx] = 1

    out["is_champion"] = is_champ
    return out

# ---------------------------- Preprocessors ----------------------------
def build_preprocessor(num_cols: List[str], cat_cols: List[str]):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # sklearn >=1.2 uses sparse_output; older uses sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )

def make_classifier(num_cols, cat_cols):
    return Pipeline([
        ("pre", build_preprocessor(num_cols, cat_cols)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   solver="lbfgs", random_state=RANDOM_STATE)),
    ])

def make_regressor(num_cols, cat_cols, *, max_iter=300, lr=0.08, l2=0.5):
    return Pipeline([
        ("pre", build_preprocessor(num_cols, cat_cols)),
        ("reg", HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=lr,
                                              l2_regularization=l2, random_state=RANDOM_STATE)),
    ])

# ---------------------------- OOF helpers ----------------------------
def fit_oof_classifier(pipe, X, y, groups):
    # Skip entirely if single class overall
    if len(np.unique(y)) < 2:
        return np.full(len(X), np.nan, dtype=float)
    uniq = pd.Series(groups).nunique()
    if uniq < 2 or len(X) < 10:
        return np.full(len(X), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=min(5, uniq))
    oof = np.zeros(len(X), dtype=float)
    for tr, te in gkf.split(X, y, groups):
        # Skip fold if it becomes single-class
        if len(np.unique(y[tr])) < 2:
            oof[te] = np.nan
            continue
        fold = clone(pipe)
        fold.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = fold.predict_proba(X.iloc[te])[:, 1]
    return oof

def fit_oof_regressor(pipe, X, y, groups):
    uniq = pd.Series(groups).nunique()
    if uniq < 2 or len(X) < 10:
        return np.full(len(X), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=min(5, uniq))
    oof = np.zeros(len(X), dtype=float)
    for tr, te in gkf.split(X, y, groups):
        fold = clone(pipe)
        fold.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = fold.predict(X.iloc[te])
    return oof

# ---------------------------- Metrics ----------------------------
def eval_binary(y_true, p):
    p = np.asarray(p, dtype=float)
    mask = ~np.isnan(p)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return {"accuracy": np.nan, "f1": np.nan, "roc_auc": np.nan, "brier": np.nan, "note":"insufficient data"}
    y_pred = (p[mask] >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true[mask], y_pred)),
        "f1": float(f1_score(y_true[mask], y_pred)),
        "roc_auc": float(roc_auc_score(y_true[mask], p[mask])),
        "brier": float(brier_score_loss(y_true[mask], p[mask])),
    }

def eval_reg(y_true, y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "note":"insufficient data"}
    return {
        "MAE": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "RMSE": float(mean_squared_error(y_true[mask], y_pred[mask], squared=False)),
    }

# ---------------------------- Recommendation ----------------------------
def recommend(df, p_sla, pred_gap, pred_dist, pred_run, cfg: RecoConfig):
    work = pd.DataFrame({
        "instance": df["instance"].values,
        "method": df["method"].values,
        "p_meets_SLA": p_sla,
        "pred_gap_to_best_pct": pred_gap,
        "pred_distance": pred_dist,
        "pred_runtime_s": pred_run,
    })

    # Fallbacks to observed data if predictions are entirely NaN
    if np.all(np.isnan(work["p_meets_SLA"])) and "meets_SLA" in df.columns:
        work["p_meets_SLA"] = pd.to_numeric(df["meets_SLA"], errors="coerce").fillna(0.0)
    if np.all(np.isnan(work["pred_gap_to_best_pct"])) and "gap_to_best_pct" in df.columns:
        work["pred_gap_to_best_pct"] = pd.to_numeric(df["gap_to_best_pct"], errors="coerce")
    if np.all(np.isnan(work["pred_distance"])) and "distance" in df.columns:
        work["pred_distance"] = pd.to_numeric(df["distance"], errors="coerce")
    if np.all(np.isnan(work["pred_runtime_s"])) and "runtime_s" in df.columns:
        work["pred_runtime_s"] = pd.to_numeric(df["runtime_s"], errors="coerce")

    recos = []
    for inst, block in work.groupby("instance"):
        b = block.copy()
        if b["p_meets_SLA"].isna().all():
            b["p_meets_SLA"] = 0.0
        b["pred_gap_to_best_pct"] = b["pred_gap_to_best_pct"].fillna(
            b["pred_gap_to_best_pct"].median() if not b["pred_gap_to_best_pct"].dropna().empty else 999.0
        )
        b["pred_distance"] = b["pred_distance"].fillna(
            b["pred_distance"].median() if not b["pred_distance"].dropna().empty else 1e12
        )
        b["pred_runtime_s"] = b["pred_runtime_s"].fillna(
            b["pred_runtime_s"].median() if not b["pred_runtime_s"].dropna().empty else 1e12
        )

        qualified = b[b["p_meets_SLA"] >= cfg.sla_prob_threshold]
        if len(qualified) > 0:
            choice = qualified.sort_values(
                ["pred_gap_to_best_pct","p_meets_SLA","pred_distance"], ascending=[True,False,True]
            ).iloc[0]
        else:
            choice = b.sort_values(
                ["p_meets_SLA","pred_gap_to_best_pct","pred_distance"], ascending=[False,True,True]
            ).iloc[0]

        recos.append({
            "instance": inst,
            "recommended_method": choice["method"],
            "p_SLA": float(choice["p_meets_SLA"]),
            "expected_gap_to_best_pct": float(choice["pred_gap_to_best_pct"]),
            "expected_distance": float(choice["pred_distance"]),
            "expected_runtime_s": float(choice["pred_runtime_s"]),
        })
    return pd.DataFrame(recos)

# ---------------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sla_prob_threshold", type=float, default=0.90)
    ap.add_argument("--assume_instance_from_filename", action="store_true", default=True)
    ap.add_argument("--default_sla_metric", default="ontime_p50")
    ap.add_argument("--default_sla_threshold", type=float, default=95.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    raw = pd.read_csv(args.input)
    df = normalize_columns(raw)
    df = ensure_instance(df, args.input, args.assume_instance_from_filename)
    df = ensure_method(df)
    for c in ["is_champion","meets_SLA","feasible"]:
        if c in df.columns:
            df[c] = coerce_boolish(df[c])

    df = infer_meets_sla(df, args.default_sla_metric, args.default_sla_threshold)
    df = infer_gap_and_champion(df)

    # Features
    cat_cols = [c for c in FULL_CATEGORICAL if c in df.columns]
    num_cols = [c for c in FULL_NUMERIC if c in df.columns]
    if "method" not in cat_cols:
        cat_cols = ["method"] + cat_cols
    feat_cols = list(dict.fromkeys(cat_cols + num_cols))
    X = df[feat_cols].copy()
    groups = df["instance"]

    # Output containers
    n = len(df)
    p_champion = np.full(n, np.nan); p_sla = np.full(n, np.nan)
    pred_gap = np.full(n, np.nan);  pred_dist = np.full(n, np.nan); pred_runtime = np.full(n, np.nan)
    oof_p_champ = np.full(n, np.nan); oof_p_sla = np.full(n, np.nan)
    oof_gap = np.full(n, np.nan); oof_dist = np.full(n, np.nan); oof_run = np.full(n, np.nan)
    notes = []

    # -------- is_champion classifier --------
    y = df["is_champion"].astype(int).values
    if len(np.unique(y)) >= 2:
        clf = make_classifier(num_cols, cat_cols)
        oof_p_champ = fit_oof_classifier(clf, X, y, groups)
        p_champion = clone(clf).fit(X, y).predict_proba(X)[:, 1]
    else:
        # constant label; set constant probability
        const = float(np.unique(y)[0]) if len(np.unique(y)) == 1 else 0.0
        p_champion = np.full(n, const)
        notes.append("Classifier 'is_champion' skipped: only one class present.")

    # -------- meets_SLA classifier --------
    if "meets_SLA" in df.columns:
        y = df["meets_SLA"].astype(int).values
        if len(np.unique(y)) >= 2:
            clf2 = make_classifier(num_cols, cat_cols)
            oof_p_sla = fit_oof_classifier(clf2, X, y, groups)
            p_sla = clone(clf2).fit(X, y).predict_proba(X)[:, 1]
        else:
            const = float(np.unique(y)[0]) if len(np.unique(y)) == 1 else 0.0
            p_sla = np.full(n, const)
            notes.append("Classifier 'meets_SLA' skipped: only one class present.")
    else:
        notes.append("Classifier 'meets_SLA' skipped: column missing.")

    # -------- Regressors --------
    y_gap = pd.to_numeric(df["gap_to_best_pct"], errors="coerce").values
    reg_gap = make_regressor(num_cols, cat_cols, max_iter=400, lr=0.06, l2=1.0)
    oof_gap = fit_oof_regressor(reg_gap, X, y_gap, groups)
    pred_gap = clone(reg_gap).fit(X, y_gap).predict(X)

    if "distance" in df.columns:
        y = pd.to_numeric(df["distance"], errors="coerce").values
        reg_dist = make_regressor(num_cols, cat_cols)
        oof_dist = fit_oof_regressor(reg_dist, X, y, groups)
        pred_dist = clone(reg_dist).fit(X, y).predict(X)
    else:
        notes.append("Regressor 'distance' skipped: column missing.")

    if "runtime_s" in df.columns:
        y = pd.to_numeric(df["runtime_s"], errors="coerce").values
        reg_run = make_regressor(num_cols, cat_cols)
        oof_run = fit_oof_regressor(reg_run, X, y, groups)
        pred_runtime = clone(reg_run).fit(X, y).predict(X)
    else:
        notes.append("Regressor 'runtime_s' skipped: column missing.")

    # -------- Metrics --------
    metrics = {"binary":{}, "regression":{}, "recommendation":{}, "notes":notes}
    metrics["binary"]["is_champion"] = eval_binary(df["is_champion"].values, oof_p_champ)
    if "meets_SLA" in df.columns:
        metrics["binary"]["meets_SLA"] = eval_binary(df["meets_SLA"].values, oof_p_sla)
    metrics["regression"]["gap_to_best_pct"] = eval_reg(y_gap, oof_gap)
    if "distance" in df.columns:
        metrics["regression"]["distance"] = eval_reg(pd.to_numeric(df["distance"], errors="coerce").values, oof_dist)
    if "runtime_s" in df.columns:
        metrics["regression"]["runtime_s"] = eval_reg(pd.to_numeric(df["runtime_s"], errors="coerce").values, oof_run)

    # -------- Per-row predictions CSV --------
    cols_exist = [c for c in ["instance","family","n_customers","method","method_family","metaheuristic","tag"] if c in df.columns]
    pred_df = df[cols_exist].copy()
    pred_df["p_champion"] = p_champion
    pred_df["p_meets_SLA"] = p_sla
    pred_df["pred_gap_to_best_pct"] = pred_gap
    pred_df["pred_distance"] = pred_dist
    pred_df["pred_runtime_s"] = pred_runtime
    pred_df["OOF_p_champion"] = oof_p_champ
    pred_df["OOF_p_meets_SLA"] = oof_p_sla
    pred_df["OOF_pred_gap_to_best_pct"] = oof_gap
    pred_df["OOF_pred_distance"] = oof_dist
    pred_df["OOF_pred_runtime_s"] = oof_run

    per_row_path = os.path.join(args.outdir, "ml_predictions.csv")
    pred_df.to_csv(per_row_path, index=False)

    # -------- Recommendations CSV --------
    reco_df = recommend(
        df=df,
        p_sla=p_sla,
        pred_gap=pred_gap,
        pred_dist=pred_dist,
        pred_run=pred_runtime,
        cfg=RecoConfig(args.sla_prob_threshold)
    )
    if all(col in df.columns for col in ["is_champion","gap_to_best_pct","method"]):
        truth = df[["instance","method","is_champion","gap_to_best_pct"]].copy()
        reco_df = reco_df.merge(truth, left_on=["instance","recommended_method"], right_on=["instance","method"], how="left")
        reco_df = reco_df.drop(columns=["method"]).rename(columns={
            "is_champion":"is_true_champion","gap_to_best_pct":"actual_gap_to_best_pct"
        })

    reco_path = os.path.join(args.outdir, "instance_recommendations.csv")
    reco_df.to_csv(reco_path, index=False)

    # -------- OOF recommendation evaluation (if available) --------
    if not np.isnan(oof_gap).all():
        tmp = pd.DataFrame({
            "instance": df["instance"],
            "method": df["method"],
            "is_champion": df["is_champion"],
            "true_gap": y_gap,
            "p_sla": oof_p_sla,
            "pred_gap": oof_gap,
        })
        hits, gaps = [], []
        for inst, g in tmp.groupby("instance"):
            q = g[g["p_sla"] >= args.sla_prob_threshold]
            ch = (q if len(q) > 0 else g).sort_values(["p_sla","pred_gap"], ascending=[False, True]).iloc[0]
            hits.append(int(ch["is_champion"] == 1))
            gaps.append(float(ch["true_gap"]))
        metrics["recommendation"] = {
            "top1_hit_rate": float(np.mean(hits)),
            "mean_actual_gap_of_recommended_pct": float(np.mean(gaps)),
            "oof_used": True,
        }
    else:
        metrics["recommendation"] = {"note":"skipped OOF reco eval (insufficient groups/OOD labels)."}

    metrics_path = os.path.join(args.outdir, "ml_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote per-row predictions -> {per_row_path}")
    print(f"Wrote per-instance recommendations -> {reco_path}")
    print(f"Wrote metrics -> {metrics_path}")
    print(f"\nRecommendation rule: SLA prob >= {args.sla_prob_threshold} then pick min predicted gap; "
          f"fallbacks use observed columns when predictions are unavailable.")

if __name__ == "__main__":
    main()
