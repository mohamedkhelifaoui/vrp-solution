# scripts/pfe_to_ml.py
# End-to-end pipeline for PFE → ML
# - Reads the CSV
# - Cleans/casts columns
# - Builds feature sets (start-only vs. full but no-leakage)
# - Optional: rebuilds the boolean target from a metric + threshold/quantile/target rate
# - Trains & evaluates models for meets_SLA or is_champion
# - Grouped CV (default by 'instance'); uses StratifiedGroupKFold if available
# - Auto-creates output directories for --save-model / --metrics-json

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import GroupKFold
# Try to use StratifiedGroupKFold if available (sklearn >= 1.3)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    StratifiedGroupKFold = None  # type: ignore
    HAS_SGKF = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


# ---------- Column configuration (covers "all the PFE") ----------
CATEGORICAL_COLS = [
    "instance","family","method","method_family","tag","metaheuristic",
    "sla_metric","status","version_tag"
]

BOOLEAN_COLS = ["feasible","meets_SLA","is_champion"]

NUMERIC_COLS = [
    "n_customers","time_limit_s","vehicle_cost","K_eval","seed_eval",
    "cv_global_eval","cv_link_eval","vehicles","distance","runtime_s",
    "ontime_mean","ontime_p50","ontime_p95","tard_mean",
    "sla_threshold","seed_build","gap_to_det_pct","gap_to_best_pct"
]

# Columns that are post-solve evaluation signals (leakage risk for most targets)
POST_SOLVE_RESULT_COLS = [
    "feasible","vehicles","distance","runtime_s",
    "ontime_mean","ontime_p50","ontime_p95","tard_mean",
    "gap_to_det_pct","gap_to_best_pct"
]

# "Start" (pre-solve) feature set: safe for early ML (no eval outcomes)
START_FEATURES = (
    CATEGORICAL_COLS
    + [
        "n_customers","time_limit_s","vehicle_cost",
        "K_eval","seed_eval","cv_global_eval","cv_link_eval",
        "sla_threshold","seed_build"
      ]
)

# Full-but-no-leak feature set (keeps design/configuration; drops outcome-like columns)
FULL_NOLEAK_FEATURES = (
    list(set(CATEGORICAL_COLS + NUMERIC_COLS) - set(POST_SOLVE_RESULT_COLS))
)

DEFAULT_TARGET = "meets_SLA"  # alternative: "is_champion"


# ---------- Data utilities ----------
def load_and_cast(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Make sure expected columns exist (soft check)
    missing = set(CATEGORICAL_COLS + BOOLEAN_COLS + NUMERIC_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # Cast types
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("string").fillna("NA")

    for b in BOOLEAN_COLS:
        # Handle 'True'/'False' strings, 1/0, or actual bools
        df[b] = (
            df[b]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "1": True, "false": False, "0": False})
            .astype("boolean")
        )

    for n in NUMERIC_COLS:
        df[n] = pd.to_numeric(df[n], errors="coerce")

    return df


def pick_feature_cols(
    df: pd.DataFrame,
    mode: str,
    target: str
) -> List[str]:
    """
    mode in {"start","full_noleak"}
    Additionally removes direct target proxies (hard leakage guards).
    """
    if mode == "start":
        cols = [c for c in START_FEATURES if c in df.columns]
    elif mode == "full_noleak":
        cols = [c for c in FULL_NOLEAK_FEATURES if c in df.columns]
    else:
        raise ValueError("mode must be 'start' or 'full_noleak'")

    # Guard against using the target or columns that literally define it
    leakage_like = set([target])

    # If predicting meets_SLA, do NOT use the ontime/tard outcomes (they define it)
    if target == "meets_SLA":
        leakage_like |= set(["ontime_mean","ontime_p50","ontime_p95","tard_mean"])

    # If predicting is_champion, avoid using "gap_to_*" columns
    if target == "is_champion":
        leakage_like |= set(["gap_to_det_pct","gap_to_best_pct"])

    cols = [c for c in cols if c not in leakage_like]
    return cols


def build_preprocessor(
    feature_cols: List[str], df: pd.DataFrame
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS]
    num_cols = [c for c in feature_cols if c in NUMERIC_COLS]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, cat_cols, num_cols


def default_model(kind: str = "lr"):
    if kind == "lr":
        # Good baseline, balanced for class imbalance
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs"
        )
    elif kind == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42
        )
    else:
        raise ValueError("Unknown model kind; use 'lr' or 'rf'")


def evaluate_binary(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # If the validation set contains a single class, skip AUC (undefined)
    classes = np.unique(y_true)
    if classes.size < 2:
        out["roc_auc"] = float("nan")
        # AP is defined for single-class labels; keep it.
        out["avg_precision"] = average_precision_score(y_true, y_prob)
        return out

    # Normal case
    out["roc_auc"] = roc_auc_score(y_true, y_prob)
    out["avg_precision"] = average_precision_score(y_true, y_prob)
    return out


def _predict_proba_safe(pipe: Pipeline, Xva: pd.DataFrame) -> np.ndarray:
    """
    Handle single-class training folds gracefully:
    if the fitted model has only one class, return a constant probability
    of 1.0 (for class 1) or 0.0 (for class 0).
    """
    model = pipe.named_steps["model"]
    classes = getattr(model, "classes_", None)

    # If model learned both classes, use normal proba
    if classes is not None and len(classes) == 2:
        return pipe.predict_proba(Xva)[:, 1]

    # Single-class model: constant prob
    if classes is not None and len(classes) == 1:
        c = int(classes[0])
        return np.full(len(Xva), 1.0 if c == 1 else 0.0, dtype=float)

    # Fallback (shouldn't happen): try predict_proba, otherwise zeros
    try:
        proba = pipe.predict_proba(Xva)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0]
        return np.zeros(len(Xva), dtype=float)
    except Exception:
        return np.zeros(len(Xva), dtype=float)


# ---------- Target rebuild helpers ----------
def _make_boolean_from_threshold(
    s: pd.Series, op: str, thr: float
) -> pd.Series:
    if op == "ge":
        return (s >= thr)
    if op == "gt":
        return (s > thr)
    if op == "le":
        return (s <= thr)
    if op == "lt":
        return (s < thr)
    raise ValueError("op must be one of ge, gt, le, lt")


def _positive_rate_from_threshold(
    s: pd.Series, op: str, thr: float
) -> float:
    return float(_make_boolean_from_threshold(s, op, thr).mean())


def _quantile_for_rate(op: str, rate: float) -> float:
    """
    Map desired positive rate (0..1) to a pandas quantile q (0..1)
    consistent with the chosen inequality.
      - For ge / gt (positives are high values): rate ~= 1 - q
      - For le / lt (positives are low values):  rate ~= q
    """
    rate = min(max(rate, 0.0), 1.0)
    if op in ("ge", "gt"):
        return 1.0 - rate
    elif op in ("le", "lt"):
        return rate
    else:
        raise ValueError("op must be one of ge, gt, le, lt")


def _auto_adjust_threshold_for_bounds(
    s: pd.Series,
    op: str,
    thr: float,
    min_rate: Optional[float],
    max_rate: Optional[float],
    rate_hint: Optional[float],
) -> tuple[float, str]:
    """
    If current positive rate is out of [min_rate, max_rate], pick a
    quantile-based threshold to bring it inside. If rate_hint is provided,
    target that; otherwise aim for the midpoint within bounds.
    Returns (new_thr, reason_suffix).
    """
    cur_rate = _positive_rate_from_threshold(s, op, thr)
    lo = 0.0 if min_rate is None else min_rate
    hi = 1.0 if max_rate is None else max_rate
    if lo <= cur_rate <= hi:
        return thr, ""

    # Choose a desired rate
    if rate_hint is not None:
        target_rate = min(max(rate_hint, lo), hi)
    else:
        target_rate = (lo + hi) / 2.0

    q = _quantile_for_rate(op, target_rate)
    new_thr = float(s.quantile(q))
    suffix = f"auto-adjust: quantile q={q:.2f}"
    return new_thr, suffix


# ---------- Optional: rebuild target from a metric & threshold ----------
def rebuild_target_from_metric(df: pd.DataFrame, args, target: str) -> pd.DataFrame:
    """
    If --make-target-metric is provided, rebuilds the boolean target using:
      metric {op} threshold
    The threshold can come from:
      - --make-target-threshold
      - --make-target-quantile (q in [0..1])
      - --target-positive-rate (desired positive prevalence, [0..1])
      - default median (q=0.5)
    If --min-positive-rate / --max-positive-rate are set (and auto-adjust is on),
    the script will adjust the threshold to make the final prevalence fall within
    those bounds.
    """
    if not args.make_target_metric:
        return df

    mcol = args.make_target_metric
    if mcol not in df.columns:
        raise ValueError(f"--make-target-metric '{mcol}' not found in CSV columns")
    if not pd.api.types.is_numeric_dtype(df[mcol]):
        raise ValueError(f"--make-target-metric '{mcol}' must be numeric")

    s = pd.to_numeric(df[mcol], errors="coerce")
    op = args.make_target_op

    reason = ""
    thr: Optional[float] = None

    # Priority: explicit threshold > explicit quantile > target-positive-rate > default median
    if args.make_target_threshold is not None:
        thr = float(args.make_target_threshold)
        reason = "explicit"
    elif args.make_target_quantile is not None:
        q = float(args.make_target_quantile)
        thr = float(s.quantile(q))
        reason = f"quantile q={q:.2f}"
    elif args.target_positive_rate is not None:
        r = float(args.target_positive_rate)
        q = _quantile_for_rate(op, r)
        thr = float(s.quantile(q))
        reason = f"target-rate r={r:.2f} → quantile q={q:.2f}"
    else:
        thr = float(s.quantile(0.5))
        reason = "default median"

    # Optionally clamp prevalence into [min, max]
    if args.auto_adjust_labels and (args.min_positive_rate is not None or args.max_positive_rate is not None):
        new_thr, suffix = _auto_adjust_threshold_for_bounds(
            s=s,
            op=op,
            thr=thr,
            min_rate=args.min_positive_rate,
            max_rate=args.max_positive_rate,
            rate_hint=args.target_positive_rate,
        )
        if suffix:
            reason = f"{reason} → {suffix}"
            thr = new_thr

    new_y = _make_boolean_from_threshold(s, op, thr).astype(bool)

    df = df.copy()
    df[target] = new_y

    # quick visibility
    pos = int(new_y.sum()); n = int(len(new_y))
    print(f"[Target rebuilt] {target} := {mcol} {op} {thr:.4g} | positives={pos}/{n} ({pos/n:.1%}) [{reason}]")
    return df


# ---------- Training / CV ----------
def run_grouped_cv(
    df: pd.DataFrame,
    target: str,
    feature_mode: str = "start",
    model_kind: str = "rf",
    n_splits: int = 5,
    group_col: str = "instance",
    random_state: int = 42,
) -> Dict:
    # Drop rows missing the target
    data = df.dropna(subset=[target]).copy()
    data[target] = data[target].astype(bool)

    # Show label balance overall and by group (helps explain single-class folds)
    print(target)
    print(data[target].value_counts(dropna=False).rename("count"))
    vc_by_group = (
        data.groupby(group_col)[target]
            .value_counts()
            .unstack(fill_value=0)
            .sort_index()
    )
    print(vc_by_group)

    feature_cols = pick_feature_cols(data, feature_mode, target)
    pre, cat_cols, num_cols = build_preprocessor(feature_cols, data)

    model = default_model(model_kind)
    pipe = Pipeline([("pre", pre), ("model", model)])

    groups = data[group_col].astype(str)  # group by chosen column

    X = data[feature_cols]
    y = data[target].astype(int).values

    # Guard against single-class labels across the whole dataset
    if np.unique(y).size < 2:
        raise ValueError(
            f"Target '{target}' has a single class after filtering. "
            "Use --make-target-* to rebuild a meaningful label."
        )

    # Prefer stratified grouping if available to reduce single-class folds
    if HAS_SGKF:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = splitter.split(X, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups)

    fold_metrics: List[Dict[str, float]] = []
    fold_reports: List[Dict[str, float]] = []

    for fold_idx, (tr, va) in enumerate(splits, start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        pipe.fit(Xtr, ytr)
        prob = _predict_proba_safe(pipe, Xva)
        pred = (prob >= 0.5).astype(int)

        m = evaluate_binary(yva, prob, pred)

        # Optional: per-group Top-1 selection metric for is_champion
        if target == "is_champion":
            va_df = data.iloc[va].copy()
            va_df["_prob"] = prob
            hits = []
            for _, g in va_df.groupby(group_col):
                idx = g["_prob"].idxmax()
                hits.append(bool(g.loc[idx, target]))
            if hits:
                m["top1_is_champion"] = float(np.mean(hits))

        fold_metrics.append(m)
        fold_reports.append({
            "fold": fold_idx,
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            **m
        })

    # Train final model on full data
    pipe.fit(X, y)

    # Robust CV averages (avoid empty-slice warnings)
    keys = sorted({k for m in fold_metrics for k in m.keys()})
    cv_avg = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}

    results = {
        "feature_mode": feature_mode,
        "model_kind": model_kind,
        "target": target,
        "group_col": group_col,
        "feature_cols": feature_cols,
        "categoricals": cat_cols,
        "numerics": num_cols,
        "cv": fold_reports,
        "cv_avg": cv_avg,
        "fitted_pipeline": pipe
    }
    return results


# ---------- CLI ----------
@dataclass
class Args:
    csv_path: str
    target: str
    feature_mode: str
    model_kind: str
    n_splits: int
    save_model: str | None
    group_col: str
    metrics_json: str | None
    make_target_metric: str | None
    make_target_op: str
    make_target_threshold: float | None
    make_target_quantile: float | None
    target_positive_rate: float | None
    min_positive_rate: float | None
    max_positive_rate: float | None
    auto_adjust_labels: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="PFE → ML pipeline")
    p.add_argument("--csv", dest="csv_path", required=True, help="Path to CSV with PFE results")
    p.add_argument("--target", default=DEFAULT_TARGET, choices=["meets_SLA","is_champion"], help="Target label")
    p.add_argument("--features", dest="feature_mode", default="start", choices=["start","full_noleak"], help="Feature set")
    p.add_argument("--model", dest="model_kind", default="rf", choices=["rf","lr"], help="Model type")
    p.add_argument("--splits", dest="n_splits", type=int, default=5, help="GroupKFold splits")
    p.add_argument("--group-col", dest="group_col", default="instance", help="Column to group folds by (default: instance)")
    p.add_argument("--save-model", dest="save_model", default=None, help="Path to save the fitted pipeline (.joblib)")
    p.add_argument("--metrics-json", dest="metrics_json", default=None, help="Optional path to save CV metrics as JSON")

    # Target rebuild options
    p.add_argument("--make-target-metric", dest="make_target_metric", default=None,
                   help="Rebuild boolean target from this numeric metric column")
    p.add_argument("--make-target-op", dest="make_target_op", default="ge",
                   choices=["ge","gt","le","lt"], help="Comparison op for rebuilt target")
    p.add_argument("--make-target-threshold", dest="make_target_threshold", type=float, default=None,
                   help="Numeric threshold for --make-target-metric")
    p.add_argument("--make-target-quantile", dest="make_target_quantile", type=float, default=None,
                   help="If set (e.g., 0.7), use the metric quantile as threshold")
    p.add_argument("--target-positive-rate", dest="target_positive_rate", type=float, default=None,
                   help="Desired fraction of positives in [0,1]. For ge/gt this targets high values, for le/lt low values.")
    p.add_argument("--min-positive-rate", dest="min_positive_rate", type=float, default=None,
                   help="Minimum acceptable positive rate in [0,1] (auto-adjust).")
    p.add_argument("--max-positive-rate", dest="max_positive_rate", type=float, default=None,
                   help="Maximum acceptable positive rate in [0,1] (auto-adjust).")
    p.add_argument("--no-auto-adjust-labels", dest="auto_adjust_labels", action="store_false",
                   help="Disable automatic prevalence adjustment when outside the given bounds.")
    p.set_defaults(auto_adjust_labels=True)

    a = p.parse_args()
    return Args(
        csv_path=a.csv_path,
        target=a.target,
        feature_mode=a.feature_mode,
        model_kind=a.model_kind,
        n_splits=a.n_splits,
        save_model=a.save_model,
        group_col=a.group_col,
        metrics_json=a.metrics_json,
        make_target_metric=a.make_target_metric,
        make_target_op=a.make_target_op,
        make_target_threshold=a.make_target_threshold,
        make_target_quantile=a.make_target_quantile,
        target_positive_rate=a.target_positive_rate,
        min_positive_rate=a.min_positive_rate,
        max_positive_rate=a.max_positive_rate,
        auto_adjust_labels=a.auto_adjust_labels,
    )


def _ensure_parent_dir(path_str: str) -> None:
    if not path_str:
        return
    p = Path(path_str)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    df = load_and_cast(args.csv_path)

    # Optionally rebuild the target from a metric threshold/quantile/desired rate
    df = rebuild_target_from_metric(df, args, target=args.target)

    results = run_grouped_cv(
        df=df,
        target=args.target,
        feature_mode=args.feature_mode,
        model_kind=args.model_kind,
        n_splits=args.n_splits,
        group_col=args.group_col,
    )

    # Print a compact summary
    print("\n=== PFE → ML Summary ===")
    print(f"Target: {results['target']}")
    print(f"Features: {results['feature_mode']}  |  Model: {results['model_kind']}")
    print(f"n_features: {len(results['feature_cols'])}  (num={len(results['numerics'])}, cat={len(results['categoricals'])})")
    print("\nCV by {gc} (GroupKFold{sgkf}):".format(
        gc=results["group_col"], sgkf=" + Stratified" if HAS_SGKF else "")
    )
    for r in results["cv"]:
        line = (f"  Fold {r['fold']}: acc={r['accuracy']:.3f} f1={r['f1']:.3f} "
                f"auc={r['roc_auc']:.3f} ap={r['avg_precision']:.3f}  "
                f"n_train={r['n_train']} n_valid={r['n_valid']}")
        if "top1_is_champion" in r:
            line += f" top1={r['top1_is_champion']:.3f}"
        print(line)

    avg = results["cv_avg"]
    extras = ""
    if "top1_is_champion" in avg:
        extras = f"  top1={avg.get('top1_is_champion', float('nan')):.3f}"
    print("\nCV averages:")
    print(f"  acc={avg.get('accuracy', float('nan')):.3f}  "
          f"f1={avg.get('f1', float('nan')):.3f}  "
          f"auc={avg.get('roc_auc', float('nan')):.3f}  "
          f"ap={avg.get('avg_precision', float('nan')):.3f}{extras}")

    if args.save_model:
        _ensure_parent_dir(args.save_model)
        joblib.dump(results["fitted_pipeline"], args.save_model)
        print(f"\nSaved fitted pipeline → {args.save_model}")

    if args.metrics_json:
        _ensure_parent_dir(args.metrics_json)
        safe_results = dict(results)
        safe_results.pop("fitted_pipeline", None)  # not JSON-serializable
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(safe_results, f, indent=2)
        print(f"Saved metrics JSON → {args.metrics_json}")


if __name__ == "__main__":
    main()
