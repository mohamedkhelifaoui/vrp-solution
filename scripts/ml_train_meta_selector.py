#!/usr/bin/env python3
# scripts/ml_train_meta_selector.py  (fixed)
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = Path("data/ml/meta_train.csv")
MODEL_PATH = MODEL_DIR / "meta_selector.pkl"

def load_model_or_fallback():
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        used = "LightGBM"
    except Exception:
        clf = HistGradientBoostingClassifier(random_state=42, max_depth=None)
        used = "HistGradientBoosting (sklearn)"
    return clf, used

def main():
    if not DATA_CSV.exists():
        sys.exit(f"[ERROR] Missing {DATA_CSV}. Run ml_build_meta_dataset.py first.")
    df = pd.read_csv(DATA_CSV)

    # Target
    y_raw = df["method"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Features
    cat_cols = ["family", "horizon"]
    num_cols = [c for c in df.columns if c not in ["instance","method"] + cat_cols]
    X = df[cat_cols + num_cols]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    base_clf, used = load_model_or_fallback()
    pipe = Pipeline([("pre", pre), ("clf", base_clf)])

    # --- Cross-validation (grouped by family when possible) ---
    groups = df["family"].astype(str)
    n_groups = groups.nunique()
    try:
        if n_groups >= 3:
            n_splits = min(5, n_groups)
            gkf = GroupKFold(n_splits=n_splits)
            scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="accuracy")
            print(f"[CV(GroupKFold x{n_splits})] {used}  "
                  f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        else:
            # Fallback if not enough families: stratified CV on labels
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
            print(f"[CV(StratifiedKFold x3)] {used}  "
                  f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    except Exception as e:
        print(f"[WARN] CV failed ({e}); fitting without CV...")

    # Fit full model and save
    pipe.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "pipeline": pipe,
            "label_encoder": le,
            "feature_cols": cat_cols + num_cols,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "classes_": list(le.classes_),
        }, f)
    print(f"[OK] Saved model to {MODEL_PATH}")

    # Quick training-set report
    y_hat = pipe.predict(X)
    y_hat_lbl = le.inverse_transform(y_hat)
    print("\n[Train-set classification report]")
    print(classification_report(df["method"], y_hat_lbl, zero_division=0))

if __name__ == "__main__":
    main()
