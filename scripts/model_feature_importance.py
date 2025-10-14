# scripts/model_feature_importance.py
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="artifacts/feature_importance.csv")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    pre  = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(getattr(model, "n_features_in_", len(getattr(model, "feature_importances_", []))))]

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        kind = "rf_importance"
    else:
        # fall back for linear models: absolute coefficient magnitude
        coef = getattr(model, "coef_", None)
        if coef is None:
            raise ValueError("Model has neither feature_importances_ nor coef_.")
        imp = np.abs(coef.ravel())
        kind = "abs_coef"

    df = pd.DataFrame({"feature": feat_names, "importance": imp})
    df = df.sort_values("importance", ascending=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.head(args.top).to_string(index=False))
    print(f"\nSaved full table → {args.out}")

if __name__ == "__main__":
    main()
