# scripts/score_pfe.py
import argparse
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv-in", required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability cutoff for label=1")
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    df = pd.read_csv(args.csv_in)

    # ColumnTransformer in the pipeline will select the right feature columns by name.
    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out = df.copy()
    out["pred_prob"] = proba
    out["pred_label"] = pred

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv_out, index=False)
    print(f"Wrote {len(out)} rows → {args.csv_out} (threshold={args.threshold:.2f})")

if __name__ == "__main__":
    main()
