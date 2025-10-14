# scripts/ml_autorun.py
import argparse, subprocess, sys, shutil, re
from pathlib import Path
import pandas as pd

def method_to_dir_label(m: str):
    m = (m or "").strip()
    candidates = {
        "SAA16-b0p3": [
            ("data/solutions_saa/k16_b0p3", "SAA16-b0p3"),
            ("data/solutions_saa/k16-b0P3", "SAA16-b0p3"),
        ],
        "SAA32-b0p5": [
            ("data/solutions_saa/k32_b0p5", "SAA32-b0p5"),
            ("data/solutions_saa/k32-b0P5", "SAA32-b0p5"),
        ],
        "SAA64-b0p7": [
            ("data/solutions_saa/k64_b0p7", "SAA64-b0p7"),
        ],
        "Q120": [
            ("data/solutions_quantile/m1.2_a0", "Q120"),
        ],
        "Q110": [
            ("data/solutions_quantile/m1.1_a0", "Q110"),
        ],
        "Gamma1-q1p645": [
            ("data/solutions_gamma/g1_q1p645_hybrid", "Gamma1"),
        ],
        "Gamma2-q1p645": [
            ("data/solutions_gamma/g2_q1p645_hybrid", "Gamma2"),
        ],
        "DET": [
            ("data/solutions_ortools", "DET"),
        ],
    }
    for d, lab in candidates.get(m, []):
        if Path(d).exists():
            return d, lab
    # Fallback to deterministic OR-Tools if nothing matched
    return "data/solutions_ortools", "DET"

def pick_method(pkl_path: Path, csv_path: Path) -> str:
    """
    Calls ml_pick_config.py as a subprocess and parses: 'Predicted method : <LABEL>'.
    Robust to missing deps (it should print DET in that case).
    """
    cmd = [sys.executable, "scripts/ml_pick_config.py", "--csv", str(csv_path)]
    if pkl_path and pkl_path.exists():
        cmd += ["--model", str(pkl_path)]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("[WARN] picker failed, defaulting to DET:", e)
        return "DET"

    pred = "DET"
    for line in out.splitlines():
        ls = line.strip()
        if ls.lower().startswith("predicted method"):
            # accept both "Predicted method : X" and "Predicted method: X"
            if ":" in ls:
                pred = ls.split(":", 1)[1].strip()
            else:
                parts = ls.split()
                pred = parts[-1] if parts else "DET"
            break
    return pred or "DET"

def _match_instance_stem(stem: str, inst: str) -> bool:
    """
    True if `stem` contains `inst` as a token separated by non-alnum boundaries.
    E.g., matches: C101, C101_plan, plan-C101, inst_C101
          does NOT match: RC101, C101R
    """
    s = stem.lower()
    i = inst.lower()
    pat = rf'(^|[^a-z0-9]){re.escape(i)}([^a-z0-9]|$)'
    return re.search(pat, s) is not None

def materialize_instance_subset(instance: str, method_dirs, labels, base_tmp="data/ml_runs/tmp"):
    """Create per-method temp dirs that contain ONLY the JSON(s) for this instance."""
    tmp_root = Path(base_tmp) / instance
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    out_dirs, out_labels = [], []

    for d, lab in zip(method_dirs, labels):
        src = Path(d)
        if not src.exists():
            print(f"[WARN] missing dir {src}, skipping {lab}")
            continue

        # Scan all JSONs; keep only stems that token-match the instance
        matches = [f for f in src.glob("*.json") if _match_instance_stem(f.stem, instance)]

        # de-dup while preserving order
        seen, uniq = set(), []
        for f in matches:
            if f not in seen:
                uniq.append(f); seen.add(f)

        if not uniq:
            print(f"[WARN] no plan for {instance} in {src}, skipping {lab}")
            continue

        dst = tmp_root / lab
        dst.mkdir(parents=True, exist_ok=True)
        for f in uniq:
            shutil.copy2(f, dst / f.name)
        out_dirs.append(str(dst))
        out_labels.append(lab)
    return out_dirs, out_labels

def run_eval(dirs, labels, K, seed, cvg, cvl, instance_name: str):
    tmp_dirs, tmp_labels = materialize_instance_subset(instance_name, dirs, labels)
    if not tmp_dirs:
        print(f"[ERROR] No candidate plans found for {instance_name}.")
        return
    # Build command
    cmd = [
        sys.executable, "scripts/evaluate_plans.py",
        "--dirs", *tmp_dirs,
        "--labels", *tmp_labels,
        "--K", str(K),
        "--seed", str(seed),
        "--cv_global", f"{cvg:.2f}",
        "--cv_link", f"{cvl:.2f}",
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def _dedup_preserve_order(xs):
    seen = set(); out = []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default="models/meta_selector.pkl")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_global", type=float, default=0.20)
    ap.add_argument("--cv_link", type=float, default=0.10)
    ap.add_argument("--fallbacks", nargs="*", default=["Q120", "Gamma1-q1p645"])
    ap.add_argument("--outdir", default="data/ml_runs")
    args = ap.parse_args()

    raw_dir = Path(args.csv).parent if args.csv else Path(args.raw_dir)
    model = Path(args.model)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(args.csv)] if args.csv else sorted(raw_dir.glob("*.csv"))
    rows = []

    for csv_path in csv_paths:
        instance = csv_path.stem.upper()
        print(f"\n=== {instance} ===")
        try:
            pred = pick_method(model, csv_path)
        except Exception as e:
            print("[WARN] picker failed, defaulting to DET:", e)
            pred = "DET"

        # Build evaluation list: predicted + fallbacks (dedup, keep order)
        methods = _dedup_preserve_order([pred] + [m for m in args.fallbacks if m])
        pairs = [method_to_dir_label(m) for m in methods]
        method_dirs = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

        run_eval(method_dirs, labels, args.K, args.seed, args.cv_global, args.cv_link, instance)
        rows.append({"instance": instance, "pred": pred, "evaluated": ";".join(labels)})

    pd.DataFrame(rows).to_csv(outdir / "ml_autorun_summary.csv", index=False)
    print("\n[OK] Wrote", outdir / "ml_autorun_summary.csv")

if __name__ == "__main__":
    main()
