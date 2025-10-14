from pathlib import Path
import argparse, subprocess, sys

BASE = Path(__file__).resolve().parents[1]

def run_one(csv: Path, outdir: Path):
    cmd = [
        sys.executable, str(BASE/"scripts"/"ml_predict_and_run.py"),
        "--csv", str(csv),
        "--K", "200", "--seed", "42",
        "--cv_global", "0.20", "--cv_link", "0.10",
        "--fallbacks", "Q120", "Gamma1-q1p645",
        "--outdir", str(outdir)
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder with *.csv")
    ap.add_argument("--outdir", default=str(BASE/"data/solutions_champions/new"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for csv in Path(args.dir).glob("*.csv"):
        run_one(csv, outdir)

if __name__ == "__main__":
    main()
