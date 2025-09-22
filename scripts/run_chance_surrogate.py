from pathlib import Path
import argparse, math, subprocess, shutil, time

BASE = Path(__file__).resolve().parents[1]
SOL_Q = BASE / "data" / "solutions_quantile"
SOL_CH = BASE / "data" / "solutions_chance"
SOL_CH.mkdir(parents=True, exist_ok=True)

Z = { 0.90:1.2815515655, 0.95:1.644853627, 0.975:1.959963984, 0.99:2.326347874 }

def ln1p(x):  # stable ln(1+x)
    return math.log1p(x)

def quantile_multiplier(alpha: float, cv_global: float, cv_link: float) -> float:
    s2 = ln1p(cv_global**2) + ln1p(cv_link**2)
    mu = -0.5 * s2
    sig = math.sqrt(s2)
    z = Z.get(alpha, 1.644853627)  # default ~95%
    return math.exp(mu + sig * z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.95, help="target service level, e.g. 0.95")
    ap.add_argument("--cv_global", type=float, default=0.20)
    ap.add_argument("--cv_link", type=float, default=0.10)
    ap.add_argument("--time_limit", type=int, default=30)
    ap.add_argument("--vehicle_cost", type=int, default=10000)
    ap.add_argument("--meta", default="GLS")
    ap.add_argument("--all", action="store_true", help="solve all 56 instances")
    ap.add_argument("--instance", help="single instance path (data/raw/C101.csv, etc.)")
    args = ap.parse_args()

    m = quantile_multiplier(args.alpha, args.cv_global, args.cv_link)
    tag = f"q{int(round(args.alpha*100))}"
    out_dir = SOL_CH / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) call the existing quantile solver with computed multiplier
    cmd = [
        "python", str(BASE/"scripts/vrptw_quantile.py"),
        "--mult", f"{m:.3f}",
        "--time_limit", str(args.time_limit),
        "--vehicle_cost", str(args.vehicle_cost),
        "--meta", args.meta
    ]
    if args.all:
        cmd.append("--all")
    elif args.instance:
        cmd += ["--instance", args.instance]
    else:
        raise SystemExit("Use --all or --instance")
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir/"run_stdout.txt").write_text(res.stdout)
    (out_dir/"run_stderr.txt").write_text(res.stderr)

    # 2) copy results from solutions_quantile into our chance folder
    #    (vrptw_quantile writes under data/solutions_quantile/m{mult}_a0)
    produced = max(SOL_Q.glob("m*_a0"), key=lambda p: p.stat().st_mtime)
    print("Copying from:", produced.name)
    shutil.copy2(produced/"summary.csv", out_dir/"summary.csv")
    dest = out_dir/"solutions"; dest.mkdir(exist_ok=True)
    for js in produced.glob("*.json"):
        shutil.copy2(js, dest/js.name)

    print(f"Done. Chance solutions in: {out_dir}")

if __name__ == "__main__":
    main()
