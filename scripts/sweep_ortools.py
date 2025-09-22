from pathlib import Path
import subprocess, shutil, time, re

BASE = Path(__file__).resolve().parents[1]
SOL_DIR = BASE / "data" / "solutions_ortools"         # where vrptw_ortools.py writes
BENCH = BASE / "data" / "benchmarks"
BENCH.mkdir(parents=True, exist_ok=True)

# ====== TUNING GRID (edit if you like) ======
TIME_LIMITS   = [10, 30, 60]
METAHEUR      = ["GLS", "TABU"]        # add "SA" or "GD" only if your solver supports them
VEHICLE_COSTS = [0, 10000]             # 0 = “pure distance”, 10000 = “strongly penalize vehicles”
# ============================================

def rm_if_exists(p: Path):
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()

def run_one(tl, meta, vcost):
    tag = f"TL{tl}_M{meta}_V{vcost}"
    out_dir = BENCH / tag
    print(f"\n=== Running {tag} ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) run solver on all 56
    cmd = [
        "python", str(BASE / "scripts" / "vrptw_ortools.py"),
        "--all",
        "--time_limit", str(tl),
        "--vehicle_cost", str(vcost),
        "--meta", meta,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "run_stdout.txt").write_text(res.stdout)
    (out_dir / "run_stderr.txt").write_text(res.stderr)
    if res.returncode != 0:
        print(f"WARNING: process exited with code {res.returncode}")

    # 2) collect results from solutions_ortools
    src_sum = SOL_DIR / "summary.csv"
    if src_sum.exists():
        # copy summary
        shutil.copy2(src_sum, out_dir / "summary.csv")
    else:
        print("!! No summary.csv found – skipping copy.")

    # copy JSON solutions too (optional but useful)
    dest_solutions = out_dir / "solutions"
    rm_if_exists(dest_solutions)
    dest_solutions.mkdir(exist_ok=True)
    for js in SOL_DIR.glob("*.json"):
        shutil.copy2(js, dest_solutions / js.name)

def main():
    for tl in TIME_LIMITS:
        for meta in METAHEUR:
            for v in VEHICLE_COSTS:
                run_one(tl, meta, v)
                # a tiny pause to keep things tidy on Windows FS
                time.sleep(0.5)
    print("\nAll sweeps finished. Results in data/benchmarks/<TAG>/")

if __name__ == "__main__":
    main()
