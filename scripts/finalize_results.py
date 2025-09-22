# scripts/finalize_results.py
from pathlib import Path
import argparse, shutil, json, time

BASE = Path(__file__).resolve().parents[1]
BENCH = BASE / "data" / "benchmarks"
FINAL = BASE / "data" / "final"

def rm_if_exists(p: Path):
    if p.is_symlink() or p.is_file():
        p.unlink(missing_ok=True)
    elif p.is_dir():
        shutil.rmtree(p, ignore_errors=True)

def copy_tree(src: Path, dst: Path):
    rm_if_exists(dst)
    shutil.copytree(src, dst)

def main():
    ap = argparse.ArgumentParser(description="Freeze a benchmark tag as FINAL results.")
    ap.add_argument("--tag", required=True, help="e.g., TL60_MGLS_V0")
    args = ap.parse_args()

    tag_dir = BENCH / args.tag
    if not tag_dir.exists():
        raise SystemExit(f"Tag folder not found: {tag_dir}")

    sumf = tag_dir / "summary.csv"
    sol_src = tag_dir / "solutions"
    if not sumf.exists():
        raise SystemExit(f"Missing {sumf}")
    if not sol_src.exists():
        print("WARN: solutions/ folder not found under the tag; continuing with summary only.")

    # prepare target
    FINAL.mkdir(parents=True, exist_ok=True)
    final_tag = FINAL / args.tag
    rm_if_exists(final_tag)
    final_tag.mkdir(parents=True, exist_ok=True)

    # copy summary + solutions
    shutil.copy2(sumf, final_tag / "summary.csv")
    if sol_src.exists():
        shutil.copytree(sol_src, final_tag / "solutions")

    # small info file
    info = {
        "tag": args.tag,
        "source": str(tag_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (final_tag / "INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # make a convenient "current" folder that mirrors the chosen tag
    current = FINAL / "current"
    copy_tree(final_tag, current)

    # small readme
    (FINAL / "FINAL_README.txt").write_text(
        f"Final deterministic results are in: {final_tag}\n"
        f"'current' is a copy of that folder for scripts/plots to consume.\n",
        encoding="utf-8"
    )

    print("Finalized deterministic results.")
    print(f" - {final_tag}")
    print(f" - {current}")

if __name__ == "__main__":
    main()
