import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def plot_one(csv_path: Path, outdir: Path, show: bool=False):
    inst = csv_path.stem  # e.g., R101
    sol_path = Path("data/solutions") / f"{inst}.json"
    if not sol_path.exists():
        print(f"[skip] no solution JSON: {sol_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Column names in Solomon csv
    idcol   = [c for c in df.columns if "CUST"   in c.upper()][0]
    xcol    = [c for c in df.columns if "XCOORD" in c.upper()][0]
    ycol    = [c for c in df.columns if "YCOORD" in c.upper()][0]
    demcol  = [c for c in df.columns if "DEMAND" in c.upper()][0]

    df = df.set_index(idcol)
    coords = df[[xcol, ycol, demcol]]

    # depot: demand==0 (Solomon)
    depot_id = coords.index[coords[demcol] == 0][0]

    with open(sol_path, "r", encoding="utf-8") as f:
        sol = json.load(f)

    if not show:
        matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(8,6))
    # plot customers
    ax.scatter(coords[xcol], coords[ycol], s=12)
    # highlight depot
    ax.scatter([coords.loc[depot_id, xcol]], [coords.loc[depot_id, ycol]], s=80, marker="s")

    # draw routes
    for r in sol["routes"]:
        seq = [depot_id] + r + [depot_id]
        xs = [coords.loc[i, xcol] for i in seq]
        ys = [coords.loc[i, ycol] for i in seq]
        ax.plot(xs, ys, linewidth=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{inst}: {sol['vehicles']} vehicles, dist={sol['total_distance']:.1f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{inst}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {outpath}")

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--instance", help="Path to one CSV instance (e.g., data/raw/R101.csv)")
    g.add_argument("--all", action="store_true", help="Plot for every instance in solutions/summary.csv")
    p.add_argument("--outdir", "--save", dest="outdir",
                   default="data/figures/routes",
                   help="Directory to save route plots (default: data/figures/routes)")
    p.add_argument("--show", action="store_true", help="Also display the figure window")
    args = p.parse_args()

    outdir = Path(args.outdir)

    if args.instance:
        plot_one(Path(args.instance), outdir, show=args.show)
    else:
        summary = pd.read_csv("data/solutions/summary.csv")
        for inst in summary["instance"]:
            csv_path = Path("data/raw") / f"{inst}.csv"
            if csv_path.exists():
                plot_one(csv_path, outdir, show=False)
            else:
                print(f"[skip] missing CSV: {csv_path}")

if __name__ == "__main__":
    main()
