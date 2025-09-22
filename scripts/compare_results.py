import pandas as pd
from pathlib import Path

BASE = Path("data/solutions/summary.csv")
NEW  = Path("data/solutions_ortools/summary.csv")
OUT  = Path("data/reports")
OUT.mkdir(parents=True, exist_ok=True)

def parse_group(name):
    # C101 -> C1, R210 -> R2, RC203 -> RC2, etc.
    fam = "".join([c for c in name if c.isalpha()])  # C, R, RC
    horizon = "1" if name[1] == "1" else "2"
    return f"{fam}{horizon}"

def main():
    base = pd.read_csv(BASE)
    new  = pd.read_csv(NEW)

    base["group"] = base["instance"].apply(parse_group)
    new["group"]  = new["instance"].apply(parse_group)

    merged = base.merge(new, on="instance", suffixes=("_baseline","_ortools"))
    merged["d_vehicles"] = merged["vehicles_baseline"] - merged["vehicles_ortools"]
    merged["d_distance"] = merged["total_distance_baseline"] - merged["total_distance_ortools"]
    merged["pct_distance"] = 100 * merged["d_distance"] / merged["total_distance_baseline"]

    merged.to_csv(OUT / "comparison.csv", index=False)

    print("Saved:", OUT / "comparison.csv")
    print("\nOverall:")
    print(merged[["d_vehicles","d_distance","pct_distance"]].mean().round(2))

    print("\nBy family:")
    merged["family"] = merged["instance"].str.extract(r"^(RC|R|C)")
    fam = merged.groupby("family")[["d_vehicles","d_distance","pct_distance"]].mean().round(2)
    print(fam)

    print("\nBy group (C1,C2,R1,R2,RC1,RC2):")
    grp = merged.groupby("group")[["d_vehicles","d_distance","pct_distance"]].mean().round(2)
    print(grp.sort_index())

if __name__ == "__main__":
    main()
