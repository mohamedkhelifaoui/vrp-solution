from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

REPORTS = Path("data/reports")
FIGS = Path("data/figures")
SUMMARY = Path("data/solutions/summary.csv")

def parse_family_hgroup(inst: str):
    """
    Parse 'C101','R204','RC107', etc. -> family ('C','R','RC') and horizon group (1 or 2).
    We rely on the Solomon naming where the first digit after the family is 1 or 2.
    """
    m = re.match(r'^([A-Z]+?)([12])\d+$', inst.upper())
    if m:
        fam, h = m.group(1), int(m.group(2))
        return pd.Series({"family": fam, "hgroup": h})
    # Fallback: infer family from leading letters, leave hgroup NA
    m2 = re.match(r'^([A-Z]+)', inst.upper())
    fam = m2.group(1) if m2 else "UNK"
    return pd.Series({"family": fam, "hgroup": pd.NA})

def agg_stats(df: pd.DataFrame, group_cols):
    out = (
        df.groupby(group_cols, dropna=False)
          .agg(n_instances=("instance", "count"),
               vehicles_mean=("vehicles", "mean"),
               vehicles_std=("vehicles", "std"),
               vehicles_min=("vehicles", "min"),
               vehicles_max=("vehicles", "max"),
               distance_mean=("total_distance", "mean"),
               distance_std=("total_distance", "std"),
               distance_min=("total_distance", "min"),
               distance_max=("total_distance", "max"))
          .reset_index()
    )
    return out

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY)
    # Parse family (C/R/RC) and horizon group (1/2) from the instance name
    parsed = df["instance"].apply(parse_family_hgroup)
    df = pd.concat([df, parsed], axis=1)
    df["group"] = df["family"].astype(str) + df["hgroup"].astype("Int64").astype(str)

    # --- Tables ---
    fam_tbl = agg_stats(df, ["family"]).round(3)
    fam_tbl.to_csv(REPORTS / "family_summary.csv", index=False)

    fam_h_tbl = agg_stats(df, ["family", "hgroup"]).round(3)
    # Add a convenience label column like C1, R2, etc.
    fam_h_tbl.insert(2, "label", fam_h_tbl["family"] + fam_h_tbl["hgroup"].astype("Int64").astype(str))
    fam_h_tbl.to_csv(REPORTS / "family_horizon_summary.csv", index=False)

    # Write a tiny overall summary text
    with open(REPORTS / "overall_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Instances: {len(df)}\n")
        f.write(f"Feasible: {int(df['feasible'].sum())}/{len(df)}\n")
        f.write("Families: " + ", ".join(sorted(df['family'].unique())) + "\n")
        f.write("Groups: " + ", ".join(sorted(df['group'].dropna().unique())) + "\n")

    # --- Figures (family means) ---
    fam_means = df.groupby("family", as_index=False).agg(
        vehicles_mean=("vehicles", "mean"),
        distance_mean=("total_distance", "mean")
    )

    plt.figure()
    plt.bar(fam_means["family"], fam_means["distance_mean"])
    plt.title("Average total distance by family")
    plt.xlabel("Family (C/R/RC)")
    plt.ylabel("Avg. distance")
    plt.tight_layout()
    plt.savefig(FIGS / "avg_distance_by_family.png", dpi=150)
    plt.close()

    plt.figure()
    plt.bar(fam_means["family"], fam_means["vehicles_mean"])
    plt.title("Average number of vehicles by family")
    plt.xlabel("Family (C/R/RC)")
    plt.ylabel("Avg. vehicles")
    plt.tight_layout()
    plt.savefig(FIGS / "avg_vehicles_by_family.png", dpi=150)
    plt.close()

    # --- Figures (family + horizon = C1,C2,...) ---
    grp_means = df.groupby(["family", "hgroup"], as_index=False).agg(
        vehicles_mean=("vehicles", "mean"),
        distance_mean=("total_distance", "mean")
    )
    grp_means["label"] = grp_means["family"] + grp_means["hgroup"].astype("Int64").astype(str)

    plt.figure()
    plt.bar(grp_means["label"], grp_means["distance_mean"])
    plt.title("Average total distance by group (C1,C2,R1,R2,RC1,RC2)")
    plt.xlabel("Group")
    plt.ylabel("Avg. distance")
    plt.tight_layout()
    plt.savefig(FIGS / "avg_distance_by_group.png", dpi=150)
    plt.close()

    plt.figure()
    plt.bar(grp_means["label"], grp_means["vehicles_mean"])
    plt.title("Average number of vehicles by group (C1,C2,R1,R2,RC1,RC2)")
    plt.xlabel("Group")
    plt.ylabel("Avg. vehicles")
    plt.tight_layout()
    plt.savefig(FIGS / "avg_vehicles_by_group.png", dpi=150)
    plt.close()

    # --- Scatter for all instances ---
    plt.figure()
    plt.scatter(df["vehicles"], df["total_distance"])
    for _, r in df.iterrows():
        plt.annotate(r["instance"], (r["vehicles"], r["total_distance"]), fontsize=6, xytext=(3,3), textcoords="offset points")
    plt.title("Vehicles vs Total Distance (all instances)")
    plt.xlabel("Vehicles")
    plt.ylabel("Total distance")
    plt.tight_layout()
    plt.savefig(FIGS / "vehicles_vs_distance_scatter.png", dpi=150)
    plt.close()

    print("Wrote:")
    print(REPORTS / "family_summary.csv")
    print(REPORTS / "family_horizon_summary.csv")
    print(REPORTS / "overall_summary.txt")
    print(FIGS / "avg_distance_by_family.png")
    print(FIGS / "avg_vehicles_by_family.png")
    print(FIGS / "avg_distance_by_group.png")
    print(FIGS / "avg_vehicles_by_group.png")
    print(FIGS / "vehicles_vs_distance_scatter.png")

if __name__ == "__main__":
    main()
