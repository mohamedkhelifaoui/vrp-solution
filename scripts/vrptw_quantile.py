from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from math import hypot, ceil

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

BASE = Path(__file__).resolve().parents[1]
OUTROOT = BASE / "data" / "solutions_quantile"
OUTROOT.mkdir(parents=True, exist_ok=True)

def read_instance(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Normalize column names we expect
    cols = {
        "CUST NO.": "cust",
        "XCOORD.": "x",
        "YCOORD.": "y",
        "DEMAND": "demand",
        "READY TIME": "ready",
        "DUE DATE": "due",
        "SERVICE TIME": "service",
    }
    df = df.rename(columns=cols)
    # Coerce numeric
    for c in ["cust","x","y","demand","ready","due","service"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["cust","x","y","demand","ready","due","service"]).reset_index(drop=True)

    # Depot: demand==0 & service==0 (Solomon)
    depot_idx = int(df.index[(df["demand"]==0) & (df["service"]==0)][0])
    # Reorder so depot is index 0
    if depot_idx != 0:
        order = [depot_idx] + [i for i in range(len(df)) if i != depot_idx]
        df = df.iloc[order].reset_index(drop=True)

    # Family label (C/R/RC)
    name = Path(csv_path).stem
    family = "".join([ch for ch in name if ch.isalpha()])
    # Capacity and horizon (Solomon defaults)
    capacity = 200
    horizon = int(df["due"].max())

    # Dist/time matrix (Euclidean)
    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    base_time = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = xs - xs[i]
        dy = ys - ys[i]
        base_time[i, :] = np.hypot(dx, dy)

    return {
        "name": name,
        "family": family,
        "df": df,
        "base_time": base_time,
        "capacity": capacity,
        "horizon": horizon,
    }

def build_buffered_time(base_time: np.ndarray, mult: float, add: float):
    # Robust time matrix = base * mult + add (clip tiny negatives)
    M = base_time * float(mult) + float(add)
    M[M < 0] = 0.0
    return M

def solve_instance(csv_path: Path, time_mult: float, time_add: float,
                   time_limit: int, vehicle_cost: int, meta: str):
    data = read_instance(csv_path)
    name = data["name"]
    df = data["df"]
    n = len(df)

    base_time = data["base_time"]
    rob_time = build_buffered_time(base_time, time_mult, time_add)

    # OR-Tools expects integer arc costs. Scale and round.
    scale = 1.0  # keep as 1 for Solomon units
    time_mat_int = np.rint(rob_time * scale).astype(int)
    dist_base_int = np.rint(base_time * scale).astype(int)  # for reporting

    # Set vehicles upper bound (safe upper bound; vehicle fixed cost discourages extra)
    total_demand = df["demand"].sum()
    cap = data["capacity"]
    ub = int(min(30, ceil(total_demand / cap) + 10))
    num_vehicles = max(ub, 8)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Transit (travel + service at "to" node; depot has 0 service)
    service = df["service"].astype(int).to_numpy()
    def transit_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(time_mat_int[i, j] + (0 if j == 0 else service[j]))
    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Capacity
    demands = df["demand"].astype(int).to_numpy()
    def demand_cb(from_index):
        i = manager.IndexToNode(from_index)
        return int(demands[i])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [cap]*num_vehicles, True, "Capacity"
    )

    # Time dimension with windows
    horizon = int(data["horizon"])
    routing.AddDimension(
        transit_idx,  # transit
        0,            # no slack
        horizon,      # horizon
        True,         # start cumul to zero
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")
    ready = df["ready"].astype(int).to_numpy()
    due   = df["due"].astype(int).to_numpy()
    for node in range(n):
        idx = manager.NodeToIndex(node)
        # Force depot TW = [0, horizon]
        if node == 0:
            time_dim.CumulVar(idx).SetRange(0, horizon)
        else:
            time_dim.CumulVar(idx).SetRange(int(ready[node]), int(due[node]))

    # Vehicle fixed cost
    for v in range(num_vehicles):
        routing.SetFixedCostOfVehicle(int(vehicle_cost), v)

    # First solution strategy + LS meta
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    meta = (meta or "GLS").upper()
    if meta == "TABU":
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    else:
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.FromSeconds(int(time_limit))

    solution = routing.SolveWithParameters(search)
    out = {
        "instance": name,
        "buffer_mult": time_mult,
        "buffer_add": time_add,
        "vehicle_cost": vehicle_cost,
        "meta": meta,
        "time_limit": time_limit,
        "capacity": cap,
        "horizon": horizon,
    }

    if not solution:
        out.update({"feasible": False})
        return out

    # Extract routes
    routes = []
    used_vehicles = 0
    base_distance_sum = 0
    robust_time_sum = 0
    for v in range(num_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue  # unused vehicle
        used_vehicles += 1
        route_nodes = []
        prev = manager.IndexToNode(idx)
        idx = solution.Value(routing.NextVar(idx))
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route_nodes.append(int(df.loc[node, "cust"]))
            # add base distance and robust time
            base_distance_sum += int(dist_base_int[prev, node])
            robust_time_sum += int(time_mat_int[prev, node] + (0 if node == 0 else service[node]))
            prev, idx = node, solution.Value(routing.NextVar(idx))
        # return to depot
        base_distance_sum += int(dist_base_int[prev, 0])
        robust_time_sum += int(time_mat_int[prev, 0])

        if route_nodes:
            routes.append(route_nodes)

    out.update({
        "feasible": True,
        "vehicles": used_vehicles,
        "total_distance": float(base_distance_sum),  # in Solomon units (baseline distance)
        "robust_time_sum": float(robust_time_sum),
        "routes": routes
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=str, help="Path to one CSV instance")
    ap.add_argument("--all", action="store_true", help="Run on all 56 in data/raw")
    ap.add_argument("--mult", type=float, default=1.20, help="Time multiplier buffer (e.g., 1.20)")
    ap.add_argument("--add", type=float, default=0.0, help="Additive buffer per arc")
    ap.add_argument("--time_limit", type=int, default=30)
    ap.add_argument("--vehicle_cost", type=int, default=10000)
    ap.add_argument("--meta", type=str, default="GLS", choices=["GLS","TABU"])
    args = ap.parse_args()

    targets = []
    if args.all:
        for p in sorted((BASE / "data" / "raw").glob("*.csv")):
            targets.append(p)
    elif args.instance:
        targets = [Path(args.instance)]
    else:
        ap.error("Provide --instance or --all")

    # output dir tagged by buffer
    tag = f"m{args.mult:g}_a{args.add:g}"
    outdir = OUTROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in targets:
        res = solve_instance(p, args.mult, args.add, args.time_limit, args.vehicle_cost, args.meta)
        name = res.get("instance", Path(p).stem)
        (outdir / f"{name}.json").write_text(json.dumps(res, indent=2))
        if res.get("feasible", False):
            rows.append({
                "instance": name,
                "vehicles": res["vehicles"],
                "total_distance": res["total_distance"],
                "robust_time_sum": res["robust_time_sum"],
                "capacity": res["capacity"],
                "horizon": res["horizon"],
                "feasible": res["feasible"],
                "buffer_mult": args.mult,
                "buffer_add": args.add,
                "vehicle_cost": args.vehicle_cost,
                "meta": args.meta,
                "time_limit": args.time_limit,
                "objective": res["total_distance"] + args.vehicle_cost * res["vehicles"],
            })
        else:
            rows.append({
                "instance": name, "feasible": False,
                "buffer_mult": args.mult, "buffer_add": args.add,
                "vehicle_cost": args.vehicle_cost, "meta": args.meta,
                "time_limit": args.time_limit
            })

    # summary
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(outdir / "summary.csv", index=False)
    print(f"\nDone. Solutions in: {outdir}\nSummary: {outdir/'summary.csv'}")

if __name__ == "__main__":
    main()
