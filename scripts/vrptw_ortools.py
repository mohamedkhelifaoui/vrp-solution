import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# ---------- robust CSV reader ----------
def _to_num(x):
    """Extract the first numeric token from a messy cell, else NaN."""
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    # keep first number like -12, 3.45
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def read_instance(path: Path):
    # read as strings then coerce with our cleaner
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    needed = ["CUST NO.","XCOORD.","YCOORD.","DEMAND","READY TIME","DUE DATE","SERVICE TIME"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}")
        df[col] = df[col].map(_to_num)

    # drop any blank/garbage rows
    df = df.dropna(subset=["XCOORD.","YCOORD.","DEMAND","READY TIME","DUE DATE"]).copy()

    # Convert types
    df["DEMAND"] = df["DEMAND"].astype(int)
    # times are ints in Solomon; cast safely via round
    for c in ["READY TIME","DUE DATE","SERVICE TIME"]:
        df[c] = df[c].round().astype(int)

    # depot = first row where demand==0 (for these CSVs it's index 0)
    depot_idx = int(df.index[df["DEMAND"] == 0][0])

    nodes = df.reset_index(drop=True)
    depot = int(nodes.index[nodes["DEMAND"] == 0][0])
    coords = nodes[["XCOORD.","YCOORD."]].to_numpy(dtype=float)
    demand = nodes["DEMAND"].to_numpy(dtype=int)
    ready  = nodes["READY TIME"].to_numpy(dtype=int)
    due    = nodes["DUE DATE"].to_numpy(dtype=int)
    service= nodes["SERVICE TIME"].to_numpy(dtype=int)

    # Typical Solomon capacity and horizons are in the file; we can detect horizon
    horizon = int(due[depot])
    capacity = int(200)  # Solomon datasets use 200

    return {
        "name": Path(path).stem,
        "coords": coords,
        "demand": demand,
        "ready": ready,
        "due": due,
        "service": service,
        "depot": depot,
        "capacity": capacity,
        "horizon": horizon,
    }

# ---------- solver helpers ----------
def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def build_distance_matrix(coords):
    n = len(coords)
    dm = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j: 
                dm[i, j] = 0
            else:
                # cost and travel time are both rounded Euclidean
                dm[i, j] = int(round(euclidean(coords[i], coords[j])))
    return dm

def solve_vrptw(data, time_limit=10, vehicle_cost=10000, meta="GLS", max_vehicles=None):
    coords = data["coords"]
    n = len(coords)
    dist = build_distance_matrix(coords)

    depot = data["depot"]
    # allow many vehicles; fixed cost will discourage using them
    if max_vehicles is None:
        max_vehicles = 25  # enough for all 100-customer instances

    manager = pywrapcp.RoutingIndexManager(n, max_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # distance callback
    def dist_cb(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(dist[i][j])
    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # capacity
    demand = data["demand"]
    def demand_cb(from_index):
        i = manager.IndexToNode(from_index)
        return int(demand[i])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [data["capacity"]]*max_vehicles, True, "Capacity"
    )

    # time windows: travel time == distance; add service times
    ready, due, service = data["ready"], data["due"], data["service"]
    def time_cb(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(dist[i][j] + service[i])
    time_idx = routing.RegisterTransitCallback(time_cb)
    horizon = int(max(due))
    routing.AddDimension(
        time_idx,  # transit
        horizon,         # slack
        horizon,   # vehicle maximum
        True,      # force start at earliest
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # apply windows
    for node in range(n):
        idx = manager.NodeToIndex(node)
        time_dimension.CumulVar(idx).SetRange(int(ready[node]), int(due[node]))

    # penalize opening a vehicle
    routing.SetFixedCostOfAllVehicles(int(vehicle_cost))

    # search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.FromSeconds(int(time_limit))
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    meta = (meta or "").upper()
    meta_map = {
        "GLS": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SA": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "TS": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "GREEDY": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    }
    params.local_search_metaheuristic = meta_map.get(meta, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {"feasible": False}

    # extract routes (customer IDs only, no depot)
    routes = []
    used_vehicles = 0
    travel_distance = 0
    for v in range(max_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue  # unused
        used_vehicles += 1
        route_nodes = []
        prev = None
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            nxt = solution.Value(routing.NextVar(idx))
            next_node = manager.IndexToNode(nxt)
            if node != depot and node not in route_nodes:
                route_nodes.append(int(node))
            if prev is not None:
                travel_distance += dist[prev][node]
            prev = node
            idx = nxt
        # add last leg back to depot
        if prev is not None:
            travel_distance += dist[prev][depot]
        routes.append(route_nodes)

    objective_cost = solution.ObjectiveValue()
    return {
        "feasible": True,
        "vehicles": used_vehicles,
        "routes": routes,
        "objective_cost": float(objective_cost),
        "travel_distance": float(travel_distance),
    }

# ---------- IO ----------
def write_json(outdir: Path, name: str, data, meta):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance": name,
        "vehicles": data["vehicles"],
        "travel_distance": round(data["travel_distance"], 3),
        "objective_cost": round(data["objective_cost"], 3),
        "capacity": 200,
        "horizon": None,
        "feasible": data["feasible"],
        "routes": data["routes"],
        "meta": meta,
    }
    (outdir / f"{name}.json").write_text(json.dumps(payload, indent=2))

def append_summary(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.exists():
        base = pd.read_csv(csv_path)
        base = pd.concat([base, df], ignore_index=True)
        base.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", help="Path to one CSV instance")
    ap.add_argument("--all", action="store_true", help="Run on all data/raw/*.csv")
    ap.add_argument("--time_limit", type=int, default=10)
    ap.add_argument("--vehicle_cost", type=int, default=10000)
    ap.add_argument("--meta", type=str, default="GLS", help="GLS|SA|TS|GREEDY")
    args = ap.parse_args()

    out_dir = Path("data/solutions_ortools")
    summary = out_dir / "summary.csv"

    files = []
    if args.instance:
        files = [Path(args.instance)]
    elif args.all:
        files = sorted(Path("data/raw").glob("*.csv"))
    else:
        raise SystemExit("Use --instance <file> or --all")

    # reset summary if re-running --all
    if args.all and summary.exists():
        summary.unlink()

    for p in files:
        data_in = read_instance(p)
        res = solve_vrptw(
            data_in,
            time_limit=args.time_limit,
            vehicle_cost=args.vehicle_cost,
            meta=args.meta,
        )
        name = data_in["name"]
        if not res["feasible"]:
            print(f"{name}: NO SOLUTION")
            continue

        write_json(out_dir, name, res, args.meta)
        append_summary(
            summary,
            {
                "instance": name,
                "vehicles": res["vehicles"],
                "total_distance": round(res["travel_distance"], 3),
                "objective_cost": round(res["objective_cost"], 3),
                "capacity": 200,
                "horizon": data_in["horizon"],
                "feasible": True,
            },
        )
        # print both metrics
        print(f"{name}: vehicles={res['vehicles']:d}  dist={res['travel_distance']:.3f}  (objective={res['objective_cost']:.0f})  feasible=True")

    print(f"\nDone. Solutions in: {out_dir}")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
