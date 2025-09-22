from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUTROOT = BASE / "data" / "solutions_gamma"
OUTROOT.mkdir(parents=True, exist_ok=True)

CAPACITY = 200  # Solomon
DEPOT = 0

def read_instance(csv_path: Path):
    df = pd.read_csv(csv_path).rename(columns={
        "CUST NO.":"cust","XCOORD.":"x","YCOORD.":"y",
        "DEMAND":"demand","READY TIME":"ready","DUE DATE":"due","SERVICE TIME":"service"
    })
    for c in ["cust","x","y","demand","ready","due","service"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # make sure depot is first row
    depot_idx = int(df.index[(df["demand"]==0) & (df["service"]==0)][0])
    if depot_idx != 0:
        order = [depot_idx] + [i for i in range(len(df)) if i != depot_idx]
        df = df.iloc[order].reset_index(drop=True)
    # euclidean time matrix from coordinates
    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    base_time = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = xs - xs[i]; dy = ys - ys[i]
        base_time[i, :] = np.hypot(dx, dy)
    return df, base_time

def build_effective(df, base_time, gamma, q=1.645, cv_global=0.20, cv_link=0.10, mode="hybrid"):
    """Return (time_matrix_eff, ready_eff, due_eff)."""
    # 1) inflate matrix for global traffic at quantile q
    mult_global = 1.0 + q * cv_global
    T = base_time * mult_global

    # 2) compute Γ-buffer for link uncertainty (worst Γ arcs)
    # pick a route-independent reference arc time
    nonzero = base_time[base_time > 0]
    if nonzero.size == 0:
        t_ref = 0.0
    else:
        p = 75 if mode.lower()=="hybrid" else 90
        t_ref = float(np.percentile(nonzero, p))
    B = gamma * q * cv_link * t_ref  # buffer in "time units"

    ready = df["ready"].to_numpy().astype(float)
    due   = df["due"].to_numpy().astype(float)
    # tighten due dates but never before ready
    due_eff = np.maximum(ready, due - B)
    return T, ready, due_eff

def solve_vrptw(df, time_matrix, ready, due, time_limit=30, vehicle_cost=10000, meta="GLS"):
    n = len(df)
    # generous upper bound on vehicles; fixed cost will push to fewer
    max_vehicles = 30

    manager = pywrapcp.RoutingIndexManager(n, max_vehicles, DEPOT)
    routing = pywrapcp.RoutingModel(manager)

    # --- distance callback (pure euclidean, for objective) ---
    dist = time_matrix.copy()
    # we want objective on distance (travel only)
    def distance_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(dist[i, j]))
    distance_cb_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_cb_idx)

    # --- time callback (travel + service at origin node) ---
    service = df["service"].to_numpy().astype(float)
    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(dist[i, j] + service[i]))
    time_cb_idx = routing.RegisterTransitCallback(time_cb)

    # --- capacity dimension ---
    demand = df["demand"].to_numpy().astype(int)
    def demand_cb(from_index):
        i = manager.IndexToNode(from_index)
        return int(demand[i])
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,                              # null slack
        [CAPACITY]*max_vehicles,        # capacities
        True,                           # start cumul to zero
        "Capacity"
    )

    # --- time dimension & windows ---
    horizon = int(max(due) + 1000) if len(due) else 10000
    routing.AddDimension(
        time_cb_idx,
        1000000,                        # huge slack
        horizon,
        True,                           # start cumul to zero
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(n):
        idx = manager.NodeToIndex(node)
        r = int(round(ready[node]))
        d = int(round(due[node]))
        if d < r: d = r
        time_dim.CumulVar(idx).SetRange(r, d)

    # --- vehicle fixed cost to penalize extra vehicles ---
    for v in range(max_vehicles):
        routing.SetFixedCostOfVehicle(int(vehicle_cost), v)

    # --- search params ---
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.time_limit.FromSeconds(int(time_limit))
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    meta = meta.upper()
    if meta == "GLS":
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    elif meta == "TABU":
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    else:
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC

    # --- solve ---
    solution = routing.SolveWithParameters(search)
    if solution is None:
        return False, 0, 0.0, []

    # --- extract routes ---
    routes = []
    total_dist = 0.0
    vehicles_used = 0
    for v in range(max_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue
        vehicles_used += 1
        r = []
        prev = manager.IndexToNode(idx)
        while not routing.IsEnd(idx):
            nxt = solution.Value(routing.NextVar(idx))
            j = manager.IndexToNode(nxt)
            if j != DEPOT:
                r.append(int(df.loc[j, "cust"]))
            # accumulate travel distance (pure)
            total_dist += dist[prev, j]
            prev = j
            idx = nxt
        # return to depot already counted in last leg above
        routes.append(r)

    return True, vehicles_used, float(total_dist), routes

def write_json(out_dir, inst_name, ok, veh, dist, horizon, routes):
    out = {
        "instance": inst_name,
        "vehicles": veh,
        "total_distance": round(dist, 3),
        "capacity": CAPACITY,
        "horizon": float(horizon),
        "feasible": bool(ok),
        "routes": routes
    }
    (out_dir / f"{inst_name}.json").write_text(json.dumps(out, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", help="Path to one CSV instance")
    ap.add_argument("--all", action="store_true", help="Run all 56 instances")
    ap.add_argument("--gamma", type=int, default=1, help="Route budget Γ (1..3 typical)")
    ap.add_argument("--mode", choices=["hybrid","static"], default="hybrid")
    ap.add_argument("--q", type=float, default=1.645, help="Normal quantile (≈1.645 for ~95%)")
    ap.add_argument("--cv_global", type=float, default=0.20)
    ap.add_argument("--cv_link", type=float, default=0.10)
    ap.add_argument("--time_limit", type=int, default=30)
    ap.add_argument("--vehicle_cost", type=int, default=10000)
    ap.add_argument("--meta", default="GLS")
    args = ap.parse_args()

    tag = f"g{args.gamma}_q{str(args.q).replace('.','p')}_{args.mode.lower()}"
    out_dir = OUTROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if args.all:
        paths = sorted(RAW.glob("*.csv"))
    elif args.instance:
        paths = [Path(args.instance)]
    else:
        raise SystemExit("Provide --instance or --all")

    rows = []
    for p in paths:
        inst = p.stem
        df, base = read_instance(p)
        T, ready, due_eff = build_effective(df, base, args.gamma, args.q, args.cv_global, args.cv_link, args.mode)
        ok, veh, dist, routes = solve_vrptw(df, T, ready, due_eff,
                                            time_limit=args.time_limit,
                                            vehicle_cost=args.vehicle_cost,
                                            meta=args.meta)
        horizon = float(due_eff.max()) if len(due_eff) else 0.0
        if ok:
            print(f"{inst}: vehicles={veh:2d}  dist={dist:.3f}  feasible=True")
        else:
            print(f"{inst}: NO SOLUTION")
        write_json(out_dir, inst, ok, veh, dist, horizon, routes)
        rows.append({
            "instance": inst,
            "vehicles": veh if ok else np.nan,
            "total_distance": dist if ok else np.nan,
            "capacity": CAPACITY,
            "horizon": horizon,
            "feasible": bool(ok),
            "tag": tag
        })

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print("\nDone. Solutions in:", out_dir)
    print("Summary:", out_dir / "summary.csv")

if __name__ == "__main__":
    main()
