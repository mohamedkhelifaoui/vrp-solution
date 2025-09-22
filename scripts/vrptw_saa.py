from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUTROOT = BASE / "data" / "solutions_saa"
OUTROOT.mkdir(parents=True, exist_ok=True)

VEHICLE_CAPACITY = 200          # Solomon instances use Q=200
NUM_VEHICLES     = 30           # pool; fixed cost penalizes using many
SCALE            = 1            # keep ints simple

def read_instance(csv_path: Path):
    df = pd.read_csv(csv_path).rename(columns={
        "CUST NO.":"cust","XCOORD.":"x","YCOORD.":"y",
        "DEMAND":"demand","READY TIME":"ready","DUE DATE":"due","SERVICE TIME":"service"
    })
    for c in ["cust","x","y","demand","ready","due","service"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # put depot at index 0
    depot_idx = int(df.index[(df["demand"]==0) & (df["service"]==0)][0])
    if depot_idx != 0:
        order = [depot_idx] + [i for i in range(len(df)) if i != depot_idx]
        df = df.iloc[order].reset_index(drop=True)

    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    base_time = np.zeros((n,n), dtype=float)
    for i in range(n):
        dx = xs - xs[i]; dy = ys - ys[i]
        base_time[i,:] = np.hypot(dx, dy)
    return df, base_time

def lognormal_params_for_cv(cv):
    sigma2 = math.log(1.0 + cv*cv)
    sigma  = math.sqrt(sigma2)
    mu     = -0.5*sigma2
    return mu, sigma

def sample_scenarios(n, K, seed, cv_global, cv_link):
    rng = np.random.default_rng(seed)
    muG, sG = lognormal_params_for_cv(cv_global)
    muL, sL = lognormal_params_for_cv(cv_link)
    G = rng.lognormal(mean=muG, sigma=sG, size=K)         # (K,)
    E = rng.lognormal(mean=muL, sigma=sL, size=(K,n,n))   # (K,n,n)
    return G, E

def build_saa_matrix(base_time, K, seed, cv_global, cv_link, beta):
    n = base_time.shape[0]
    G, E = sample_scenarios(n, K, seed, cv_global, cv_link)
    Ts = np.empty((K, n, n), dtype=float)
    for s in range(K):
        Ts[s] = base_time * G[s] * E[s]
    mean = Ts.mean(axis=0)
    std  = Ts.std(axis=0, ddof=0)
    robust = mean + beta * std
    return robust

def solve_vrptw(df, time_mat, time_limit, vehicle_cost, meta):
    n = len(df)
    depot = 0

    travel = np.rint(time_mat * SCALE).astype(int)
    service = (df["service"].to_numpy()*SCALE).astype(int)
    demand  = df["demand"].to_numpy(dtype=int)
    ready   = (df["ready"].to_numpy()*SCALE).astype(int)
    due     = (df["due"].to_numpy()*SCALE).astype(int)

    manager = pywrapcp.RoutingIndexManager(n, NUM_VEHICLES, depot)
    routing = pywrapcp.RoutingModel(manager)

    # ---- Cost callback = travel only (so objective is distance + fixed vehicle costs)
    def cost_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(travel[i, j])
    cost_idx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

    # ---- Time dimension transit = travel + service at 'from' node (classic pattern)
    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(travel[i, j] + service[i])
    time_idx = routing.RegisterTransitCallback(time_cb)

    # Capacity
    demand_idx = routing.RegisterUnaryTransitCallback(lambda idx: int(demand[manager.IndexToNode(idx)]))
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [VEHICLE_CAPACITY]*NUM_VEHICLES, True, "Capacity"
    )

    # Time windows
    # Use a generous max to avoid artificial infeasibility; windows will still restrict visits.
    routing.AddDimension(
        time_idx,
        10**7,            # huge slack
        10**7,            # huge horizon cap
        True,             # start cumul at 0
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # DO NOT fix SlackVar to service (we already included service in transit).
    # Just set windows on cumul.
    for node in range(n):
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(int(ready[node]), int(due[node]))

    # Vehicle fixed cost
    for v in range(NUM_VEHICLES):
        routing.SetFixedCostOfVehicle(int(vehicle_cost), v)

    # Search
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.time_limit.FromSeconds(int(time_limit))
    search.log_search = False
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    meta = meta.upper()
    if meta == "GLS":
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    elif meta == "TABU":
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    else:
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT

    solution = routing.SolveWithParameters(search)
    if solution is None:
        return {"feasible": False}

    # Extract routes + travel distance (not counting service)
    routes = []
    used_vehicles = 0
    total_travel = 0
    for v in range(NUM_VEHICLES):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue
        used_vehicles += 1
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            nxt  = solution.Value(routing.NextVar(idx))
            nxt_node = manager.IndexToNode(nxt)
            if nxt_node != 0:
                route.append(int(df.loc[nxt_node, "cust"]))
            total_travel += travel[node, nxt_node]
            idx = nxt
        routes.append(route)

    return {
        "feasible": True,
        "vehicles": used_vehicles,
        "total_distance": float(total_travel / SCALE),
        "routes": routes
    }

def run_instance(csv_path: Path, out_dir: Path, K, seed, cvg, cvl, beta, time_limit, vehicle_cost, meta):
    df, base_time = read_instance(csv_path)
    robust = build_saa_matrix(base_time, K=K, seed=seed, cv_global=cvg, cv_link=cvl, beta=beta)
    res = solve_vrptw(df, robust, time_limit=time_limit, vehicle_cost=vehicle_cost, meta=meta)

    inst = csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if res["feasible"]:
        js = {
            "instance": inst,
            "vehicles": res["vehicles"],
            "total_distance": round(res["total_distance"], 3),
            "capacity": VEHICLE_CAPACITY,
            "horizon": float(df["due"].max()),
            "feasible": True,
            "routes": res["routes"]
        }
    else:
        js = {"instance": inst, "feasible": False}

    (out_dir / f"{inst}.json").write_text(json.dumps(js, indent=2))
    return inst, js

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=str, help="Path to one CSV (optional if --all)")
    ap.add_argument("--all", action="store_true", help="Run on all 56 instances in data/raw")
    ap.add_argument("--K", type=int, default=32, help="# scenarios for SAA")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cv_global", type=float, default=0.20)
    ap.add_argument("--cv_link", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=0.5, help="risk-aversion: mean + beta*std")
    ap.add_argument("--time_limit", type=int, default=30)
    ap.add_argument("--vehicle_cost", type=int, default=10000)
    ap.add_argument("--meta", type=str, default="GLS", choices=["GLS","TABU","GD"])
    args = ap.parse_args()

    tag = f"k{args.K}_b{args.beta}".replace(".","p")
    out_dir = OUTROOT / tag
    summary_rows = []

    paths = []
    if args.all:
        paths = sorted(RAW.glob("*.csv"))
    elif args.instance:
        paths = [Path(args.instance)]
    else:
        raise SystemExit("Provide --instance or --all")

    for p in paths:
        inst, js = run_instance(
            p, out_dir,
            K=args.K, seed=args.seed, cvg=args.cv_global, cvl=args.cv_link, beta=args.beta,
            time_limit=args.time_limit, vehicle_cost=args.vehicle_cost, meta=args.meta
        )
        if js.get("feasible", False):
            print(f"{inst}: vehicles={js['vehicles']}  dist={js['total_distance']:.3f}  feasible=True")
            summary_rows.append({
                "instance": inst,
                "vehicles": js["vehicles"],
                "total_distance": js["total_distance"],
                "capacity": js["capacity"],
                "horizon": js["horizon"],
                "feasible": True
            })
        else:
            print(f"{inst}: NO SOLUTION")
            summary_rows.append({
                "instance": inst,
                "vehicles": np.nan,
                "total_distance": np.nan,
                "capacity": np.nan,
                "horizon": np.nan,
                "feasible": False
            })

    pd.DataFrame(summary_rows).to_csv(out_dir/"summary.csv", index=False)
    print("\nDone. Solutions in:", out_dir)
    print("Summary:", out_dir/"summary.csv")

if __name__ == "__main__":
    main()
