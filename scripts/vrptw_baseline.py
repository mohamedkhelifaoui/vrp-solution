#!/usr/bin/env python3
"""
VRPTW baseline (Solomon-style):
- Input: your original CSVs (C/R/RC*) in data/raw
- Assumes travel_time == Euclidean distance units (Solomon convention)
- Default vehicle capacity = 200 (can be changed via --capacity)
- Builds routes using cheapest-feasible insertion, then intra-route 2-opt
- Writes per-instance JSON and a summary CSV

Run examples:
  python scripts/vrptw_baseline.py --instance data/raw/C101.csv
  python scripts/vrptw_baseline.py --all
  python scripts/vrptw_baseline.py --all --capacity 200
"""
from __future__ import annotations
import argparse, csv, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import pandas as pd
except Exception as e:
    pd = None

# Column name fallbacks (your files already match these)
HEADER_MAP = {
    "id":      ["CUST NO.", "CUST NO", "ID", "CustNo", "Customer"],
    "x":       ["XCOORD.", "XCOORD", "X", "X_COORD"],
    "y":       ["YCOORD.", "YCOORD", "Y", "Y_COORD"],
    "demand":  ["DEMAND", "Demand"],
    "a":       ["READY TIME", "READY_TIME", "READY"],
    "b":       ["DUE DATE", "DUE_DATE", "DUE"],
    "service": ["SERVICE TIME", "SERVICE_TIME", "SERVICETIME", "service"],
}

@dataclass
class Node:
    idx: int         # internal index (0..n-1), depot is 0
    cust_id: int     # original "CUST NO." value
    x: float
    y: float
    demand: float
    a: float         # ready
    b: float         # due
    service: float

@dataclass
class Instance:
    name: str
    depot: Node
    customers: List[Node]   # excludes depot
    horizon: float          # depot due (latest return time)
    capacity: float

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_instance(path: Path, capacity: float) -> Instance:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")

    df = pd.read_csv(path)
    cols = {k: find_col(df, v) for k, v in HEADER_MAP.items()}
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise ValueError(f"Missing expected columns: {missing} in {path}")

    df = df.rename(columns={
        cols["id"]: "id", cols["x"]: "x", cols["y"]: "y",
        cols["demand"]: "demand", cols["a"]: "a", cols["b"]: "b",
        cols["service"]: "service",
    }).copy()

    # Coerce numerics
    for c in ["id","x","y","demand","a","b","service"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["id","x","y","demand","a","b","service"])

    # Identify depot: demand == 0 and service == 0 (unique)
    depot_rows = df[(df["demand"]==0) & (df["service"]==0)]
    if len(depot_rows) != 1:
        raise ValueError(f"Depot rows != 1 (found {len(depot_rows)}) in {path}")
    depot_row = depot_rows.iloc[0]

    # Sort by id to keep original order stable (not required)
    df = df.sort_values("id").reset_index(drop=True)

    # Build nodes (depot first as idx=0)
    nodes: List[Node] = []
    for _, r in df.iterrows():
        nodes.append(Node(
            idx=len(nodes),
            cust_id=int(r["id"]),
            x=float(r["x"]), y=float(r["y"]),
            demand=float(r["demand"]),
            a=float(r["a"]), b=float(r["b"]),
            service=float(r["service"]),
        ))
    # Move depot to index 0 if needed
    if int(depot_row["id"]) != nodes[0].cust_id:
        # find depot in nodes
        depot_idx = next(i for i, n in enumerate(nodes) if n.demand==0 and n.service==0)
        # swap into position 0
        nodes[0], nodes[depot_idx] = nodes[depot_idx], nodes[0]
        # fix indices
        for i, n in enumerate(nodes):
            n.idx = i

    depot = nodes[0]
    customers = nodes[1:]
    horizon = depot.b  # latest return time
    return Instance(name=path.stem, depot=depot, customers=customers, horizon=horizon, capacity=capacity)

def euclid(a: Node, b: Node) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)

def route_distance(nodes: List[Node], route: List[int]) -> float:
    # route contains node indices (internal), assumed starts/ends with depot (0)
    dist = 0.0
    for i in range(len(route)-1):
        dist += euclid(nodes[route[i]], nodes[route[i+1]])
    return dist

def check_schedule(nodes: List[Node], route: List[int]) -> Tuple[bool, float, List[float]]:
    """
    Returns (feasible, total_distance, start_times)
    Feasible if:
      - each visit starts within [a,b]
      - end time at final depot <= depot.b (horizon)
    """
    t = 0.0
    start_times: List[float] = []
    total_dist = 0.0
    for i in range(len(route)-1):
        cur = nodes[route[i]]
        nxt = nodes[route[i+1]]
        travel = euclid(cur, nxt)
        total_dist += travel
        arrival = t + travel
        start = max(arrival, nxt.a)
        if start > nxt.b + 1e-9:
            return False, total_dist, start_times
        start_times.append(start)
        t = start + nxt.service
    # t now is after service at final node (which is depot if route ends at 0)
    # For depot at end, we already checked time window at depot via above loop.
    return True, total_dist, start_times

def route_load(nodes: List[Node], route: List[int]) -> float:
    # sum of demands excluding depot indices
    return sum(nodes[i].demand for i in route if i != 0)

def try_insert(nodes: List[Node], base_route: List[int], insert_node: int, capacity: float, horizon: float) -> Optional[Tuple[float, List[int]]]:
    """Try all positions to insert `insert_node` in base_route. Return (delta_cost, new_route) or None if infeasible."""
    best = None
    for pos in range(1, len(base_route)):  # between nodes; keeps depot at ends
        new_route = base_route[:pos] + [insert_node] + base_route[pos:]
        if route_load(nodes, new_route) > capacity + 1e-9:
            continue
        feasible, _, _ = check_schedule(nodes, new_route)
        if not feasible:
            continue
        delta = route_distance(nodes, new_route) - route_distance(nodes, base_route)
        if (best is None) or (delta < best[0] - 1e-9):
            best = (delta, new_route)
    return best

def two_opt_feasible(nodes: List[Node], route: List[int]) -> List[int]:
    """Intra-route 2-opt that preserves depot endpoints and time windows."""
    best_route = route[:]
    best_dist = route_distance(nodes, best_route)
    improved = True
    while improved:
        improved = False
        # i and k are breakpoints; keep start/end depot fixed
        for i in range(1, len(best_route)-2):
            for k in range(i+1, len(best_route)-1):
                # 2-opt swap
                new_route = best_route[:i] + list(reversed(best_route[i:k+1])) + best_route[k+1:]
                feasible, _, _ = check_schedule(nodes, new_route)
                if not feasible:
                    continue
                new_dist = route_distance(nodes, new_route)
                if new_dist + 1e-9 < best_dist:
                    best_route, best_dist = new_route, new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route

def solve_instance(inst: Instance) -> Dict:
    nodes = [inst.depot] + inst.customers  # ensure indices match Node.idx
    unrouted = set(n.idx for n in inst.customers)  # internal ids 1..N
    routes: List[List[int]] = []

    # Greedy parallel cheapest-feasible insertion
    while unrouted:
        best_move = None  # (delta, route_index/new, new_route, node_id)
        for node_id in list(unrouted):
            # try into existing routes
            for r_idx, r in enumerate(routes):
                ins = try_insert(nodes, r, node_id, inst.capacity, inst.horizon)
                if ins:
                    delta, new_r = ins
                    if (best_move is None) or (delta < best_move[0] - 1e-9):
                        best_move = (delta, r_idx, new_r, node_id)
            # also try opening a new route [0, node, 0]
            new_r = [0, node_id, 0]
            # capacity/time check
            if route_load(nodes, new_r) <= inst.capacity + 1e-9:
                feasible, _, _ = check_schedule(nodes, new_r)
                if feasible:
                    delta = route_distance(nodes, new_r)  # cost of new route
                    if (best_move is None) or (delta < best_move[0] - 1e-9):
                        best_move = (delta, "NEW", new_r, node_id)

        if best_move is None:
            # If we reach here, something is individually infeasible (shouldn't happen for Solomon)
            # Fallback: take an arbitrary node and force a single-customer route (may violate time window)
            nid = unrouted.pop()
            routes.append([0, nid, 0])
            continue

        _, where, new_r, nid = best_move
        if where == "NEW":
            routes.append(new_r)
        else:
            routes[where] = new_r
        unrouted.remove(nid)

    # Improve each route with feasible 2-opt
    routes = [two_opt_feasible(nodes, r) for r in routes]

    # Final metrics
    total_dist = sum(route_distance(nodes, r) for r in routes)
    feasible_all = True
    for r in routes:
        feas, _, _ = check_schedule(nodes, r)
        if not feas or route_load(nodes, r) > inst.capacity + 1e-9:
            feasible_all = False
            break

    # Export routes as original customer IDs (exclude depots)
    routes_by_ids = [[nodes[i].cust_id for i in r if i != 0] for r in routes]

    return {
        "instance": inst.name,
        "vehicles": len(routes),
        "total_distance": round(total_dist, 3),
        "capacity": inst.capacity,
        "horizon": inst.horizon,
        "feasible": bool(feasible_all),
        "routes": routes_by_ids,
    }

def save_json(obj: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def append_summary_row(summary_csv: Path, row: Dict):
    new_file = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instance","vehicles","total_distance","capacity","horizon","feasible"])
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in w.fieldnames})

def autodetect_capacity(name: str, default_cap: int) -> int:
    # You can specialize per family if you want; Solomon typically uses 200.
    return default_cap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=str, help="Path to a single CSV (e.g., data/raw/C101.csv)")
    ap.add_argument("--all", action="store_true", help="Run all CSVs in data/raw")
    ap.add_argument("--capacity", type=int, default=200, help="Vehicle capacity (default 200)")
    ap.add_argument("--outdir", type=str, default="data/solutions", help="Output dir for JSON + summary.csv")
    args = ap.parse_args()

    raw_dir = Path("data/raw")
    out_dir = Path(args.outdir)
    summary_csv = out_dir / "summary.csv"

    if args.instance:
        paths = [Path(args.instance)]
    elif args.all:
        paths = sorted(raw_dir.glob("*.csv"))
    else:
        print("Please pass --instance data/raw/C101.csv  or  --all")
        return

    # reset summary if running many
    if args.all and summary_csv.exists():
        summary_csv.unlink()

    for p in paths:
        cap = autodetect_capacity(p.stem.upper(), args.capacity)
        inst = load_instance(p, capacity=cap)
        sol = solve_instance(inst)
        save_json(sol, out_dir / f"{inst.name}.json")
        append_summary_row(summary_csv, sol)
        print(f"{inst.name}: vehicles={sol['vehicles']}  dist={sol['total_distance']}  feasible={sol['feasible']}")

    print(f"\nDone. Solutions in: {out_dir}")
    if summary_csv.exists():
        print(f"Summary: {summary_csv}")

if __name__ == "__main__":
    main()
