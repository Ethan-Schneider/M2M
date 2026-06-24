"""Gurobi-native MILP for rearrangement task insertion."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

TASK_TYPE_INBOUND = 1
TASK_TYPE_SHUFFLE = 2

# Inserted rearrangement tasks use ids in this range to avoid colliding with
# real schedule / CRG task ids.
REARRANGEMENT_TASK_ID_BASE = 1_000_000_000

from ..agent import AgentLoader
from ..graph import Graph

Location = Tuple[int, int]
ReallocationTask = Tuple[Set[Location], float, float, int]
InsertionKey = Tuple[int, int, Location, Location, int, int]
# (task_key, candidate_idx, start, goal, agent_idx, position)

def _task_direction(task_id: int, J: Dict[int, Tuple]) -> str:
    type_ = J[task_id][4]
    return "in" if type_ == TASK_TYPE_INBOUND else "out"

def _next_rearrangement_task_id(
    J: Dict[int, Tuple],
    next_task_id: Optional[int],
) -> int:
    if next_task_id is not None:
        return max(next_task_id, REARRANGEMENT_TASK_ID_BASE)
    rearrangement_ids = [k for k in J if k >= REARRANGEMENT_TASK_ID_BASE]
    if rearrangement_ids:
        return max(rearrangement_ids) + 1
    return REARRANGEMENT_TASK_ID_BASE

def benefit(G: Graph, s, g, Rs: AgentLoader) -> float:
    dummy_location = Rs.agents[0].home
    return G.get_distance(s, dummy_location) - G.get_distance(g, dummy_location)

def compute_insertion_objectives(
    G: Graph,
    Rs: AgentLoader,
    start: Location,
    goal: Location,
    agent_idx: int,
    position: int,
    lambda_: float,
) -> Tuple[float, float, float]:
    """Return benefit, detour cost, and utility for a chosen insertion."""
    prior_goal = Rs.agents[agent_idx].task_sequence[position - 1][2]
    next_anchor = Rs.agents[agent_idx].home
    task_benefit = benefit(G, start, goal, Rs)
    detour_cost = (
        dist(G, prior_goal, start)
        + dist(G, start, next_anchor)
        - dist(G, prior_goal, next_anchor)
    )
    utility = task_benefit - lambda_ * detour_cost
    return task_benefit, detour_cost, utility

def dist(G: Graph, u: Location, v: Location) -> float:
    return float(G.get_distance(u, v))


def collect_V_alloc(Rs: AgentLoader) -> Set[Location]:
    allocated: Set[Location] = set()
    for agent in Rs.agents:
        for _task_id, start, goal, _deadline in agent.task_sequence:
            allocated.add(start)
            allocated.add(goal)
    return allocated


def _reallocation_candidates(
    task_data: ReallocationTask,
) -> List[Tuple[frozenset, frozenset]]:
    C_i, _release, _deadline, _sigma = task_data
    return [(frozenset({s}), frozenset({g})) for s, g in C_i]


def apply_insertions(
    tasks_a: Dict[int, ReallocationTask],
    Rs: AgentLoader,
    G: Graph,
    chosen: Dict[InsertionKey, float],
    J: Dict[int, Tuple],
    J_a: Dict[int, Tuple],
    next_rearrangement_task_id: Optional[int] = None,
    lambda_: float = 1.0,
) -> Tuple[AgentLoader, Dict[int, Tuple], Dict[int, Dict[str, float]]]:
    """Insert Gurobi-selected rearrangement tasks into agent task sequences.

    Args:
        tasks_a: Rearrangement tasks from ``generate_reallocation_tasks``.
        Rs: Current agent loader (not mutated).
        G: Warehouse graph (unused here; kept for API symmetry).
        chosen: Mapping of ``(n, k, s, g, a, i) -> value`` for selected variables.
        J: Live task dictionary; updated in-place for each inserted task.
        next_task_id: Optional starting id; defaults to rearrangement id range.
        lambda_: Detour penalty used when computing insertion utility.

    Returns:
        Deep copy of ``Rs`` with chosen insertions applied, updated ``J_a``,
        and per-task objective metrics keyed by assigned task id.
    """
    if not chosen:
        return Rs.copy(), J_a, {}

    modified = Rs.copy()
    current_id = next_rearrangement_task_id + 1
    objectives: Dict[int, Dict[str, float]] = {}

    by_agent: Dict[int, List[Tuple[int, int, Location, Location]]] = defaultdict(list)
    for (n, _k, s, g, a, i), value in chosen.items():
        if value > 0.5:
            by_agent[a].append((i, n, s, g))

    for agent_idx, insertions in by_agent.items():
        # Insert from highest position first so earlier indices stay valid.
        for position, task_key, start, goal in sorted(
            insertions, key=lambda item: -item[0]
        ):
            _C_i, _release, deadline, sigma = tasks_a[task_key]
            task_benefit, detour_cost, utility = compute_insertion_objectives(
                G, Rs, start, goal, agent_idx, position, lambda_
            )
            task_tuple = (current_id, start, goal, int(deadline))
            modified.agents[agent_idx].task_sequence.insert(position, task_tuple)
            J_a[current_id] = (
                start,
                goal,
                int(deadline),
                G.warehouse.get_sku_at_location(start).sku_id,
                TASK_TYPE_SHUFFLE,
            )
            objectives[current_id] = {
                "benefit": float(task_benefit),
                "utility": float(utility),
                "detour_cost": float(detour_cost),
            }
            current_id += 1

    return modified, J_a, objectives


def solve_insertion(
    tasks_a: Dict[int, ReallocationTask],
    Rs: AgentLoader,
    G: Graph,
    J: Dict[int, Tuple],
    J_a: Dict[int, Tuple],
    *,
    lambda_: float = 1.0,
    t0: float = 0.0,
    next_rearrangement_task_id: Optional[int] = None,
) -> Tuple[AgentLoader, Dict[int, Tuple], int, int, float, float, Dict[int, Dict[str, float]]]:
    """Build, solve, and apply the rearrangement insertion MILP with Gurobi.

    Returns:
        Modified agent loader, updated rearrangement task dict, number of
        insertions chosen by the optimizer, number of binary variables in the
        MILP, construction time (seconds), solve time (seconds), and objective
        metrics for inserted rearrangement tasks.
    """
    del t0  # used by commented completion-time / deadline blocks
    if not tasks_a:
        return Rs.copy(), J_a, 0, 0, 0.0, 0.0, {}

    construct_tik = time.time()
    V_alloc = collect_V_alloc(Rs)

    mdl = gp.Model("rearrangement_insertion")
    mdl.setParam("OutputFlag", 0)

    #    # --- completion times of each agent's current route ---
    #    T = {}
    #    for a, ag in enumerate(Rs.agents):
    #        acc, prev, T[a] = 0.0, ag.loc, {}
    #        for i, tk in enumerate(ag.seq):
    #            acc += dist(prev, tk.start) + dist(tk.start, tk.goal)
    #            T[a][i], prev = acc, tk.goal

    z: Dict[InsertionKey, gp.Var] = {}
    for n, task_data in tasks_a.items():
        candidates = _reallocation_candidates(task_data)
        _C_i, _release, _deadline, _sigma = task_data
        for k, (S, D) in enumerate(candidates):
            for s in S:
                if s in V_alloc:
                    continue
                for g in D:
                    if g in V_alloc:
                        continue
                    if benefit(G, s, g, Rs) <= 0:
                        continue
                    for a, ag in enumerate(Rs.agents):
                        seq = ag.task_sequence
                        L = len(seq)
                        if L == 0:
                            continue
                        for i in range(1, L + 1):
                            prior_task_id = seq[i - 1][0]
                            if prior_task_id in J_a:
                                continue
                            # if _task_direction(prior_task_id, J) != "in":
                            #     continue
                            if prior_task_id in J_a:
                                if J_a[prior_task_id][4] == 0 or J_a[prior_task_id][4] == 2:
                                    continue
                            else:
                                if J[prior_task_id][4] == 0 or J[prior_task_id][4] == 2:
                                    continue
                            p = seq[i - 1][2]
                            q = ag.home
                            ddet = dist(G, p, s) + dist(G, s, q) - dist(G, p, q)
                            # if ddet > 10:
                            #     continue
                            # d = dist(G, p, s) + dist(G, s, g) + dist(G, g, q) - dist(G, p, q)
                            U = benefit(G, s, g, Rs) - lambda_ * ddet
                            if U <= 0:
                                continue
                            #    ti = T[a][i-1] + dist(p,s) + dist(s,g)        # inserted finish
                            #    if not (task.release <= ti <= task.deadline): # C3a
                            #        continue
                            #    if any(T[a][j] + d > ag.seq[j].deadline       # C3b
                            #           for j in range(i, L)):
                            #        continue
                            z[(n, k, s, g, a, i)] = mdl.addVar(
                                vtype=GRB.BINARY,
                                obj=U,
                                name=f"z_{n}_{k}_{s}_{g}_{a}_{i}",
                            )

    num_binary_vars = len(z)
    print(f"Insertion MILP: {num_binary_vars} binary variables added to optimizer")

    mdl.ModelSense = GRB.MAXIMIZE

    g1: Dict[int, List[gp.Var]] = defaultdict(list)
    for (n, _k, _s, _g, _a, _i), var in z.items():
        g1[n].append(var)
    for n, vs in g1.items():
        mdl.addConstr(gp.quicksum(vs) <= 1, f"C1_{n}")

    g5: Dict[Location, List[gp.Var]] = defaultdict(list)
    for (_n, _k, s, g, _a, _i), var in z.items():
        g5[s].append(var)
        g5[g].append(var)
    for vtx, vs in g5.items():
        mdl.addConstr(gp.quicksum(vs) <= 1, f"C5_{vtx}")

    g6: Dict[Tuple[int, int], List[gp.Var]] = defaultdict(list)
    for (_n, _k, _s, _g, a, i), var in z.items():
        g6[(a, i)].append(var)
    for slot, vs in g6.items():
        mdl.addConstr(gp.quicksum(vs) <= 1, f"C6_{slot[0]}_{slot[1]}")

    construct_time = time.time() - construct_tik

    if not z:
        return Rs.copy(), J_a, 0, num_binary_vars, construct_time, 0.0, {}

    solve_tik = time.time()
    mdl.optimize()
    solve_time = time.time() - solve_tik

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return Rs.copy(), J_a, 0, num_binary_vars, construct_time, solve_time, {}

    chosen = {key: var.X for key, var in z.items() if var.X > 0.5}
    num_chosen = len(chosen)
    Rs_modified, J_a, objectives = apply_insertions(
        tasks_a,
        Rs,
        G,
        chosen,
        J,
        J_a,
        next_rearrangement_task_id=next_rearrangement_task_id,
        lambda_=lambda_,
    )
    return Rs_modified, J_a, num_chosen, num_binary_vars, construct_time, solve_time, objectives
