"""Gurobi-native MILP for rearrangement task insertion."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from ..agent import AgentLoader
from ..graph import Graph
from .optimal_insertion import (
    REARRANGEMENT_TASK_ID_BASE,
    TASK_TYPE_SHUFFLE,
    _next_rearrangement_task_id,
    _task_direction,
)

Location = Tuple[int, int]
ReallocationTask = Tuple[Set[Location], float, float, int]
InsertionKey = Tuple[int, int, Location, Location, int, int]
# (task_key, candidate_idx, start, goal, agent_idx, position)

def benefit(G: Graph, s, g, Rs: AgentLoader) -> float:
    dummy_location = Rs.agents[0].home
    return G.get_distance(s, dummy_location) - G.get_distance(g, dummy_location)

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
) -> Tuple[AgentLoader, Dict[int, Tuple]]:
    """Insert Gurobi-selected rearrangement tasks into agent task sequences.

    Args:
        tasks_a: Rearrangement tasks from ``generate_reallocation_tasks``.
        Rs: Current agent loader (not mutated).
        G: Warehouse graph (unused here; kept for API symmetry).
        chosen: Mapping of ``(n, k, s, g, a, i) -> value`` for selected variables.
        J: Live task dictionary; updated in-place for each inserted task.
        next_task_id: Optional starting id; defaults to rearrangement id range.

    Returns:
        Deep copy of ``Rs`` with chosen insertions applied.
    """
    if not chosen:
        return Rs.copy(), J_a

    modified = Rs.copy()
    current_id = next_rearrangement_task_id + 1

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
            task_tuple = (current_id, start, goal, int(deadline))
            modified.agents[agent_idx].task_sequence.insert(position, task_tuple)
            J_a[current_id] = (
                start,
                goal,
                int(deadline),
                G.warehouse.get_sku_at_location(start).sku_id,
                TASK_TYPE_SHUFFLE,
            )
            current_id += 1

    return modified, J_a


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
) -> Tuple[AgentLoader, Dict[int, Tuple]]:
    """Build, solve, and apply the rearrangement insertion MILP with Gurobi."""
    del t0  # used by commented completion-time / deadline blocks
    if not tasks_a:
        return Rs.copy(), J_a

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

    if not z:
        return Rs.copy(), J_a

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

    mdl.optimize()

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return Rs.copy(), J_a

    chosen = {key: var.X for key, var in z.items() if var.X > 0.5}
    Rs_modified, J_a = apply_insertions(
        tasks_a, Rs, G, chosen, J, J_a, next_rearrangement_task_id=next_rearrangement_task_id
    )
    return Rs_modified, J_a
