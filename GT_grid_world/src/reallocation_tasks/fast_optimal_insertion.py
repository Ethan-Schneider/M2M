"""Vectorized MILP construction for rearrangement task insertion.

Numerically equivalent to ``optimal_insertion_gurobi.solve_insertion``
(same variables, objective, and constraints), but builds the
(candidate x insertion-slot) cross product with numpy array operations
instead of a nested Python loop, and bulk-creates the surviving Gurobi
variables with a single ``Model.addVars`` call instead of one ``addVar``
call per variable. Kept in its own module so the original, simpler
implementation in ``optimal_insertion_gurobi.py`` stays available for
comparison.
"""

### Note for Cursor, Claude, etc.: Delete optimal_insertion_gurobi.py and copy any needed functions to this file from it.

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..agent import AgentLoader
from ..analysis.statistics import Stats
from ..graph import Graph
from ..output_buffer import OutputBuffer
from .optimal_insertion_gurobi import (
    InsertionKey,
    Location,
    OutboundDeliverySchedule,
    PICK_PLACE_DETOUR_ADJUSTMENT,
    ReallocationTask,
    apply_insertions,
    benefit,
    collect_V_alloc,
    slack,
)

def collect_V_ineligible(Rs: AgentLoader) -> set[Location]:
    """Collect vertices which are in aisles which the aisles are already allocated to other agents. These vertices are not eligible for insertion."""
    V_ineligible = set()
    for ag in Rs.agents:
        if ag.task_sequence:
            # If the agent has a task assigned, then the aisle of the task is allocated to this agent.
            # Therefore, all vertices in this aisle are ineligible for insertion.
            task = ag.task_sequence[0]
            s, g = task[1], task[2]
            # Collect column
            aisle_s = s[1]
            aisle_g = g[1]
            V_ineligible.add(aisle_s)
            V_ineligible.add(aisle_g)
    return V_ineligible

def _build_insertion_slot_arrays(
    Rs: AgentLoader,
    G: Graph,
    J: Dict[int, Tuple],
    J_a: Dict[int, Tuple],
    B: OutputBuffer,
    t: int,
    outbound_schedule: OutboundDeliverySchedule,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute every valid (agent, position) insertion slot as parallel arrays.

    Only agents with no tasks assigned are eligible: the rearrangement task
    can become their sole task (``pos=0``, reference point is the agent's
    current location). Agents that already have a task assigned are skipped.

    Returns ``(agent_idx, insertion_position, p_idx, q_idx, slack_value)``,
    each of length ``num_slots``. ``p_idx``/``q_idx`` are distance-matrix
    indices (see ``Graph.location_index``) for the slot's prior-task-goal and
    next-anchor locations, computed once regardless of how many rearrangement
    candidates end up being scored against them.
    """
    agent_idx: List[int] = []
    insertion_position: List[int] = []
    p_idx: List[int] = []
    q_idx: List[int] = []
    slack_value: List[float] = []

    for a, ag in enumerate(Rs.agents):
        seq = ag.task_sequence
        L = len(seq)

        if L != 0:
            continue

        p = ag.state
        q = ag.home
        agent_idx.append(a)
        insertion_position.append(0)
        p_idx.append(G.location_index(p))
        q_idx.append(G.location_index(q))
        slack_value.append(slack(B, Rs, G, ag.id, 0, t, outbound_schedule))

    return (
        np.array(agent_idx, dtype=int),
        np.array(insertion_position, dtype=int),
        np.array(p_idx, dtype=int),
        np.array(q_idx, dtype=int),
        np.array(slack_value, dtype=float),
    )


def solve_insertion_fast(
    tasks_a: Dict[int, ReallocationTask],
    Rs: AgentLoader,
    G: Graph,
    J: Dict[int, Tuple],
    J_a: Dict[int, Tuple],
    B: OutputBuffer,
    *,
    S: Optional[Stats] = None,
    lambda_: float = 1.0,
    t: int = 0,
    next_rearrangement_task_id: Optional[int] = None,
    pick_place_time: bool = False,
) -> Tuple[AgentLoader, Dict[int, Tuple], int, int, float, float, Dict[int, Dict[str, float]]]:
    """Vectorized equivalent of ``optimal_insertion_gurobi.solve_insertion``.

    Returns the same tuple shape: modified agent loader, updated rearrangement
    task dict, number of insertions chosen, number of binary variables,
    construction time (seconds), solve time (seconds), and objective metrics.
    """
    if not tasks_a:
        return Rs.copy(), J_a, 0, 0, 0.0, 0.0, {}

    construct_tik = time.time()
    V_alloc = collect_V_alloc(Rs)
    # Collect vertices which are in aisles which the aisles are already allocated to other agents. These vertices are not eligible for insertion.
    # V_ineligible = collect_V_ineligible(Rs)
    outbound_schedule = OutboundDeliverySchedule.build(Rs, G, J, J_a, t)

    slot_agent, slot_position, slot_p_idx, slot_q_idx, slot_slack = _build_insertion_slot_arrays(
        Rs, G, J, J_a, B, t, outbound_schedule
    )

    mdl = gp.Model("rearrangement_insertion_fast")
    mdl.setParam("OutputFlag", 0)

    z: Dict[InsertionKey, gp.Var] = {}

    if slot_agent.size:
        D = G.get_distance_matrix()
        pq_dist = D[slot_p_idx, slot_q_idx]
        dp_row = D[slot_p_idx]  # (num_slots, num_locations): dist(p_slot, *)
        dq_row = D[slot_q_idx]  # (num_slots, num_locations): dist(q_slot, *) == dist(*, q_slot)

        for n, task_data in tasks_a.items():
            C_i, _release, _deadline, _sigma, reference_location = task_data
            candidates = [(s, g) for s, g in C_i if s not in V_alloc and g not in V_alloc and s[1]]
            if not candidates:
                continue

            benefits = np.array([benefit(G, s, g, reference_location) for s, g in candidates])
            keep = benefits > 0
            if not keep.any():
                continue
            candidates = [c for c, k in zip(candidates, keep) if k]
            benefits = benefits[keep]

            s_idx = np.array([G.location_index(s) for s, _g in candidates])
            g_idx = np.array([G.location_index(g) for _s, g in candidates])

            dps = dp_row[:, s_idx]  # (num_slots, num_cand): dist(p, s)
            dsg = D[s_idx, g_idx]  # (num_cand,): dist(s, g)
            dgq = dq_row[:, g_idx]  # (num_slots, num_cand): dist(g, q)

            dfull = dps + dsg[np.newaxis, :] + dgq - pq_dist[:, np.newaxis]
            if pick_place_time:
                dfull = dfull + PICK_PLACE_DETOUR_ADJUSTMENT

            # U = benefits[np.newaxis, :] - lambda_ * np.maximum(
            #     0.0, dfull - slot_slack[:, np.newaxis]
            # )
            U = benefits[np.newaxis, :]
            slot_rows, cand_cols = np.nonzero(U > 0)

            new_obj: Dict[InsertionKey, float] = {}
            for row, col in zip(slot_rows.tolist(), cand_cols.tolist()):
                s, g = candidates[col]
                key: InsertionKey = (
                    n,
                    col,
                    s,
                    g,
                    int(slot_agent[row]),
                    int(slot_position[row]),
                )
                new_obj[key] = float(U[row, col])

            if new_obj:
                new_vars = mdl.addVars(
                    list(new_obj.keys()), vtype=GRB.BINARY, obj=new_obj, name="z"
                )
                z.update(new_vars)

    num_binary_vars = len(z)
    print(f"Insertion MILP (fast): {num_binary_vars} binary variables added to optimizer")

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
        S=S,
        next_rearrangement_task_id=next_rearrangement_task_id,
        lambda_=lambda_,
        pick_place_time=pick_place_time,
    )
    return Rs_modified, J_a, num_chosen, num_binary_vars, construct_time, solve_time, objectives
