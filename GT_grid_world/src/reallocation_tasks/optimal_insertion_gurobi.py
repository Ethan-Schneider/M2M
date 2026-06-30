"""Gurobi-native MILP for rearrangement task insertion."""

from __future__ import annotations

import time, os
import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

# os.environ["GRB_LICENSE_FILE"] = os.path.join("data/licenses/gurobi.lic")

TASK_TYPE_INBOUND = 1
TASK_TYPE_SHUFFLE = 2

# Pick (5s) + place (5s) detour adjustment when pick/place time is enabled.
PICK_PLACE_DETOUR_ADJUSTMENT = 8.0

# Inserted rearrangement tasks use ids in this range to avoid colliding with
# real schedule / CRG task ids.
REARRANGEMENT_TASK_ID_BASE = 1_000_000_000

from ..agent import AgentLoader
from ..graph import Graph
from ..output_buffer import OutputBuffer
from ..simulate import (
    STATUS_PICKING,
    STATUS_PLACING,
    STATUS_TO_DELIVERY,
    TASK_TYPE_OUTBOUND,
)

Location = Tuple[int, int]
ReallocationTask = Tuple[Set[Location], float, float, int]
InsertionKey = Tuple[int, int, Location, Location, int, int]
# (task_key, candidate_idx, start, goal, agent_idx, insertion_position)
#
# Indexing convention:
# - ``insertion_position`` is 1-based: insert after task index ``insertion_position - 1``.
# - ``task_index`` is 0-based into ``agent.task_sequence``.
# - ``agent_idx`` is the index in ``Rs.agents`` (not necessarily ``agent.id``).

def dist(G: Graph, u: Location, v: Location) -> float:
    return float(G.get_distance(u, v))


def prior_task_index(insertion_position: int) -> int:
    """Map 1-based insertion slot to the 0-based index of the preceding task."""
    if insertion_position < 1:
        raise ValueError(f"insertion_position must be >= 1, got {insertion_position}")
    return insertion_position - 1


def _has_reached_pickup(agent, start: Location) -> bool:
    """True when the agent has arrived at the pickup cell for its current task."""
    if agent.status in (STATUS_TO_DELIVERY, STATUS_PLACING, STATUS_PICKING):
        return True
    return agent.state == start


def _remaining_travel_current_task(
    G: Graph,
    agent,
    start: Location,
    goal: Location,
) -> float:
    """Estimated travel time left for the agent's in-progress task."""
    loc = agent.state
    if agent.status == STATUS_PLACING and loc == goal:
        return 0.0
    if _has_reached_pickup(agent, start):
        return dist(G, loc, goal)
    return dist(G, loc, start) + dist(G, start, goal)


def arrival_time(
    G: Graph,
    Rs: AgentLoader,
    agent_id: int,
    task_index: int,
    t: int,
) -> float:
    """Estimate when the agent finishes the task at ``task_index`` in its sequence.

    ``task_index`` is 0-based. Uses ``dist`` from the agent's current location for
    the active task (index 0), accounting for whether pickup has already occurred.
    Later tasks are chained from the prior task's goal through pickup and delivery.
    """
    agent = Rs.get_agent(agent_id)
    if agent is None:
        raise ValueError(f"Agent {agent_id} not found")
    seq = agent.task_sequence
    if task_index < 0 or task_index >= len(seq):
        raise ValueError(
            f"task_index {task_index} out of range for agent {agent_id} "
            f"with {len(seq)} tasks"
        )

    total_travel = 0.0
    for idx in range(task_index + 1):
        _task_id, start, goal, _deadline = seq[idx]
        if idx == 0:
            total_travel += _remaining_travel_current_task(G, agent, start, goal)
        else:
            prev_goal = seq[idx - 1][2]
            total_travel += dist(G, prev_goal, start) + dist(G, start, goal)

    return float(t) + total_travel


def completion_time_before_insertion(
    G: Graph,
    Rs: AgentLoader,
    agent_id: int,
    insertion_position: int,
    t: int,
) -> float:
    """Estimated finish time of the task immediately before an insertion slot."""
    return arrival_time(G, Rs, agent_id, prior_task_index(insertion_position), t)


def _task_type(task_id: int, J: Dict[int, Tuple], J_a: Dict[int, Tuple]) -> Optional[int]:
    if task_id in J_a:
        return J_a[task_id][4]
    if task_id in J:
        return J[task_id][4]
    return None


@dataclass
class OutboundDeliverySchedule:
    """Estimated outbound driveway delivery times, sorted for fast range queries."""

    entries: List[Tuple[int, float]] = field(default_factory=list)
    _times: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.entries = sorted(self.entries, key=lambda item: item[1])
        self._times = [delivery_time for _agent_id, delivery_time in self.entries]

    @classmethod
    def build(
        cls,
        Rs: AgentLoader,
        G: Graph,
        J: Dict[int, Tuple],
        J_a: Dict[int, Tuple],
        t: int,
    ) -> OutboundDeliverySchedule:
        entries: List[Tuple[int, float]] = []
        for agent in Rs.agents:
            for idx, (task_id, _start, _goal, _deadline) in enumerate(agent.task_sequence):
                task_type = _task_type(task_id, J, J_a)
                if task_type != TASK_TYPE_OUTBOUND:
                    continue
                delivery_time = arrival_time(G, Rs, agent.id, idx, t)
                entries.append((agent.id, delivery_time))
        return cls(entries)

    def count_other_deliveries(
        self,
        t_start: float,
        t_end: float,
        excluded_agent_id: Optional[int] = None,
    ) -> int:
        """Count outbound deliveries in (t_start, t_end] from agents other than excluded."""
        if t_end <= t_start or not self.entries:
            return 0
        left = bisect.bisect_right(self._times, t_start)
        right = bisect.bisect_right(self._times, t_end)
        return sum(
            1
            for agent_id, _delivery_time in self.entries[left:right]
            if agent_id != excluded_agent_id
        )


def predict_output_buffer_level(
    B: OutputBuffer,
    query_arrival_time: float,
    t: int,
    outbound_schedule: OutboundDeliverySchedule,
    excluded_agent_id: Optional[int] = None,
) -> float:
    """Predict shared output-buffer level at ``query_arrival_time``.

    Starts from ``B.level`` at time ``t``, subtracts linear consumption over the
    elapsed interval, adds one unit per outbound delivery from other agents
    estimated to finish in ``(t, query_arrival_time]``, then clips to
    ``[0, B.capacity]``.
    """
    elapsed = max(0.0, float(query_arrival_time) - float(t))
    level = float(B.level) - B.consumption_rate * elapsed
    level += outbound_schedule.count_other_deliveries(
        float(t),
        float(query_arrival_time),
        excluded_agent_id=excluded_agent_id,
    )
    return max(0.0, min(float(B.capacity), level))


def _can_accept_delivery_at(
    B: OutputBuffer,
    query_time: float,
    t: int,
    outbound_schedule: OutboundDeliverySchedule,
    excluded_agent_id: Optional[int] = None,
) -> bool:
    level = predict_output_buffer_level(
        B, query_time, t, outbound_schedule, excluded_agent_id=excluded_agent_id
    )
    return level + 1.0 <= float(B.capacity)


def _accept_horizon(
    B: OutputBuffer,
    query_arrival_time: float,
    t: int,
    outbound_schedule: OutboundDeliverySchedule,
    excluded_agent_id: Optional[int] = None,
) -> float:
    """Conservative upper bound for earliest time the buffer can accept one item."""
    if B.consumption_rate <= 0.0:
        return float("inf")
    max_future_deliveries = outbound_schedule.count_other_deliveries(
        float(t),
        float("inf"),
        excluded_agent_id=excluded_agent_id,
    )
    level_at_arrival = predict_output_buffer_level(
        B,
        query_arrival_time,
        t,
        outbound_schedule,
        excluded_agent_id=excluded_agent_id,
    )
    excess = max(0.0, level_at_arrival - (float(B.capacity) - 1.0))
    drain_time = excess / B.consumption_rate
    refill_time = max_future_deliveries / B.consumption_rate
    return float(query_arrival_time) + drain_time + refill_time + 1.0


def _earliest_accept_time(
    B: OutputBuffer,
    query_arrival_time: float,
    t: int,
    outbound_schedule: OutboundDeliverySchedule,
    excluded_agent_id: Optional[int] = None,
) -> float:
    """Earliest time at or after ``query_arrival_time`` with room for one delivery."""
    if _can_accept_delivery_at(
        B, query_arrival_time, t, outbound_schedule, excluded_agent_id
    ):
        return float(query_arrival_time)

    if B.consumption_rate <= 0.0:
        return float("inf")

    low = float(query_arrival_time)
    high = _accept_horizon(B, query_arrival_time, t, outbound_schedule, excluded_agent_id)
    for _ in range(64):
        mid = 0.5 * (low + high)
        if _can_accept_delivery_at(B, mid, t, outbound_schedule, excluded_agent_id):
            high = mid
        else:
            low = mid
    return high


def accept(
    B: OutputBuffer,
    query_arrival_time: float,
    t: int,
    outbound_schedule: OutboundDeliverySchedule,
    excluded_agent_id: Optional[int] = None,
) -> float:
    """Return when an outbound delivery can enter the shared buffer.

    If the predicted level at ``query_arrival_time`` has room for one item,
    returns ``query_arrival_time``. Otherwise returns the earliest later time
    when consumption (and scheduled deliveries from other agents) frees capacity.
    """
    return _earliest_accept_time(
        B,
        query_arrival_time,
        t,
        outbound_schedule,
        excluded_agent_id=excluded_agent_id,
    )


def slack(
    B: OutputBuffer,
    Rs: AgentLoader,
    G: Graph,
    agent_id: int,
    insertion_position: int,
    t: int,
    outbound_schedule: OutboundDeliverySchedule,
) -> float:
    """Buffer wait slack before an insertion at ``insertion_position`` (1-based)."""
    t_f = completion_time_before_insertion(G, Rs, agent_id, insertion_position, t)
    t_accept = accept(B, t_f, t, outbound_schedule, excluded_agent_id=agent_id)
    if t_accept == float("inf"):
        return float("inf")
    return max(0.0, t_accept - t_f)

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
    insertion_position: int,
    lambda_: float,
    pick_place_time: bool = False,
) -> Tuple[float, float, float]:
    """Return benefit, detour cost, and utility for a chosen insertion."""
    prior_goal = Rs.agents[agent_idx].task_sequence[prior_task_index(insertion_position)][2]
    next_anchor = Rs.agents[agent_idx].home
    task_benefit = benefit(G, start, goal, Rs)
    detour_cost = (
        dist(G, prior_goal, start)
        + dist(G, start, next_anchor)
        - dist(G, prior_goal, next_anchor)
    )
    if pick_place_time:
        detour_cost -= PICK_PLACE_DETOUR_ADJUSTMENT
    utility = task_benefit - lambda_ * detour_cost
    return task_benefit, detour_cost, utility


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
    pick_place_time: bool = False,
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
    for (n, _k, s, g, a, insertion_position), value in chosen.items():
        if value > 0.5:
            by_agent[a].append((insertion_position, n, s, g))

    for agent_idx, insertions in by_agent.items():
        # Insert from highest position first so earlier indices stay valid.
        for insertion_position, task_key, start, goal in sorted(
            insertions, key=lambda item: -item[0]
        ):
            _C_i, _release, deadline, sigma = tasks_a[task_key]
            task_benefit, detour_cost, utility = compute_insertion_objectives(
                G, Rs, start, goal, agent_idx, insertion_position, lambda_, pick_place_time
            )
            task_tuple = (current_id, start, goal, int(deadline))
            modified.agents[agent_idx].task_sequence.insert(insertion_position, task_tuple)
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
    B : OutputBuffer,
    *,
    lambda_: float = 1.0,
    t: int = 0,
    next_rearrangement_task_id: Optional[int] = None,
    pick_place_time: bool = False,
) -> Tuple[AgentLoader, Dict[int, Tuple], int, int, float, float, Dict[int, Dict[str, float]]]:
    """Build, solve, and apply the rearrangement insertion MILP with Gurobi.

    Returns:
        Modified agent loader, updated rearrangement task dict, number of
        insertions chosen by the optimizer, number of binary variables in the
        MILP, construction time (seconds), solve time (seconds), and objective
        metrics for inserted rearrangement tasks.
    """
    if not tasks_a:
        return Rs.copy(), J_a, 0, 0, 0.0, 0.0, {}

    construct_tik = time.time()
    V_alloc = collect_V_alloc(Rs)
    _outbound_schedule = OutboundDeliverySchedule.build(Rs, G, J, J_a, t)

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
                    if benefit(G, s, g, Rs) <= 5:
                        continue
                    for a, ag in enumerate(Rs.agents):
                        seq = ag.task_sequence
                        L = len(seq)
                        if L == 0:
                            continue
                        for insertion_position in range(1, L + 1):
                            prior_idx = prior_task_index(insertion_position)
                            prior_task_id = seq[prior_idx][0]

                            if prior_task_id in J_a:
                                continue
                            else:
                                if J[prior_task_id][4] == 0 or J[prior_task_id][4] == 2:
                                    continue
                            p = seq[prior_idx][2]
                            if insertion_position < L:
                                q = seq[insertion_position][1]  # start of next task
                            else:
                                q = ag.home
                            # ddet = dist(G, p, s) + dist(G, s, q) - dist(G, p, q)
                            dfull = dist(G, p, s) + dist(G, s, g) + dist(G, g, q) - dist(G, p, q)
                            # Add additional pick place time to the full distance cost
                            if pick_place_time:
                                dfull += PICK_PLACE_DETOUR_ADJUSTMENT
                            
                            slack_value = slack(
                                        B,
                                        Rs,
                                        G,
                                        ag.id,
                                        insertion_position,
                                        t,
                                        _outbound_schedule,
                                    )
                            print(f"slack_value: {slack_value}")
                            print(f"dfull: {dfull}")
                            print(f"benefit: {benefit(G, s, g, Rs)}")
                            U = benefit(G, s, g, Rs) - lambda_ * max(0.0, (dfull - slack_value))
                            if U <= 0:
                                continue
                            print(f"U: {U}")
                            z[(n, k, s, g, a, insertion_position)] = mdl.addVar(
                                vtype=GRB.BINARY,
                                obj=U,
                                name=f"z_{n}_{k}_{s}_{g}_{a}_{insertion_position}",
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
        pick_place_time=pick_place_time,
    )
    return Rs_modified, J_a, num_chosen, num_binary_vars, construct_time, solve_time, objectives
