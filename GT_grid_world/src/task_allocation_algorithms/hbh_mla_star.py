"""HBH (h-value-based heuristic) task allocation driver paired with MLA*.

Faithful re-implementation of Algorithm 2 from
Grenouilleau, van Hoeve, Hooker, "A Multi-Label A* Algorithm for Multi-Agent
Pathfinding," ICAPS 2019, pp. 181-185.

The paper's HBH is an outer ``while not all tasks assigned`` loop that
internally advances simulator time by one tick per iteration. M2M's
``execute()`` already runs ``TaskAllocation`` once per simulator timestep,
so this module implements *one iteration* of HBH (the body of the paper's
while loop) per call. The cadence is therefore equivalent.

HBH + MLA* is a coupled method: MLA* is used during HBH both as a
feasibility test (does a collision-free path actually exist for this
(agent, task) candidate?) and as the path planner that produces the
movement sequence. We therefore *also* fill in ``agent.path_sequence`` here
and ask ``execute()`` to skip the external ECBS/PBS routing call when this
strategy is selected (otherwise ECBS would overwrite our paths with its own,
defeating the point of MLA*).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..agent import AgentLoader, Agent
from ..analysis.statistics import Stats
from ..graph import Graph
from ..path_finding_algorithms.mla_star import (
    ReservationTable,
    mla_star_search,
    mla_star_single_goal,
)


Loc = Tuple[int, int]


def _select_pickup_delivery(
    G: Graph,
    agent_state: Loc,
    start_locs,
    goal_locs,
    forbidden: Set[Loc],
) -> Optional[Tuple[Loc, Loc]]:
    """Pick one ``(pickup, delivery)`` cell pair from the task's candidate
    sets, mirroring the heuristic ``c_lns`` already uses (min-distance
    pickup -> delivery, excluding cells reserved by other already-placed
    pickups/deliveries this tick). Returns ``None`` if no usable pair
    exists.
    """
    candidate_starts = [s for s in start_locs if s not in forbidden]
    candidate_goals = [g for g in goal_locs if g not in forbidden]
    if not candidate_starts or not candidate_goals:
        return None
    best = None
    best_cost = float("inf")
    for s in candidate_starts:
        d_as = G.get_distance(agent_state, s)
        for g in candidate_goals:
            cost = d_as + G.get_distance(s, g)
            if cost < best_cost:
                best_cost = cost
                best = (s, g)
    return best


def _free_endpoints(G: Graph, blocked: Set[Loc]) -> List[Loc]:
    """Endpoints in the paper's sense: cells where an agent may rest
    indefinitely without obstructing tasks. In M2M the natural choices are
    warehouse aisle cells (deep shelf positions) plus any non-obstacle
    location that isn't currently a pickup/delivery candidate. Conservative
    pick: aisle cells. ``blocked`` is the set of cells we're not allowed to
    park on (pickups/deliveries of currently-open tasks plus cells reserved
    elsewhere this tick).
    """
    return [loc for loc in G.get_aisle_locations() if loc not in blocked]


def hbh_mla_star_call(
    S: Stats,
    G: Graph,
    Rs: AgentLoader,
    J: Dict[int, Tuple],
    t: int,
) -> Tuple[AgentLoader, list, float]:
    """One iteration of HBH (paper Algorithm 2) coupled with MLA*.

    Mutates ``Rs.agents`` in place by appending newly-assigned tasks to
    ``task_sequence`` and replacing ``path_sequence`` with the MLA*-found
    path for that agent. Idle agents that currently sit on a pickup or
    delivery cell of an open task are pushed to the closest free aisle cell
    via the single-goal variant of MLA*.

    Returns ``(Rs, [], 0.0)`` to mirror ``c_lns_call`` /  ``py_lns_call``'s
    3-tuple signature, even though the second and third elements are unused
    for HBH.
    """
    free_agents: List[Agent] = [a for a in Rs.agents if a.status == 0]

    # Build the reservation table from every agent's current state and any
    # already-planned path. Two invariants matter for collision-freeness:
    #
    # 1. Every agent that has a path in flight contributes a *permanent*
    #    reservation at its delivery cell from path-end onwards. The agent
    #    will physically sit there once the path runs out (status flips to
    #    0 in simulate.py the moment ``agent.state == delivery``), so any
    #    other agent we plan in this tick that would route through the
    #    delivery cell at or after path-end would collide. This is the
    #    invariant Silver (AIIDE 2005, "Cooperative Pathfinding") spells
    #    out for the reservation table: cells where an agent ends up must
    #    stay reserved forever, otherwise later-planned agents will route
    #    through them. Without ``is_permanent_terminal=True`` here we
    #    observed 145 ticks with vertex collisions in a single 600-tick
    #    HBH run at 30%/30 bots (LNS-PBS at the same conditions: 0).
    #
    # 2. Every stationary agent (status=0 with empty path, or stuck
    #    status>0 with empty path that the recovery pass below will try to
    #    re-plan) reserves its current cell *permanently* from the current
    #    tick. This is what stops a later-planned agent's MLA* from
    #    routing into the parked agent's cell at any future timestep.
    #
    # The planning agent's own MLA* call passes ``ignore_agent=ag.id``, so
    # the search transparently skips its own self-reservations and is
    # never blocked by them. There is therefore no need to "leave a hole"
    # for free agents the way the previous version did.
    reservations = ReservationTable()
    for ag in Rs.agents:
        if ag.path_sequence:
            reservations.reserve_path(
                agent_id=ag.id,
                start_loc=ag.state,
                path=list(ag.path_sequence),
                start_t=t,
                is_permanent_terminal=True,
            )
        else:
            reservations.reserve_path(
                agent_id=ag.id,
                start_loc=ag.state,
                path=[],
                start_t=t,
                is_permanent_terminal=True,
            )

    # Recovery pass: any agent in status 1/2 whose ``path_sequence`` is
    # empty is *stuck* -- the simulator can't advance them and HBH won't
    # see them as ``free`` next tick because their status is still 1/2.
    # Without this pass a single MLA* edge case (e.g. start==pi1, or any
    # future case where MLA* returns a path the simulator can't execute
    # to completion) cascades into permanent deadlock as agents finish
    # in-flight tasks one by one. Plan a fresh path from the agent's
    # current location to whichever goal corresponds to its current
    # status. We do this *before* the new-assignment pair scan so the
    # recovered path's reservations also apply to subsequent planning.
    for ag in Rs.agents:
        if ag.path_sequence:
            continue
        if ag.status not in (1, 2):
            continue
        if not ag.task_sequence:
            continue
        task_id, pickup, delivery, _deadline = ag.task_sequence[0][:4]
        if ag.status == 1:
            recovered = mla_star_search(
                G=G, start=ag.state, pi1=pickup, pi2=delivery,
                reservations=reservations, current_t=t, agent_id=ag.id,
            )
        else:  # status == 2 (carrying, just need to reach delivery)
            recovered = mla_star_single_goal(
                G=G, start=ag.state, goal=delivery,
                reservations=reservations, current_t=t, agent_id=ag.id,
            )
        if recovered:
            ag.path_sequence = list(recovered)
            reservations.reserve_path(
                agent_id=ag.id, start_loc=ag.state, path=list(recovered),
                start_t=t, is_permanent_terminal=True,
            )

    # Collect unassigned task ids: those that no agent currently has in its
    # task_sequence.
    assigned_ids: Set[int] = set()
    for ag in Rs.agents:
        for task_tuple in ag.task_sequence:
            assigned_ids.add(task_tuple[0])
    unassigned_ids = [tid for tid in J.keys() if tid not in assigned_ids]

    # Build (agent, task) candidate list, sorted by ascending h-value
    # (Manhattan distance from agent state to *some* candidate pickup cell).
    # We use the min over candidate starts so that a task with multiple
    # pickup options is judged by its best one, matching how ``c_lns`` later
    # commits to the single best pickup cell.
    pairs: List[Tuple[float, int, int]] = []  # (h_val, agent_id, task_id)
    for ag in free_agents:
        for tid in unassigned_ids:
            start_locs = J[tid][0]
            if not start_locs:
                continue
            h = min(G.get_distance(ag.state, s) for s in start_locs)
            pairs.append((h, ag.id, tid))
    pairs.sort()

    # Greedy scan over sorted pairs. Track which (start, goal) cells are
    # already committed this tick so two new assignments don't collide on
    # the same shelf or driveway slot.
    committed_cells: Set[Loc] = set()
    assigned_this_tick: Set[int] = set()
    busy_agent_ids: Set[int] = set()
    for ag in Rs.agents:
        if ag.task_sequence:
            for task_tuple in ag.task_sequence:
                committed_cells.add(task_tuple[1])
                committed_cells.add(task_tuple[2])

    for h_val, agent_id, task_id in pairs:
        if agent_id in busy_agent_ids:
            continue
        if task_id in assigned_this_tick:
            continue
        ag = Rs.get_agent(agent_id)
        start_locs, goal_locs, deadline, _sku, _type = J[task_id]
        chosen = _select_pickup_delivery(
            G, ag.state, start_locs, goal_locs, committed_cells
        )
        if chosen is None:
            continue
        pickup, delivery = chosen
        path = mla_star_search(
            G=G,
            start=ag.state,
            pi1=pickup,
            pi2=delivery,
            reservations=reservations,
            current_t=t,
            agent_id=ag.id,
        )
        if path is None:
            continue
        # Commit assignment and reserve the new path against subsequent
        # MLA* calls in this tick.
        ag.task_sequence.append((task_id, pickup, delivery, deadline if deadline is not None else 9999))
        ag.status = 1  # to_pickup
        ag.path_sequence = list(path)
        committed_cells.add(pickup)
        committed_cells.add(delivery)
        assigned_this_tick.add(task_id)
        busy_agent_ids.add(agent_id)
        # Replace the agent's stale single-cell reservation with the full
        # planned trajectory.
        # ``ReservationTable`` doesn't support clean removal, so we layer
        # the new (longer) reservation on top -- vertex/edge entries it
        # creates simply overwrite the prior single-cell entries by key.
        reservations.reserve_path(
            agent_id=ag.id,
            start_loc=ag.state,
            path=list(path),
            start_t=t,
            is_permanent_terminal=True,
        )
        # Mirror c_lns_call's Stats bookkeeping: simulate.py later
        # increments these per-task accumulators (actual_pickup_distance,
        # actual_duration, ...) and assumes the keys already exist. Without
        # these calls the first ``simulate()`` tick after assignment dies
        # with a ``KeyError``.
        try:
            S.append_early_task_ids(task_id)
            S.add_actual_distance(task_id)
            S.add_actual_pickup_distance(task_id)
            S.add_actual_duration(task_id)
            S.add_actual_pickup_duration(task_id)
        except AttributeError:
            # ``Stats`` may not have one of the methods available in some
            # test harnesses; the unit tests don't run simulate(), so it's
            # safe to swallow this in that path. The full simulator runs do
            # have the methods.
            pass

    # Endpoint-clearing pass: for each *still-unassigned* free agent that
    # currently sits on the pickup or delivery cell of some open task, push
    # it to the closest free aisle cell. Without this step the paper's
    # Algorithm 2 would deadlock when an idle agent happens to be parked on
    # a shelf another agent will need.
    pickup_delivery_cells: Set[Loc] = set()
    for tid in unassigned_ids:
        if tid in assigned_this_tick:
            continue
        for s in J[tid][0]:
            pickup_delivery_cells.add(s)
        for g in J[tid][1]:
            pickup_delivery_cells.add(g)

    for ag in free_agents:
        if ag.id in busy_agent_ids:
            continue
        if ag.state not in pickup_delivery_cells:
            # Not blocking anything -- leave path_sequence empty so the
            # simulator's "len(path_sequence) == 0 means wait" branch keeps
            # the agent stationary.
            continue
        endpoints = _free_endpoints(G, blocked=pickup_delivery_cells | committed_cells)
        if not endpoints:
            continue
        # Sort endpoints by Manhattan distance and try them in order until
        # MLA* succeeds. Manhattan rather than ``G.get_distance`` here
        # because the latter does a full A* lookup per pair; Manhattan is a
        # cheap admissible proxy for ordering.
        endpoints.sort(key=lambda e: abs(e[0] - ag.state[0]) + abs(e[1] - ag.state[1]))
        for endpoint in endpoints[:8]:  # cap candidates to keep this O(k)
            path = mla_star_single_goal(
                G=G,
                start=ag.state,
                goal=endpoint,
                reservations=reservations,
                current_t=t,
                agent_id=ag.id,
            )
            if path is not None:
                ag.path_sequence = list(path)
                reservations.reserve_path(
                    agent_id=ag.id,
                    start_loc=ag.state,
                    path=list(path),
                    start_t=t,
                    is_permanent_terminal=True,
                )
                break

    return Rs, [], 0.0
