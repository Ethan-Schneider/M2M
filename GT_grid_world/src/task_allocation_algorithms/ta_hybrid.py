"""TA-Hybrid (Liu, Ma, Li, Koenig, AAMAS 2019) implementation for M2M.

This module hosts the TA-Hybrid algorithms from the paper:

-   :func:`plan_paths_to_delivery` -- paper Algorithm 2 / Section 5.2
-   :func:`plan_paths_to_pickup`    -- paper Algorithm 3 / Section 5.3 (TBD,
    Phase C)
-   ``ta_hybrid_step`` outer driver -- paper Algorithm 1 / Section 5.1 (TBD,
    Phase D)

The task-assignment stage (Section 3) lives in :mod:`ta_assignment`; the
ICBS low-level path planning calls :mod:`path_finding_algorithms.icbs_planner`.

Paper fidelity notes
--------------------

The paper's PlanPathsToDelivery modifies the low-level A-star of ICBS so that
its goal test passes only when (a) the agent has reached the delivery
location and (b) a "dummy path" from there to the agent's parking
location can be planned with the current reservations. The vendored
``gloriyo/MAPF-ICBS`` does not expose a goal-test hook, and we avoid
modifying its source for licensing-pending reasons. Our implementation
approximates this by:

1.  Running ICBS to plan sub-paths from each agent's current cell to its
    delivery cell, respecting reservations of all other agents.
2.  *After* ICBS returns, planning each agent's dummy path
    (delivery -> parking) sequentially via :func:`mla_star_single_goal`,
    each one observing the previously planned sub-paths and dummy paths.
3.  Failing the whole call if any dummy path cannot be planned.

In practice this differs from the paper only in pathological cases where
a feasible joint (sub-path, dummy-path) exists at a *longer* sub-path
length than the ICBS-optimal one. The paper's goal-test gate would let
ICBS keep searching at greater depths; our two-phase approach gives up.
We have not seen this happen in realistic M2M instances. If it bites in
the smoke run we can either retry with relaxed reservations or fork the
submodule and patch its A-star goal test.

Coordinate / time conventions
-----------------------------

-   All reservations passed to :func:`plan_paths_to_delivery` are in
    **absolute simulation time**. The function converts to ICBS-relative
    time internally.
-   ICBS sub-paths *include* the start cell:
    ``sub_path[0] == current_loc``, ``sub_path[-1] == delivery_loc``.
-   Dummy paths from :func:`mla_star_single_goal` *exclude* the start
    cell: ``dummy_path[0]`` is the first move AFTER the agent arrives at
    delivery. ``dummy_path[-1] == parking_loc``.
-   The outer driver concatenates the agent's future cells as
    ``sub_path[1:] + dummy_path`` (skipping the current cell, since the
    agent is already there). The first element of that concatenation is
    the cell the agent occupies at ``t_now + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..graph import Graph
from ..logging_config import get_logger
from ..path_finding_algorithms.icbs_planner import plan_icbs_paths
from ..path_finding_algorithms.mla_star import ReservationTable, mla_star_single_goal
from .ta_hybrid_amapf import (
    AmapfAgent,
    AmapfWalker,
    detect_pickup_hold_violations,
    solve_amapf_single_L,
)


_LOG = get_logger("alloc.ta_hybrid")

Loc = Tuple[int, int]


# ---------------------------------------------------------------------------
# Data classes (the surface area Phase D's outer driver will hand us)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeliveryPlanRequest:
    """One agent's input to PlanPathsToDelivery.

    All cells are absolute grid coordinates ``(row, col)``. The agent is
    expected to currently be standing at ``current_loc`` (which equals the
    pickup of its current task -- it just transitioned from "free" to
    "executing").
    """

    agent_id: int
    current_loc: Loc
    delivery_loc: Loc
    parking_loc: Loc


@dataclass(frozen=True)
class DeliveryPlanResult:
    """One agent's output from PlanPathsToDelivery.

    Both paths are ordered lists of grid cells. Conventions:

    -   ``sub_path[0] == current_loc`` (start cell included).
    -   ``sub_path[-1] == delivery_loc``.
    -   ``dummy_path[0]`` is the *first move after* arriving at delivery;
        the delivery cell is **not** repeated at the start of
        ``dummy_path``.
    -   ``dummy_path[-1] == parking_loc``.
    -   The agent's future positions starting at simulation time
        ``t_now + 1`` are ``sub_path[1:] + dummy_path``.
    """

    sub_path: List[Loc]
    dummy_path: List[Loc]


# ---------------------------------------------------------------------------
# Reservation-table helpers (private)
# ---------------------------------------------------------------------------
def _copy_reservation_table(rt: ReservationTable) -> ReservationTable:
    """Return a shallow-but-independent copy of a ``ReservationTable``.

    We need to mutate the table while planning successive dummy paths in
    one ``plan_paths_to_delivery`` call without affecting the caller's
    state. Reservation values are immutable plain dicts of primitives so
    a shallow dict copy is sufficient.
    """
    out = ReservationTable()
    out._vertex = dict(rt._vertex)  # noqa: SLF001 -- intentional private-attr touch
    out._edge = dict(rt._edge)  # noqa: SLF001
    out._permanent = dict(rt._permanent)  # noqa: SLF001
    return out


def _reservations_to_cbs_lists(
    abs_table: ReservationTable,
    t_now: int,
    horizon: int,
) -> Tuple[List[Tuple[Loc, int]], List[Tuple[Loc, Loc, int]]]:
    """Convert ``ReservationTable`` contents to CBS-relative (loc, t) lists.

    ``plan_icbs_paths`` takes flat lists; ``ReservationTable`` keeps a
    structured view that includes permanent terminals. Here we:

    -   shift every reservation's timestamp from absolute to CBS-relative
        (``cbs_t = abs_t - t_now``);
    -   drop reservations outside ``[0, horizon]`` (out of the search
        horizon -- ICBS won't reach them);
    -   expand permanent-terminal entries into per-tick vertex
        reservations from their absolute start time through ``t_now +
        horizon``.

    Args:
        abs_table: Reservation table in absolute simulator time.
        t_now: Current simulator tick. CBS-relative time 0 corresponds to
            this tick.
        horizon: Maximum CBS-relative timestep to emit reservations for.
            Reservations beyond this are dropped (and permanent
            terminals are truncated to this horizon).

    Returns:
        ``(vertex_reservations, edge_reservations)`` in the shape
        :func:`plan_icbs_paths` expects.
    """
    vertex: List[Tuple[Loc, int]] = []
    edge: List[Tuple[Loc, Loc, int]] = []

    for (abs_t, loc), _owner in abs_table._vertex.items():  # noqa: SLF001
        cbs_t = abs_t - t_now
        if 0 <= cbs_t <= horizon:
            vertex.append((loc, cbs_t))

    # ReservationTable.reserve_path keys edges as (t, my_to, my_from)
    # -- the entry forbids the symmetric move (from -> to) at arrival t.
    # plan_icbs_paths expects (from, to, arrival_t).
    for (abs_t, my_to, my_from), _owner in abs_table._edge.items():  # noqa: SLF001
        cbs_t = abs_t - t_now
        if 0 <= cbs_t <= horizon:
            edge.append((my_to, my_from, cbs_t))

    for _agent_id, (perm_loc, perm_start_t) in abs_table._permanent.items():  # noqa: SLF001
        cbs_start = max(0, perm_start_t - t_now)
        for cbs_t in range(cbs_start, horizon + 1):
            vertex.append((perm_loc, cbs_t))

    return vertex, edge


# ---------------------------------------------------------------------------
# Algorithm 2: PlanPathsToDelivery
# ---------------------------------------------------------------------------
def plan_paths_to_delivery(
    G: Graph,
    requests: Sequence[DeliveryPlanRequest],
    other_agent_reservations: ReservationTable,
    t_now: int,
    horizon: int = 200,
    disjoint_splitting: bool = False,
) -> Optional[Dict[int, DeliveryPlanResult]]:
    """Algorithm 2: plan delivery-bound paths plus dummy parking paths.

    Implements paper Section 5.2 with the two-phase approximation of the
    goal-test gate described at module top.

    Args:
        G: M2M ``Graph``. Used for obstacle layout and as the underlying
            cost model for :func:`mla_star_single_goal`.
        requests: One ``DeliveryPlanRequest`` per agent that just
            transitioned from free to executing (i.e., grew the
            executing-set). Agents already in the executing set must
            **not** be in ``requests``; their current paths must instead
            be reflected in ``other_agent_reservations``.
        other_agent_reservations: Reservations from every agent **not**
            in ``requests`` -- their planned future occupancy in
            absolute simulator time. Typically built by the Phase D
            outer driver by aggregating Group 2 (free agent) paths,
            already-planned Group 1 sub-paths, and all currently
            committed dummy paths. The caller's table is **not**
            mutated.
        t_now: Current simulator tick. ICBS-relative time 0 corresponds
            to this tick.
        horizon: Maximum CBS-relative timestep to keep reservations for
            when handing them to ICBS. Default 200 (about 2x the
            diameter of the ``study_*`` maps) is conservative.
        disjoint_splitting: Forwarded to :func:`plan_icbs_paths`. The
            paper TA-Hybrid uses standard splitting; default ``False``
            matches.

    Returns:
        Dict mapping ``agent_id`` to :class:`DeliveryPlanResult` on
        success. ``None`` if ICBS failed to find sub-paths, or if any
        dummy-path planning failed.
    """
    if not requests:
        return {}

    # ----- Step 1: ICBS sub-paths -----
    cbs_vertex, cbs_edge = _reservations_to_cbs_lists(
        other_agent_reservations, t_now, horizon
    )
    starts = [r.current_loc for r in requests]
    goals = [r.delivery_loc for r in requests]

    sub_paths = plan_icbs_paths(
        G,
        starts=starts,
        goals=goals,
        vertex_reservations=cbs_vertex,
        edge_reservations=cbs_edge,
        disjoint_splitting=disjoint_splitting,
        silence_stdout=True,
    )
    if sub_paths is None:
        _LOG.warning(
            "plan_paths_to_delivery: ICBS failed for %d new Group-1 agents at t=%d",
            len(requests),
            t_now,
        )
        return None

    # ----- Step 2: sequential dummy paths -----
    # Augment the caller's reservation table without mutating it: we use the
    # augmented copy to (a) plan each dummy path and (b) keep successive
    # dummy paths from colliding with each other.
    augmented = _copy_reservation_table(other_agent_reservations)
    results: Dict[int, DeliveryPlanResult] = {}

    for i, req in enumerate(requests):
        sub_path = sub_paths[i]
        if sub_path[0] != req.current_loc or sub_path[-1] != req.delivery_loc:
            # plan_icbs_paths should always return paths anchored at the
            # given starts/goals; if not, something is very wrong upstream.
            raise RuntimeError(
                f"ICBS produced unanchored sub_path for agent {req.agent_id}: "
                f"got start={sub_path[0]} end={sub_path[-1]} "
                f"expected start={req.current_loc} end={req.delivery_loc}"
            )

        # Reserve the sub-path in absolute simulator time. The agent is
        # at sub_path[0] at t_now and at sub_path[1+i] at t_now+1+i.
        # is_permanent_terminal=False because the dummy path that follows
        # will set the (later) permanent terminal at the parking cell.
        augmented.reserve_path(
            agent_id=req.agent_id,
            start_loc=sub_path[0],
            path=list(sub_path[1:]),
            start_t=t_now,
            is_permanent_terminal=False,
        )

        delivery_arrival_t = t_now + len(sub_path) - 1
        dummy_path = mla_star_single_goal(
            G,
            start=req.delivery_loc,
            goal=req.parking_loc,
            reservations=augmented,
            current_t=delivery_arrival_t,
            agent_id=req.agent_id,
        )
        if dummy_path is None:
            _LOG.warning(
                "plan_paths_to_delivery: dummy path failed for agent %d "
                "(delivery=%s parking=%s arrival_t=%d)",
                req.agent_id,
                req.delivery_loc,
                req.parking_loc,
                delivery_arrival_t,
            )
            return None

        # Reserve the dummy path. mla_star_single_goal strips its start
        # cell, so ``dummy_path[0]`` is the first move AFTER delivery;
        # we pass start_loc=delivery and path=dummy_path to reserve_path,
        # which puts the agent at delivery at delivery_arrival_t and at
        # dummy_path[i] at delivery_arrival_t + 1 + i.
        # is_permanent_terminal=True so the parking cell is "owned"
        # indefinitely from len(dummy_path) ticks after arrival.
        augmented.reserve_path(
            agent_id=req.agent_id,
            start_loc=req.delivery_loc,
            path=list(dummy_path),
            start_t=delivery_arrival_t,
            is_permanent_terminal=True,
        )

        results[req.agent_id] = DeliveryPlanResult(
            sub_path=list(sub_path),
            dummy_path=list(dummy_path),
        )

    _LOG.info(
        "plan_paths_to_delivery: planned %d new Group-1 agents at t=%d "
        "(avg sub_path=%.1f, avg dummy_path=%.1f)",
        len(requests),
        t_now,
        sum(len(r.sub_path) for r in results.values()) / len(results),
        sum(len(r.dummy_path) for r in results.values()) / len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Algorithm 3: PlanPathsToPickup (paper Section 5.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PickupPlanRequest:
    """One Group-2 (free) agent's input to PlanPathsToPickup.

    The agent's *initial* deadline is ``L_j`` at the caller's chosen
    initial makespan bound ``L``. When PlanPathsToPickup increments
    ``L`` internally (paper Section 5.4 paragraph 1), the deadline
    moves up in lock-step (``L_j = L - remaining_exec_time`` is
    affine in ``L``).
    """

    agent_id: int
    current_loc: Loc        # c_i at t_now (where the free agent currently is)
    pickup_loc: Loc         # s_i of the agent's current (first) task
    parking_loc: Loc        # agent.home
    release_time: int       # r_i of current task
    initial_deadline: int   # L_i at L = initial_L (an absolute timestep)


@dataclass(frozen=True)
class PickupPlanResult:
    """One agent's output from PlanPathsToPickup.

    Conventions match :class:`DeliveryPlanResult`:

    -   ``sub_path[0] == current_loc``,
        ``sub_path[-1] == pickup_loc_of_assigned_task``.
    -   ``dummy_path[0]`` is the first move *after* the pickup;
        ``dummy_path[-1] == parking_loc``.
    -   ``assigned_task_agent_id`` is the *original* agent whose task
        sequence this walker ends up executing. When this differs from
        the agent_id, the outer driver must swap their task sequences
        (paper Section 5.3 paragraph 3).
    """

    sub_path: List[Loc]
    dummy_path: List[Loc]
    assigned_task_agent_id: int


# ----- Partitioning -----
def _partition_group2(
    requests: Sequence[PickupPlanRequest],
) -> List[List[PickupPlanRequest]]:
    """Partition requests into subgroups with pairwise-distinct pickups.

    Paper Section 5.3 paragraph 4: "It assigns integers to all agents
    with the same pickup locations, starting with 1, and then puts
    agents with the same integer into the same subgroup."
    """
    by_pickup: Dict[Loc, List[PickupPlanRequest]] = {}
    for req in requests:
        by_pickup.setdefault(req.pickup_loc, []).append(req)
    if not by_pickup:
        return []
    max_count = max(len(lst) for lst in by_pickup.values())
    subgroups: List[List[PickupPlanRequest]] = [[] for _ in range(max_count)]
    # For determinism, iterate pickups in sorted order.
    for pickup in sorted(by_pickup.keys()):
        for i, req in enumerate(by_pickup[pickup]):
            subgroups[i].append(req)
    return subgroups


# ----- Edge-collision resolution (paper Section 5.4, last paragraph) -----
def _detect_edge_collision(
    a: AmapfWalker, b: AmapfWalker
) -> Optional[int]:
    """Return the earliest step ``k`` (0-based, relative to ``t_now``) at
    which walkers ``a`` and ``b`` swap positions between step k and
    k+1, or ``None`` if no edge collision exists.

    A swap means: ``a[k] != a[k+1]`` (so ``a`` actually moves),
    ``a[k] == b[k+1]``, and ``a[k+1] == b[k]``.
    """
    n = min(len(a.subpath), len(b.subpath))
    for k in range(n - 1):
        ak, ak1 = a.subpath[k], a.subpath[k + 1]
        bk, bk1 = b.subpath[k], b.subpath[k + 1]
        if ak != ak1 and ak == bk1 and ak1 == bk:
            return k
    return None


def _swap_walkers_at(
    a: AmapfWalker, b: AmapfWalker, k: int
) -> Tuple[AmapfWalker, AmapfWalker]:
    """Apply the paper's swap-resolution at step k.

    -   ``a``'s new path: ``a[:k+1] + [a[k] (wait)] + b[k+1:]``
    -   ``b``'s new path: ``b[:k+1] + [b[k] (wait)] + a[k+1:]``
    -   Swap ``assigned_task_agent_id``s.

    Both new paths are 1 cell longer than the originals (the wait step
    is inserted at k+1). Each walker now ends at the *other* walker's
    original endpoint, which is why the task-assignment swap is
    paired with the path swap.
    """
    new_a_path = list(a.subpath[: k + 1]) + [a.subpath[k]] + list(b.subpath[k + 1 :])
    new_b_path = list(b.subpath[: k + 1]) + [b.subpath[k]] + list(a.subpath[k + 1 :])
    new_a = AmapfWalker(
        start_agent_id=a.start_agent_id,
        assigned_task_agent_id=b.assigned_task_agent_id,
        subpath=new_a_path,
    )
    new_b = AmapfWalker(
        start_agent_id=b.start_agent_id,
        assigned_task_agent_id=a.assigned_task_agent_id,
        subpath=new_b_path,
    )
    return new_a, new_b


def _resolve_edge_collisions(
    walkers: Sequence[AmapfWalker], max_iters: int = 100
) -> Optional[List[AmapfWalker]]:
    """Iteratively resolve pairwise edge collisions via swap-and-tail.

    Returns the resolved walker list, or ``None`` if more than
    ``max_iters`` swaps were needed (a degenerate case the caller may
    treat as infeasibility at this ``L``).
    """
    ws = list(walkers)
    for _ in range(max_iters):
        found = False
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                k = _detect_edge_collision(ws[i], ws[j])
                if k is not None:
                    ws[i], ws[j] = _swap_walkers_at(ws[i], ws[j], k)
                    found = True
                    break
            if found:
                break
        if not found:
            return ws
    return None


# ----- Subgroup solve (one subgroup, one L attempt) -----
def _solve_subgroup_at_L(
    G: Graph,
    subgroup: Sequence[PickupPlanRequest],
    external_paths_list: Sequence[Tuple[Loc, List[Loc], int]],
    external_reservations: ReservationTable,
    t_now: int,
    L: int,
    deadline_delta: int,
) -> Optional[Dict[int, PickupPlanResult]]:
    """Try AMAPF + edge-collision resolution + dummy paths at one ``L``.

    Returns the result dict (one entry per agent in subgroup) on
    success, or ``None`` if AMAPF infeasible, edge-collision resolution
    failed to converge, or any dummy-path A* failed.

    ``deadline_delta`` is how much each agent's deadline has been
    bumped up from its ``initial_deadline`` (i.e., ``L - initial_L``).
    """
    amapf_agents = [
        AmapfAgent(
            agent_id=req.agent_id,
            current_loc=req.current_loc,
            pickup_loc=req.pickup_loc,
            release_time=req.release_time,
            deadline=req.initial_deadline + deadline_delta,
        )
        for req in subgroup
    ]

    # Paper Section 5.5 "first try without holds": attempt AMAPF without
    # the pickup-hold edge removal first. The unconstrained network is
    # strictly larger (= less work for max_flow_min_cost) and avoids
    # spurious structural infeasibility caused by one agent's pickup
    # cell being on the only path to another agent's pickup. If the
    # returned walkers have no pickup-cell violations the solution is
    # already collision-free at the level the holds were meant to
    # protect, so we use it directly. Otherwise we retry the same L
    # with holds enabled, which is equivalent to the canonical
    # Section 5.4 path.
    amapf_result = solve_amapf_single_L(
        G, amapf_agents, external_paths_list, t_now, L,
        enforce_pickup_hold=False,
    )
    if amapf_result is None:
        # With-holds is strictly more restrictive (subset of edges) so
        # it would also be infeasible at this L; no retry needed.
        return None

    violations = detect_pickup_hold_violations(
        amapf_result.walkers, amapf_agents, t_now
    )
    if violations:
        _LOG.debug(
            "_solve_subgroup_at_L: %d pickup-hold violation(s) at L=%d; "
            "retrying with holds enabled",
            len(violations),
            L,
        )
        amapf_result = solve_amapf_single_L(
            G, amapf_agents, external_paths_list, t_now, L,
            enforce_pickup_hold=True,
        )
        if amapf_result is None:
            return None

    resolved = _resolve_edge_collisions(amapf_result.walkers)
    if resolved is None:
        _LOG.debug(
            "_solve_subgroup_at_L: edge-collision resolution failed at L=%d "
            "for %d-agent subgroup",
            L,
            len(subgroup),
        )
        return None

    # Plan dummy paths sequentially, in order of arrival time at pickup
    # (earliest first). This gives the most-constrained agent the most
    # planning freedom.
    sub_rt = _copy_reservation_table(external_reservations)

    # Reserve all walker sub-paths first so dummy paths respect them.
    for w in resolved:
        sub_rt.reserve_path(
            agent_id=w.start_agent_id,
            start_loc=w.subpath[0],
            path=list(w.subpath[1:]),
            start_t=t_now,
            is_permanent_terminal=False,
        )

    request_by_id = {req.agent_id: req for req in subgroup}
    ordered = sorted(resolved, key=lambda w: len(w.subpath))

    results: Dict[int, PickupPlanResult] = {}
    for w in ordered:
        req = request_by_id[w.start_agent_id]
        arrival_t = t_now + len(w.subpath) - 1
        dummy = mla_star_single_goal(
            G,
            start=w.subpath[-1],
            goal=req.parking_loc,
            reservations=sub_rt,
            current_t=arrival_t,
            agent_id=w.start_agent_id,
        )
        if dummy is None:
            _LOG.debug(
                "_solve_subgroup_at_L: dummy A* failed for agent %d at L=%d "
                "(pickup=%s parking=%s arrival_t=%d)",
                w.start_agent_id,
                L,
                w.subpath[-1],
                req.parking_loc,
                arrival_t,
            )
            return None
        sub_rt.reserve_path(
            agent_id=w.start_agent_id,
            start_loc=w.subpath[-1],
            path=list(dummy),
            start_t=arrival_t,
            is_permanent_terminal=True,
        )
        results[w.start_agent_id] = PickupPlanResult(
            sub_path=list(w.subpath),
            dummy_path=list(dummy),
            assigned_task_agent_id=w.assigned_task_agent_id,
        )

    return results


# ----- Top-level: plan_paths_to_pickup -----
def plan_paths_to_pickup(
    G: Graph,
    requests: Sequence[PickupPlanRequest],
    external_paths: Dict[int, Tuple[Loc, List[Loc], int, bool]],
    t_now: int,
    initial_L: int,
    max_L: int = 500,
) -> Optional[Dict[int, PickupPlanResult]]:
    """Algorithm 3: plan pickup-bound paths + dummy parking paths.

    Implements paper Section 5.3 with the Section 5.5 "first try
    without holds" optimisation enabled: each per-subgroup AMAPF call
    first solves the unconstrained network and, only if any walker's
    sub-path crosses another agent's terminal pickup cell after that
    agent's arrival, retries the same ``L`` with the canonical
    pickup-hold constraint. This both avoids spurious structural
    infeasibility (e.g. when one pickup is the only path to another)
    and accelerates the dominant solve in the common case. The
    L-increment loop is per-subgroup: each subgroup independently
    finds its own feasible ``L`` (the global makespan is the max over
    subgroups).

    Args:
        G: M2M ``Graph``.
        requests: One :class:`PickupPlanRequest` per Group-2 (free)
            agent. The set is partitioned internally so that within
            each subgroup all pickups are pairwise distinct.
        external_paths: Reservations from agents *not* in any
            ``requests`` subgroup -- typically Group 1 agents and any
            previously-committed Group 2 paths whose owners are not
            being re-planned this call. Each value is
            ``(start_loc_at_start_t, path_after_start, start_t,
            is_permanent_terminal)`` in absolute simulator time.
        t_now: Current simulator tick.
        initial_L: Initial makespan bound; per paper Section 5.4 the
            caller computes this as
            ``max_i estimated_execution_time(task_seq_of_a_i)`` over
            ``requests``.
        max_L: Hard upper bound on the per-subgroup L-increment loop.

    Returns:
        Dict mapping ``agent_id`` to :class:`PickupPlanResult` for
        every agent in ``requests``, or ``None`` if any subgroup is
        infeasible up to ``max_L``.
    """
    if not requests:
        return {}

    subgroups = _partition_group2(requests)

    # Build the running state: a path list (for AMAPF avoidance) and a
    # reservation table (for the dummy-path A*). Both start from the
    # caller's external_paths and accumulate finished subgroups'
    # commitments.
    paths_list: List[Tuple] = []
    external_rt = ReservationTable()
    for ag_id, (sl, p, st, perm) in external_paths.items():
        paths_list.append((sl, list(p), st, perm))
        external_rt.reserve_path(
            agent_id=ag_id,
            start_loc=sl,
            path=list(p),
            start_t=st,
            is_permanent_terminal=perm,
        )

    final: Dict[int, PickupPlanResult] = {}

    for idx, subgroup in enumerate(subgroups):
        L = initial_L
        sub_result: Optional[Dict[int, PickupPlanResult]] = None
        while L <= max_L:
            sub_result = _solve_subgroup_at_L(
                G,
                subgroup=subgroup,
                external_paths_list=paths_list,
                external_reservations=external_rt,
                t_now=t_now,
                L=L,
                deadline_delta=L - initial_L,
            )
            if sub_result is not None:
                break
            L += 1
        if sub_result is None:
            _LOG.warning(
                "plan_paths_to_pickup: subgroup %d (%d agents) infeasible "
                "after L reached %d",
                idx,
                len(subgroup),
                max_L,
            )
            return None

        # Commit this subgroup's paths to running state for the next subgroup.
        # Sub-paths are NOT permanent (the agent moves on along the dummy);
        # dummy paths ARE permanent (the agent parks indefinitely at the
        # parking cell).
        for ag_id, result in sub_result.items():
            arrival_t = t_now + len(result.sub_path) - 1
            paths_list.append(
                (result.sub_path[0], list(result.sub_path[1:]), t_now, False)
            )
            paths_list.append(
                (result.sub_path[-1], list(result.dummy_path), arrival_t, True)
            )
            external_rt.reserve_path(
                agent_id=ag_id,
                start_loc=result.sub_path[0],
                path=list(result.sub_path[1:]),
                start_t=t_now,
                is_permanent_terminal=False,
            )
            external_rt.reserve_path(
                agent_id=ag_id,
                start_loc=result.sub_path[-1],
                path=list(result.dummy_path),
                start_t=arrival_t,
                is_permanent_terminal=True,
            )
            final[ag_id] = result

    _LOG.info(
        "plan_paths_to_pickup: planned %d Group-2 agents across %d subgroups at t=%d",
        len(requests),
        len(subgroups),
        t_now,
    )
    return final
