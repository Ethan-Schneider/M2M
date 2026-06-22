"""TA-Hybrid outer driver (paper Algorithm 1).

Per-tick coordination of TA-Hybrid's two-stage offline algorithm by
Liu, Ma, Li, Koenig, "Task and Path Planning for Multi-Agent Pickup and
Delivery," *AAMAS 2019*. The two stages already exist in this codebase:

-   Task assignment (Algorithm 1 Line 2) -- :mod:`ta_assignment` (special
    TSP via LKH-3 over ``G' = A union T``).
-   PlanPathsToDelivery (Algorithm 2) -- :func:`ta_hybrid.plan_paths_to_delivery`.
-   PlanPathsToPickup  (Algorithm 3) -- :func:`ta_hybrid.plan_paths_to_pickup`.

This module is the per-tick *outer* loop that owns:

1.  *Initialisation* (run once at t=0):

    -   Solve the special TSP to get ``plan: agent_id -> [MaterializedTask]``.
    -   Populate ``agent.task_sequence`` with the full plan in M2M tuple
        form ``(task_id, pickup, delivery, deadline)`` so the simulator
        pops one task per completion.
    -   Call ``plan_paths_to_pickup`` for all agents (initial Group 2).
    -   Write ``sub_path`` into ``agent.path_sequence`` and cache the
        ``dummy_path`` for future planning rounds.

2.  *Per-tick coordination* (Algorithm 1 Line 7-onwards):

    -   Detect Group 2 -> Group 1 transitions (M2M status 1 -> 2,
        i.e. agent reached pickup last simulate step).
    -   Detect Group 1 -> Group 2 transitions (M2M status 2 -> 0/1,
        i.e. agent just delivered).
    -   When new agents enter Group 1 -> call ``plan_paths_to_delivery``
        for them with all other agents' commitments as external
        reservations.
    -   When the Group 2 set changes -> call ``plan_paths_to_pickup``
        for the current Group 2 with Group 1 commitments as external
        paths. Apply any anonymity-swap task-sequence swaps from the
        AMAPF result.

Coordinate / time conventions:

-   All cells are absolute ``(row, col)`` tuples.
-   Time is absolute simulator ticks (matching ``t`` from
    ``GT_grid_world.execute``).
-   ``agent.path_sequence`` holds the cells the agent will *physically*
    occupy at ticks ``t_now + 1, t_now + 2, ...``. ``sub_path[0]`` is
    the agent's current cell at ``t_now``; we write ``sub_path[1:]``
    (everything after the start) into ``path_sequence``.
-   Dummy paths are *never* executed by the agent. They're cached as
    reservations only so the next call's ``plan_paths_to_{delivery,
    pickup}`` honours the parking spot the previous round committed to.

Required preconditions:

-   ``--use-precomputed-schedule`` (full task pool known at t=0).
-   ``add_tasks_from_schedule`` returns deterministic, stable task-id
    mapping (we rely on
    ``J[schedule_row_index + 1] == materialized[schedule_row_index]``).
    M2M's stable sort by release_time gives this iff no task is
    *skipped* by ``add_tasks_from_schedule`` at its first eligible
    tick. ``study_small_restricted`` at the densities in the roadmap
    satisfies this; see ``ta_assignment._materialize_schedule`` and
    ``import_schedule.add_tasks_from_schedule`` for the matching
    invariant. Failure of this assumption manifests as a ``ValueError``
    raised from :func:`_sync_task_sequence_cells` on the first
    mismatched task.
-   Dual-cycling (``--aisle-dual-cycle`` / ``--driveway-dual-cycle``)
    must be OFF. The TSP plan is the only source of truth for the
    agent's task ordering; dual-cycle chaining would inject tasks not
    in the plan and break the invariant.
-   For inbound tasks, ``add_tasks_from_schedule`` must use a
    deterministic driveway-cell rule (``deterministic=True``) so that
    every run picks the same cell at the same time. The cell may
    *differ* from the one ``materialize_schedule`` pre-committed at
    ``t=0`` -- as inbound SKUs accumulate, the lowest *empty* driveway
    cell drifts. The driver reconciles this each tick via
    :func:`_sync_task_sequence_cells`, which rewrites each agent's
    pre-committed pickup to the cell ``J`` actually chose and forces
    a Group 2 replan if the head task's pickup moved.

Known limitations:

-   Initial deadline ``L_j`` is a single global value (max execution
    time over agents) for all (agent, task) pairs, rather than the
    paper's tighter ``L_j = L - remaining_exec_time``. Correctness is
    unaffected -- the MCMF just has more flexibility than strictly
    needed (looser pruning). Tightening to the paper's per-task bound
    is a deferred optimisation.
-   An AMAPF sub-path passing through its own pickup cell at
    ``t < r_j`` (en route to an arrival at ``t >= r_j``) would cause a
    spurious M2M ``status 1 -> 2`` transition. The min-cost-flow
    objective makes this rare (any extra detour costs flow) and we do
    not currently guard against it. Flag for revisit if smoke runs
    surface premature pickup transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from numpy.typing import NDArray

from ..agent import Agent, AgentLoader
from ..analysis.statistics import Stats
from ..graph import Graph
from ..logging_config import get_logger
from .ta_assignment import (
    MaterializedTask,
    compute_per_task_L_bounds,
    ta_assignment_plan,
)
from .ta_hybrid import (
    DeliveryPlanRequest,
    PickupPlanRequest,
    plan_paths_to_delivery,
    plan_paths_to_pickup,
)


_LOG = get_logger("alloc.ta_hybrid.driver")

Loc = Tuple[int, int]

# Cap PlanPathsToDelivery's ICBS horizon. Picked large enough to cover
# any reasonable single-task delivery distance on study_small_restricted
# at 30% / 30 bots (longest pickup-to-delivery <= 30 cells, plus ~50 for
# congestion slack); bumped if smoke runs hit the bound.
_DEFAULT_DELIVERY_HORIZON = 200

# Per-call AMAPF planning-horizon ceiling. ``initial_L`` is capped at
# ``t_now + AMAPF_HORIZON_CAP_MULTIPLIER * grid_diameter`` so the
# (cells x time) AMAPF network stays tractable for pure-Python
# construction even when the bottleneck agent's per-task L_j (paper
# Section 3.2) extends ~1500-2000 ticks ahead. Set at 4x diameter:
# enough single-grid-traversals to absorb any reasonable congestion
# while keeping |V| < ~200K on study_small_restricted-shaped maps.
# This is strictly an *upper* bound on L (the per-task L_j and the
# floor below still apply), so it can only cause AMAPF to under-saturate
# meta-vertex windows -- never over-saturate them, never break paper
# semantics. A future optimisation should split long-horizon AMAPF
# into sequenced shorter-horizon solves and remove this cap.
AMAPF_HORIZON_CAP_MULTIPLIER = 4


# ---------------------------------------------------------------------------
# Persistent across-tick state
# ---------------------------------------------------------------------------
@dataclass
class _AgentRoute:
    """One agent's currently-committed route summary.

    ``sub_path`` is the path the agent is *currently* executing toward
    the *next* milestone (pickup if Group 2, delivery if Group 1).
    ``dummy_path`` is the parking-bound path past that milestone --
    cached for reservation purposes only.

    ``sub_path[0]`` is the agent's location at ``start_t``;
    ``sub_path[-1]`` is the milestone cell. ``dummy_path[0]`` is the
    first cell AFTER the milestone (the milestone is NOT repeated).
    ``dummy_path[-1]`` is the parking cell.

    For a Group 1 agent (en route to delivery), ``milestone`` is the
    delivery cell. For a Group 2 agent (en route to pickup),
    ``milestone`` is the pickup cell.
    """

    start_t: int
    sub_path: List[Loc]
    dummy_path: List[Loc]


@dataclass
class _DriverState:
    """Module-level cache for one AgentLoader's TA-Hybrid run."""

    initialized: bool = False
    plan: Dict[int, List[MaterializedTask]] = field(default_factory=dict)
    initial_L: int = 0
    # Per-(agent_id, schedule_row_index) latest-arrival deadline L_{i,k}
    # from paper Section 3.2. Used as the meta-vertex window upper bound
    # for that agent in AMAPF (overrides ``initial_L`` when present).
    # Empty until ``_initialize`` runs.
    per_task_L: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # Last tick we observed each agent's status (used to detect transitions).
    prev_status: Dict[int, int] = field(default_factory=dict)
    # Last tick's Group 2 membership, snapshotted explicitly (not derived
    # from prev_status alone, because an agent that has finished all its
    # tasks still has status 0 but is no longer "free with work to do").
    prev_group_2: set = field(default_factory=set)
    # Currently-committed route per agent. Routes are replaced (not
    # mutated) on each successful replan.
    route: Dict[int, _AgentRoute] = field(default_factory=dict)
    # Task ids whose Stats accumulators we've already initialised --
    # ``Stats.add_actual_*`` is destructive (resets the accumulator to
    # 0), so we must call it AT MOST ONCE per task id. See
    # ``_stats_init_tasks``.
    stats_initialized_task_ids: set = field(default_factory=set)
    # Group 2 sets where the AMAPF was infeasible. We avoid retrying
    # the *exact same* group composition on every subsequent tick --
    # nothing about reservations changes if no other agent transitions,
    # so the retry is guaranteed to fail again. The cache is cleared
    # whenever Group 2 / Group 1 membership changes.
    last_pickup_infeasible_group: Optional[frozenset] = None


# Per-AgentLoader state. We key on ``id(Rs)`` rather than wrapping
# AgentLoader so this stays a strict additive change (no churn in the
# core Agent / AgentLoader classes that the rest of the codebase
# depends on).
_DRIVERS: Dict[int, _DriverState] = {}


def reset_driver_state() -> None:
    """Test helper: clear all TA-Hybrid driver state. Idempotent."""
    _DRIVERS.clear()


def _get_driver(Rs: AgentLoader) -> _DriverState:
    key = id(Rs)
    if key not in _DRIVERS:
        _DRIVERS[key] = _DriverState()
    return _DRIVERS[key]


# ---------------------------------------------------------------------------
# Helpers: M2M task-tuple <-> MaterializedTask
# ---------------------------------------------------------------------------
def _materialized_to_tuple(task: MaterializedTask) -> Tuple[int, Loc, Loc, int]:
    """Convert :class:`MaterializedTask` to M2M's
    ``(task_id, pickup, delivery, deadline)`` shape.

    ``task_id == schedule_row_index + 1`` matches
    :func:`import_schedule.add_tasks_from_schedule`'s assignment order
    when no row is skipped (see module docstring's "Required
    preconditions").
    """
    return (
        task.schedule_row_index + 1,
        task.pickup,
        task.delivery,
        task.deadline,
    )


def _route_to_external_entry(
    route: _AgentRoute,
) -> Tuple[Loc, List[Loc], int, bool]:
    """Pack one agent's committed route into the 4-tuple format that
    ``plan_paths_to_pickup``'s ``external_paths`` expects.

    The full path the other agents must avoid is the concatenation of
    ``sub_path[1:]`` (the moves after the start cell) and ``dummy_path``.
    Marking ``is_permanent_terminal=True`` extends the parking cell's
    occupancy to the planning horizon, which matches the paper's
    "agent rests at parking after dummy path" semantic.
    """
    start_loc = route.sub_path[0]
    tail = list(route.sub_path[1:]) + list(route.dummy_path)
    return (start_loc, tail, route.start_t, True)


def _route_to_delivery_reservation(
    route: _AgentRoute,
    rt: "ReservationTable",  # noqa: F821  -- forward import below
) -> None:
    """Add one Group-1 agent's commitments to a ``ReservationTable``.

    Marks the parking cell as a permanent terminal so it's reserved
    indefinitely against future plans -- same semantic as the
    ``external_paths`` 4-tuple above.
    """
    start_loc = route.sub_path[0]
    tail = list(route.sub_path[1:]) + list(route.dummy_path)
    rt.reserve_path(
        agent_id=0,  # owner id only matters when callers pass ignore_agent
        start_loc=start_loc,
        path=tail,
        start_t=route.start_t,
        is_permanent_terminal=True,
    )


# ---------------------------------------------------------------------------
# Initialisation (t == 0)
# ---------------------------------------------------------------------------
def _initialize(
    driver: _DriverState,
    G: Graph,
    Rs: AgentLoader,
    schedule: NDArray,
) -> None:
    """Run the special TSP and seed driver state.

    Mutates ``Rs.agents`` by setting each agent's ``task_sequence`` to
    the full TSP-ordered plan (M2M tuple form). Does *not* set
    ``path_sequence`` -- that's the next step's responsibility (initial
    PlanPathsToPickup).
    """
    plan, exec_times = ta_assignment_plan(G, Rs, schedule)
    driver.plan = plan
    driver.initial_L = max(exec_times.values()) if exec_times else 0
    # Per-task L_{i,k} (paper Section 3.2) -- tighter, per-(agent, task)
    # window upper bound used by AMAPF instead of a single global L.
    driver.per_task_L = compute_per_task_L_bounds(plan, exec_times, G)
    driver.initialized = True
    for ag in Rs.agents:
        tasks = plan.get(ag.id, [])
        ag.task_sequence = [_materialized_to_tuple(t) for t in tasks]
    _LOG.info(
        "TA-Hybrid driver initialised: %d agents, initial_L=%d, "
        "per_task_L: %d entries (min=%s, max=%s)",
        len(Rs.agents),
        driver.initial_L,
        len(driver.per_task_L),
        min(driver.per_task_L.values(), default=None),
        max(driver.per_task_L.values(), default=None),
    )


# ---------------------------------------------------------------------------
# Group classification
# ---------------------------------------------------------------------------
def _is_group_1(agent: Agent) -> bool:
    """Group 1 (paper): "task agents" -- agent has picked up its current
    task and is carrying. In M2M terms that's ``status == 2``.
    """
    return agent.status == 2


def _is_group_2(agent: Agent) -> bool:
    """Group 2 (paper): "free agents" -- have not yet picked up. In M2M
    terms that's ``status in (0, 1)`` and the agent has at least one
    task remaining.
    """
    return agent.status in (0, 1) and bool(agent.task_sequence)


# ---------------------------------------------------------------------------
# Apply replan results
# ---------------------------------------------------------------------------
def _apply_delivery_results(
    driver: _DriverState,
    Rs: AgentLoader,
    results: Dict[int, "DeliveryPlanResult"],  # noqa: F821
    t_now: int,
) -> None:
    """Write ``plan_paths_to_delivery`` outputs into agents and cache.

    Each agent's ``path_sequence`` is set to ``sub_path[1:]`` (the moves
    *after* the agent's current cell -- the simulator pops one cell per
    tick starting at ``t_now + 1``).
    """
    for agent_id, res in results.items():
        ag = Rs.get_agent(agent_id)
        if ag is None:
            continue
        ag.path_sequence = list(res.sub_path[1:])
        driver.route[agent_id] = _AgentRoute(
            start_t=t_now,
            sub_path=list(res.sub_path),
            dummy_path=list(res.dummy_path),
        )


def _apply_pickup_results(
    driver: _DriverState,
    Rs: AgentLoader,
    results: Dict[int, "PickupPlanResult"],  # noqa: F821
    t_now: int,
) -> None:
    """Write ``plan_paths_to_pickup`` outputs and apply anonymity swaps.

    Anonymity-swap semantics (paper Section 5.3 paragraph 3): when the
    AMAPF assigns walker ``i`` to end at agent-``j``'s pickup cell
    (``assigned_task_agent_id != start_agent_id``), the walker
    "becomes" agent j for task-execution purposes. We mirror this in
    M2M by swapping the *remaining* task sequences (the head task and
    any future tasks) between the two agents. Their parking cells stay
    with the physical agent, since ``home`` is a hardware property.
    """
    # First, perform all task-sequence swaps. We collect swap pairs to
    # avoid double-swapping a triangle of anonymity moves.
    swap_pairs: List[Tuple[int, int]] = []
    seen: set = set()
    for agent_id, res in results.items():
        other_id = res.assigned_task_agent_id
        if other_id == agent_id:
            continue
        key = tuple(sorted((agent_id, other_id)))
        if key in seen:
            continue
        seen.add(key)
        swap_pairs.append((agent_id, other_id))

    for a, b in swap_pairs:
        ag_a = Rs.get_agent(a)
        ag_b = Rs.get_agent(b)
        if ag_a is None or ag_b is None:
            continue
        ag_a.task_sequence, ag_b.task_sequence = (
            list(ag_b.task_sequence),
            list(ag_a.task_sequence),
        )
        _LOG.debug(
            "Anonymity swap at t=%d: agents %d <-> %d", t_now, a, b
        )

    # Then write paths.
    for agent_id, res in results.items():
        ag = Rs.get_agent(agent_id)
        if ag is None:
            continue
        ag.path_sequence = list(res.sub_path[1:])
        # Status 0 -> 1 because the agent is now committed to walking
        # toward a pickup. Idempotent if already 1.
        ag.status = 1
        driver.route[agent_id] = _AgentRoute(
            start_t=t_now,
            sub_path=list(res.sub_path),
            dummy_path=list(res.dummy_path),
        )


# ---------------------------------------------------------------------------
# Build replan inputs
# ---------------------------------------------------------------------------
def _delivery_requests(
    Rs: AgentLoader, agent_ids: Sequence[int]
) -> List[DeliveryPlanRequest]:
    """Build :class:`DeliveryPlanRequest` for each agent that just
    transitioned to Group 1.

    ``current_loc`` is the agent's state right now (which, after the
    M2M simulator's status flip, equals the pickup cell of its current
    task -- the agent literally just arrived there). The task tuple's
    ``[2]`` field is the delivery cell.
    """
    out: List[DeliveryPlanRequest] = []
    for aid in agent_ids:
        ag = Rs.get_agent(aid)
        if ag is None or not ag.task_sequence:
            continue
        _, _, delivery, _ = ag.task_sequence[0]
        out.append(
            DeliveryPlanRequest(
                agent_id=aid,
                current_loc=ag.state,
                delivery_loc=delivery,
                parking_loc=ag.home,
            )
        )
    return out


def _pickup_requests(
    driver: _DriverState,
    Rs: AgentLoader,
    group_2: Sequence[Agent],
    t_now: int,
    deadline_floor: int,
    release_horizon: Optional[int] = None,
) -> List[PickupPlanRequest]:
    """Build :class:`PickupPlanRequest` for each currently-Group-2 agent.

    Release time is sourced from the matching
    :class:`MaterializedTask` (the TSP plan holds release_time;
    M2M's J does not surface it as a structured field). We use the
    initial-task release time of the agent's *current* head task --
    which, after any prior completion-driven ``task_sequence.pop(0)``
    in the simulator, is the right one.

    ``initial_deadline`` is the per-task ``L_{i,k}`` (paper Section 3.2)
    for agent ``i``'s current head task ``k``, computed once at
    initialisation in ``compute_per_task_L_bounds``. Falls back to
    ``driver.initial_L`` (global TSP makespan) if the head task is
    not in the TSP plan.

    The deadline is also floored at ``t_now + deadline_floor`` so an
    agent that has *missed* its TSP-optimal arrival (per-task L_j is
    in the past at the current simulator time -- common at scale, where
    real congestion exceeds the TSP's idealised travel times) still
    gets a non-empty AMAPF meta-vertex window. Without this floor,
    ``window_end = min(deadline, L)`` collapses to a past time, every
    AMAPF call returns infeasible, and the driver's persistent-
    infeasibility cache pins the agent in place forever. The floor
    sacrifices paper-strict makespan guarantees to keep the simulation
    making progress -- for a research benchmark that's the right
    trade-off; the global makespan is then a *measured* outcome rather
    than a *guaranteed* upper bound.
    """
    out: List[PickupPlanRequest] = []
    for ag in group_2:
        if not ag.task_sequence:
            continue
        task_id, pickup, _, _ = ag.task_sequence[0]
        release_time = _release_time_for_task_id(driver, task_id)
        # Skip agents whose head task hasn't been released yet (i.e. is
        # not in J at this tick). The paper's Group 2 by definition
        # plans against the *live* task set; including a task with a
        # future release time bloats the AMAPF horizon (its meta-vertex
        # window is [release_time, deadline], and L must extend through
        # that window for the network to admit a feasible flow). Agents
        # whose head task isn't released wait at home; they re-enter
        # this batch on a later tick once their task fires into J.
        # ``release_horizon`` is preserved as a soft secondary guard for
        # any edge case where ``_release_time_for_task_id`` reports 0
        # (treat-as-released default) but the simulator hasn't actually
        # added the task to J yet.
        if release_time > t_now:
            continue
        if release_horizon is not None and release_time > t_now + release_horizon:
            continue
        # Map M2M task_id back to the schedule_row_index used as the
        # per_task_L key. ``_materialized_to_tuple`` sets
        # ``task_id = schedule_row_index + 1`` so the inverse is
        # ``task_id - 1``.
        schedule_row_index = task_id - 1
        per_task_L = driver.per_task_L.get(
            (ag.id, schedule_row_index), driver.initial_L
        )
        # Also floor the deadline at release_time + a small slack so the
        # AMAPF meta-vertex window [release_time, deadline] is non-empty
        # even when the per-task L_j has been outpaced by reality.
        deadline = max(per_task_L, t_now + deadline_floor, release_time + deadline_floor // 2)
        out.append(
            PickupPlanRequest(
                agent_id=ag.id,
                current_loc=ag.state,
                pickup_loc=pickup,
                parking_loc=ag.home,
                release_time=release_time,
                initial_deadline=deadline,
            )
        )
    return out


def _release_time_for_task_id(
    driver: _DriverState, task_id: int
) -> int:
    """Find ``release_time`` for an M2M task id by reverse-mapping
    through the TSP plan.

    Linear scan -- this only fires on Group 2 replan, which is rare.
    Returns 0 if the task can't be found (defensive default: treat as
    already released).
    """
    target_idx = task_id - 1
    for tasks in driver.plan.values():
        for t in tasks:
            if t.schedule_row_index == target_idx:
                return t.release_time
    return 0


def _other_routes_for_delivery(
    driver: _DriverState, exclude: Sequence[int], t_now: int
) -> "ReservationTable":  # noqa: F821
    """Build a ``ReservationTable`` covering every agent NOT being
    replanned by this :func:`plan_paths_to_delivery` call.

    Sub-paths older than ``t_now`` get truncated (the agent has already
    walked those cells -- they're irrelevant to future planning).
    """
    from ..path_finding_algorithms.mla_star import ReservationTable

    rt = ReservationTable()
    excluded = set(exclude)
    for aid, route in driver.route.items():
        if aid in excluded:
            continue
        truncated = _truncate_route(route, t_now)
        if truncated is None:
            continue
        _route_to_delivery_reservation(truncated, rt)
    return rt


def _other_routes_for_pickup(
    driver: _DriverState, exclude: Sequence[int], t_now: int
) -> Dict[int, Tuple[Loc, List[Loc], int, bool]]:
    """Build the ``external_paths`` dict for ``plan_paths_to_pickup``.

    Mirrors :func:`_other_routes_for_delivery` but in the 4-tuple
    format ``plan_paths_to_pickup`` expects.
    """
    excluded = set(exclude)
    out: Dict[int, Tuple[Loc, List[Loc], int, bool]] = {}
    for aid, route in driver.route.items():
        if aid in excluded:
            continue
        truncated = _truncate_route(route, t_now)
        if truncated is None:
            continue
        out[aid] = _route_to_external_entry(truncated)
    return out


def _truncate_route(
    route: _AgentRoute, t_now: int
) -> Optional[_AgentRoute]:
    """Drop already-walked cells from a cached route.

    If ``t_now`` is past the end of ``sub_path + dummy_path``, the
    agent is parked -- return a degenerate route whose ``sub_path`` is
    just ``[parking_cell]`` so reservations still cover the cell.
    """
    if not route.sub_path:
        return None
    # The agent's location at simulator time ``start_t + k`` is
    # ``(sub_path + dummy_path)[k]`` (where the milestone cell is
    # implicitly between sub_path[-1] and dummy_path[0] -- they share
    # that cell).
    full = list(route.sub_path) + list(route.dummy_path)
    elapsed = t_now - route.start_t
    if elapsed <= 0:
        return route
    if elapsed >= len(full) - 1:
        # Agent has finished both paths -- it's sitting at parking.
        return _AgentRoute(
            start_t=t_now, sub_path=[full[-1]], dummy_path=[]
        )
    # Split point: ``elapsed`` cells walked, ``len(sub_path) - 1`` is
    # the index at which milestone is reached. If elapsed < that, the
    # agent is still in sub_path; otherwise it's in dummy_path. Either
    # way, the "remaining sub_path" is the truncated portion of
    # ``full`` from ``elapsed`` onward.
    remaining_sub = full[elapsed:]
    return _AgentRoute(
        start_t=t_now,
        sub_path=[remaining_sub[0]],
        dummy_path=list(remaining_sub[1:]),
    )


# ---------------------------------------------------------------------------
# Transition detection
# ---------------------------------------------------------------------------
def _new_group_1(
    driver: _DriverState, Rs: AgentLoader
) -> List[int]:
    """Agents whose status went 1 -> 2 since last tick (just picked up).

    Also catches the cold-start case where prev_status was unknown
    *and* current status is 2 (e.g., agent loaded mid-run): such an
    agent gets replanned to be safe.
    """
    out: List[int] = []
    for ag in Rs.agents:
        if not _is_group_1(ag):
            continue
        prev = driver.prev_status.get(ag.id, -1)
        if prev != 2:
            out.append(ag.id)
    return out


def _current_group_2(Rs: AgentLoader) -> set:
    """Set of agent ids currently in Group 2."""
    return {ag.id for ag in Rs.agents if _is_group_2(ag)}


def _group_2_set_changed(
    driver: _DriverState, Rs: AgentLoader
) -> bool:
    """True if the set of currently-Group-2 agents differs from the
    previous tick's Group 2 snapshot. Uses an explicit prior snapshot
    rather than deriving from ``prev_status``, because agents that have
    completed all of their tasks have status 0 but are *not* in Group 2.
    """
    return _current_group_2(Rs) != driver.prev_group_2


def _snapshot_statuses(
    driver: _DriverState, Rs: AgentLoader
) -> None:
    driver.prev_status = {ag.id: ag.status for ag in Rs.agents}
    driver.prev_group_2 = _current_group_2(Rs)


# ---------------------------------------------------------------------------
# Pickup-cell reconciliation
# ---------------------------------------------------------------------------
def _pick_best_pickup(
    G: Graph,
    agent_state: Loc,
    j_pickups: "frozenset[Loc]",  # noqa: F821
    j_deliveries: "frozenset[Loc]",  # noqa: F821
    forbidden: "set[Loc]",  # noqa: F821
) -> Optional[Loc]:
    """Pick the best pickup cell from ``j_pickups`` for an agent at
    ``agent_state`` whose task delivers into ``j_deliveries``.

    Mirrors ``hbh_mla_star._select_pickup_delivery``'s scoring rule:
    ``argmin_s dist(agent, s) + min_g dist(s, g)``. The
    ``min_g dist(s, g)`` term is what makes the choice of pickup
    aware of where the delivery has to land afterwards; it's the
    same scoring c_lns and HBH+MLA* use, so TA-Hybrid stays
    apples-to-apples with the other M2M baselines on the cell-choice
    question (which is M2M's, not the paper's -- the paper assumes
    one cell per task).

    ``forbidden`` are cells already bound to other agents' tasks
    this tick; the heuristic excludes them so two agents never bind
    to the same cell. Returns ``None`` if every candidate pickup is
    in ``forbidden``.
    """
    available = [s for s in j_pickups if s not in forbidden]
    if not available or not j_deliveries:
        return None
    best: Optional[Loc] = None
    best_cost = float("inf")
    for s in available:
        d_as = G.get_distance(agent_state, s)
        d_sg = min(G.get_distance(s, g) for g in j_deliveries)
        cost = d_as + d_sg
        if cost < best_cost:
            best_cost = cost
            best = s
    return best


def _reorder_task_sequence_by_availability(
    Rs: AgentLoader,
    J: Dict[int, Tuple],
) -> bool:
    """Move each Group-2 agent's earliest-available task to the head of
    its ``task_sequence``.

    M2M releases tasks continuously over the simulation (1 every ~7 ticks
    in the light-density regime). The TSP that runs once at t=0 partitions
    the *full* task pool across agents and orders each agent's queue by
    a global cost objective, *not* by release time. As a consequence the
    head of an agent's TSP-ordered queue can be a task whose release
    time is hundreds of ticks in the future, while later positions in
    the same queue contain already-released tasks. Without this reorder
    the agent walks to its (still-empty) materialised pickup cell and
    sits there idle indefinitely -- the simulator can't pick up a
    released SKU, the task isn't in J yet, and no replan would help.

    Policy: split each agent's queue into ``in_J`` (tasks already
    released) and ``not_in_J`` (release time still in the future).
    Promote ``in_J`` tasks to the front, keeping their relative TSP
    order; suffix ``not_in_J`` tasks unchanged. This means:

    -   If at least one of the agent's tasks is currently in J, the
        agent works on that task next.
    -   The TSP-derived ordering is preserved within each group, so we
        still respect the optimiser's intra-group preference.
    -   A task that becomes available later naturally rejoins the head
        queue on a subsequent tick.

    This is a M2M-specific adaptation; the TA-Hybrid paper assumes all
    tasks are present in J before allocation, so this concern doesn't
    arise there. ``_sync_task_sequence_cells`` runs *after* the reorder,
    so cell binding always operates on the post-reorder head.

    Group-1 agents (already carrying) are skipped -- their head task is
    physically committed (SKU in the gripper) and reordering would
    desync the simulator's status state machine.

    Returns ``True`` iff any agent's head task changed.
    """
    head_changed = False
    for ag in Rs.agents:
        if _is_group_1(ag):
            continue
        if not ag.task_sequence:
            continue
        in_J: List[Tuple[int, Loc, Loc, int]] = []
        not_in_J: List[Tuple[int, Loc, Loc, int]] = []
        for tup in ag.task_sequence:
            (in_J if tup[0] in J else not_in_J).append(tup)
        new_seq = in_J + not_in_J
        if new_seq and ag.task_sequence and new_seq[0][0] != ag.task_sequence[0][0]:
            head_changed = True
            _LOG.debug(
                "Agent %d task_sequence reordered: head %d -> %d (in_J=%d, not_in_J=%d)",
                ag.id,
                ag.task_sequence[0][0],
                new_seq[0][0],
                len(in_J),
                len(not_in_J),
            )
        ag.task_sequence = new_seq
    return head_changed


def _sync_task_sequence_cells(
    G: Graph,
    Rs: AgentLoader,
    J: Dict[int, Tuple],
) -> bool:
    """Re-sync each agent's pre-committed pickup cell to one that's
    still valid in ``J``.

    Why this exists -- ``materialize_schedule`` pre-commits a single
    deterministic pickup cell at ``t=0`` so the special TSP can solve
    over the full task pool. Two things can invalidate the pre-commit
    between TSP time and release time:

    -   Inbound tasks: ``add_tasks_from_schedule`` picks the lowest
        *currently empty* driveway cell at release time, which drifts
        as earlier inbound SKUs accumulate.
    -   Outbound tasks: another agent's earlier outbound pickup may
        have removed the SKU at the pre-committed warehouse cell, so
        the SKU now lives at a different cell.

    What this function does NOT touch -- delivery cells. Once the TSP
    plan is set, delivery cells are committed; changing them
    mid-flight (especially for an agent that has already picked up
    and is en route to deliver) leaves a path/sub_path stale and
    desyncs the simulator's pickup/delivery state machine. Delivery
    cells stay as ``materialize_schedule`` set them.

    Cell-binding policy -- when the pre-committed pickup is no longer
    in ``J[task_id][0]``, pick the new pickup using the same scoring
    rule that ``c_lns`` and ``hbh_mla_star`` use:
    ``argmin_s dist(agent, s) + min_g dist(s, g)``. Forbidden cells
    are those already bound to other agents' head tasks this tick,
    so two agents never re-bind to the same pickup cell.

    Group-1 agents (already carrying) are skipped entirely -- they're
    committed to a specific (sku, pickup, delivery) tuple, the SKU is
    physically in the agent, and the simulator will validate it
    against ``task_sequence[0]``. Re-binding their head pickup would
    be meaningless (it's already happened).

    Returns ``True`` iff any agent's *head* task pickup cell changed,
    which is the trigger for forcing a Group 2 replan in
    ``ta_hybrid_call``. Non-head pickup changes apply lazily when
    that task becomes the head (``task_sequence`` is updated in
    place, but the path/sub_path follows naturally on the next
    replan).
    """
    head_changed = False

    # Seed the committed set with every other agent's head pickup so
    # this re-binding never produces a collision with another agent's
    # in-flight target. Non-head pickups are NOT seeded -- they may
    # legitimately move around as the schedule evolves.
    committed_head_pickups: set = set()
    for ag in Rs.agents:
        if ag.task_sequence:
            committed_head_pickups.add(ag.task_sequence[0][1])

    for ag in Rs.agents:
        # Group-1 agents are committed (already carrying); never
        # re-bind their cells.
        if _is_group_1(ag):
            continue
        new_seq: List[Tuple[int, Loc, Loc, int]] = []
        for idx, tup in enumerate(ag.task_sequence):
            task_id, pickup, delivery, deadline = tup
            if task_id not in J:
                new_seq.append(tup)
                continue
            j_entry = J[task_id]
            j_pickups: "frozenset[Loc]" = j_entry[0]  # noqa: F821
            j_deliveries: "frozenset[Loc]" = j_entry[1]  # noqa: F821
            if not j_pickups:
                new_seq.append(tup)
                continue
            if pickup in j_pickups:
                # Pre-commit is still valid; keep it (no churn).
                new_seq.append(tup)
                continue
            # Otherwise re-bind. For the head task we exclude other
            # agents' head pickups so two agents never converge on
            # the same cell. For non-head tasks we don't bother with
            # exclusion -- when they become head later they'll be
            # re-synced again with the up-to-date forbidden set.
            forbidden = (
                committed_head_pickups - {pickup} if idx == 0 else set()
            )
            new_pickup = _pick_best_pickup(
                G, ag.state, j_pickups, j_deliveries, forbidden,
            )
            if new_pickup is None:
                # All candidates are forbidden by other heads; fall
                # back to ``min`` so the agent at least has a target.
                # The persistent-infeasibility cache in the driver
                # will catch any AMAPF deadlock that follows.
                new_pickup = min(j_pickups)
            new_seq.append((task_id, new_pickup, delivery, deadline))
            if idx == 0 and new_pickup != pickup:
                head_changed = True
                committed_head_pickups.discard(pickup)
                committed_head_pickups.add(new_pickup)
                _LOG.debug(
                    "Agent %d head task %d pickup synced %s -> %s",
                    ag.id, task_id, pickup, new_pickup,
                )
        ag.task_sequence = new_seq
    return head_changed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ta_hybrid_call(
    S: Stats,
    G: Graph,
    Rs: AgentLoader,
    J: Dict[int, Tuple],
    t: int,
    schedule: Optional[NDArray] = None,
) -> Tuple[AgentLoader, list, float]:
    """One iteration of TA-Hybrid's outer driver (paper Algorithm 1).

    Matches the 3-tuple signature of ``c_lns_call`` / ``hbh_mla_star_call``
    so ``task_allocation.TaskAllocation`` can dispatch uniformly. The
    second and third returns are unused for TA-Hybrid.

    Args:
        S: M2M ``Stats`` (unused -- TA-Hybrid stats are derived from
            the simulator's task-completion bookkeeping).
        G: M2M ``Graph``.
        Rs: ``AgentLoader``. Mutated in place: ``task_sequence`` /
            ``path_sequence`` / ``status`` updated for replanned
            agents.
        J: Current live task dict from the simulator. Used only to
            sanity-check that the task ids the driver intends to use
            exist (we read from the TSP plan rather than J).
        t: Current simulator tick.
        schedule: Full task schedule (released + unreleased), required
            on the first call. Subsequent calls ignore this argument
            because the TSP plan is already cached.

    Returns:
        ``(Rs, [], 0.0)`` on success. Replan failures are *logged* and
        swallowed -- the agents involved retain their previous
        ``path_sequence`` and the simulator will keep advancing. This
        matches HBH+MLA*'s degradation strategy.
    """
    driver = _get_driver(Rs)

    if not driver.initialized:
        if schedule is None:
            raise ValueError(
                "ta_hybrid requires a precomputed schedule "
                "(pass --use-precomputed-schedule)."
            )
        _initialize(driver, G, Rs, schedule)
        _stats_init_tasks(S, J, driver)
        # Reorder so the head of each agent's queue is one of the tasks
        # already in J (if any) before the initial PlanPathsToPickup --
        # otherwise an agent whose TSP-head has a future release time
        # would sit idle at t=0. Then sync any tasks already in J at t=0
        # before planning so paths target actual cells.
        _reorder_task_sequence_by_availability(Rs, J)
        _sync_task_sequence_cells(G, Rs, J)
        # All agents start in Group 2.
        _replan_group_2(driver, G, Rs, t)
        _snapshot_statuses(driver, Rs)
        return Rs, [], 0.0

    # Subsequent ticks.
    _stats_init_tasks(S, J, driver)
    if t % 20 == 0:
        # Diagnostic histogram printed at WARNING-stream visibility so we
        # can tell whether agents are stuck in status=0 (no task), 1
        # (heading to pickup), 2 (heading to delivery), or 3 (returning).
        hist: Dict[int, int] = {}
        head_tasks_in_J: int = 0
        head_tasks_unreleased: int = 0
        at_pickup_no_carry: int = 0
        for ag in Rs.agents:
            hist[ag.status] = hist.get(ag.status, 0) + 1
            if ag.task_sequence:
                tid = ag.task_sequence[0][0]
                if tid in J:
                    head_tasks_in_J += 1
                else:
                    head_tasks_unreleased += 1
                if (
                    ag.status == 1
                    and ag.state == ag.task_sequence[0][1]
                    and ag.get_sku_id_carrying() is None
                ):
                    at_pickup_no_carry += 1
        path_lens = [len(getattr(ag, 'path_sequence', []) or []) for ag in Rs.agents]
        sorted_lens = sorted(path_lens)
        ts_lens = sorted(len(ag.task_sequence) for ag in Rs.agents)
        empty_ts = sum(1 for n in ts_lens if n == 0)
        print(
            f"[TAHYBRID] t={t} status_hist={dict(sorted(hist.items()))} "
            f"|J|={len(J)} head_in_J={head_tasks_in_J} "
            f"head_unreleased={head_tasks_unreleased} "
            f"at_pickup_no_carry={at_pickup_no_carry} "
            f"empty_ts={empty_ts} "
            f"ts_lens(min/med/max)={(ts_lens[0], ts_lens[len(ts_lens) // 2], ts_lens[-1])} "
            f"path_lens(min/med/max)={(sorted_lens[0], sorted_lens[len(sorted_lens) // 2], sorted_lens[-1])}",
            flush=True,
        )
    head_reordered = _reorder_task_sequence_by_availability(Rs, J)
    pickup_synced = _sync_task_sequence_cells(G, Rs, J)
    new_g1_ids = _new_group_1(driver, Rs)
    g2_changed = (
        _group_2_set_changed(driver, Rs) or pickup_synced or head_reordered
    )

    # If group composition, head ordering, or pickup cells changed, the
    # previous AMAPF infeasibility result no longer applies (different
    # external reservations / different start cells).
    if new_g1_ids or g2_changed:
        driver.last_pickup_infeasible_group = None

    if new_g1_ids:
        _replan_new_group_1(driver, G, Rs, new_g1_ids, t)

    if g2_changed or new_g1_ids:
        # Group 2 also needs a re-plan whenever Group 1 grows, because
        # the freshly-frozen Group 1 reservations are tighter than what
        # the previous PlanPathsToPickup assumed.
        _replan_group_2(driver, G, Rs, t)

    _snapshot_statuses(driver, Rs)
    return Rs, [], 0.0


# ---------------------------------------------------------------------------
# Replan dispatch
# ---------------------------------------------------------------------------
def _replan_group_2(
    driver: _DriverState,
    G: Graph,
    Rs: AgentLoader,
    t: int,
) -> None:
    """Call ``plan_paths_to_pickup`` for the current Group 2."""
    group_2 = [ag for ag in Rs.agents if _is_group_2(ag)]
    if not group_2:
        return
    diameter = (G.height + G.width) if hasattr(G, "height") and hasattr(G, "width") else 50
    cap_offset = AMAPF_HORIZON_CAP_MULTIPLIER * diameter
    requests = _pickup_requests(
        driver,
        Rs,
        group_2,
        t,
        deadline_floor=2 * diameter,
        release_horizon=cap_offset,
    )
    if not requests:
        return
    g2_ids = [r.agent_id for r in requests]
    # Short-circuit: if the *same* Group 2 set was already proven
    # infeasible (e.g. another agent's Group 1 reservation blocks the
    # only path to this agent's pickup) and nothing has changed about
    # who's in which group, retrying the same MCMF will fail again
    # and just burn time. The cache is cleared in ta_hybrid_call
    # whenever Group 1 grows or any pickup cell is re-synced.
    g2_key = frozenset(g2_ids)
    if driver.last_pickup_infeasible_group == g2_key:
        return
    external = _other_routes_for_pickup(driver, exclude=g2_ids, t_now=t)
    # ``initial_L`` is an *absolute* simulator time bound for the AMAPF
    # network. We bound it on both sides relative to ``t``:
    #
    # -  Floor at ``t + 2*diameter``: guarantees ``L >= t`` even when
    #    every per-task L_{i,k} is in the past (e.g. late-tick replans
    #    where the bottleneck agent has already missed its TSP-optimal
    #    arrival), and gives AMAPF at least one full grid traversal of
    #    planning horizon to work with.
    # -  Ceiling at ``t + amapf_horizon_cap``: caps the AMAPF
    #    network's time dimension so the |V| = |open_cells| * L
    #    construction stays tractable. Without this cap, with many
    #    queued tasks per agent the per-task L_j (paper Section 3.2)
    #    can be 1500-2000 ticks even mid-simulation, producing
    #    networks of millions of nodes that pure-Python network
    #    construction cannot keep up with regardless of how fast the
    #    MCMF solver underneath is. The cap is strictly tighter than
    #    L_{i,k} (so it can only ever produce *more* meta-vertex
    #    pruning); a future optimisation should split a long-horizon
    #    AMAPF into multiple shorter-horizon solves.
    #
    # ``AMAPF_HORIZON_CAP_MULTIPLIER`` is sized at 4x diameter, which
    # empirically lets all 30 G2 agents reach any pickup cell on
    # study_small_restricted (~310 ticks for a 27x50 map) while
    # keeping the AMAPF network at ~150K nodes -- i.e. ~10x smaller
    # than the uncapped per-task L_j would have produced.
    floor = 2 * diameter
    cap = AMAPF_HORIZON_CAP_MULTIPLIER * diameter
    max_per_task_L = max((r.initial_deadline for r in requests), default=driver.initial_L)
    capped_per_task_L = min(max_per_task_L, t + cap)
    initial_L = max(t + floor, capped_per_task_L)
    safety = max(1, (initial_L - t) // 10)
    max_L = initial_L + safety
    try:
        results = plan_paths_to_pickup(
            G=G,
            requests=requests,
            external_paths=external,
            t_now=t,
            initial_L=initial_L,
            max_L=max_L,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "plan_paths_to_pickup raised at t=%d: %s", t, exc
        )
        driver.last_pickup_infeasible_group = g2_key
        return
    if results is None:
        _LOG.warning(
            "plan_paths_to_pickup infeasible at t=%d for %d agents",
            t,
            len(requests),
        )
        driver.last_pickup_infeasible_group = g2_key
        return
    driver.last_pickup_infeasible_group = None
    _apply_pickup_results(driver, Rs, results, t_now=t)


def _replan_new_group_1(
    driver: _DriverState,
    G: Graph,
    Rs: AgentLoader,
    new_g1_ids: Sequence[int],
    t: int,
) -> None:
    """Call ``plan_paths_to_delivery`` for the just-transitioned agents."""
    requests = _delivery_requests(Rs, new_g1_ids)
    if not requests:
        return
    g1_ids = [r.agent_id for r in requests]
    other_rt = _other_routes_for_delivery(driver, exclude=g1_ids, t_now=t)
    try:
        results = plan_paths_to_delivery(
            G=G,
            requests=requests,
            other_agent_reservations=other_rt,
            t_now=t,
            horizon=_DEFAULT_DELIVERY_HORIZON,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "plan_paths_to_delivery raised at t=%d: %s", t, exc
        )
        return
    if results is None:
        _LOG.warning(
            "plan_paths_to_delivery infeasible at t=%d for %d agents",
            t,
            len(requests),
        )
        return
    _apply_delivery_results(driver, Rs, results, t_now=t)


# ---------------------------------------------------------------------------
# Stats bookkeeping (mirror ``hbh_mla_star_call``)
# ---------------------------------------------------------------------------
def _stats_init_tasks(
    S: Stats, J: Dict[int, Tuple], driver: _DriverState
) -> None:
    """Ensure every TA-Hybrid-assigned task has its Stats accumulators
    initialised before ``simulate.py`` first touches them.

    ``simulate.py`` increments per-task accumulators
    (``actual_pickup_distance``, ``actual_duration``, ...) on the head
    of each agent's ``task_sequence`` every tick. With TA-Hybrid the
    full plan is materialised at t=0 -- an agent's first head task
    may be a schedule row whose release time is well after t=0 and
    which is therefore not yet in ``J``. The other allocators
    initialise Stats inside their own loops; we do it from the union
    of (a) the TSP plan (all task ids the driver will ever assign)
    and (b) the live ``J`` (covers any task id introduced by something
    other than the TSP, defensively).

    Tracked-by-id rather than idempotent because ``Stats.add_actual_*``
    is destructive (overwrites the accumulator with 0). Re-initialising
    a task whose simulator updates have already started would wipe
    those measurements.
    """
    task_ids: List[int] = []
    # Tasks from the TSP plan: these are the canonical ones the driver
    # will ever assign, and they're known up front.
    for tasks in driver.plan.values():
        for t in tasks:
            tid = t.schedule_row_index + 1  # mirror _materialized_to_tuple
            task_ids.append(tid)
    # Plus anything live in J that didn't come from the plan
    # (shouldn't happen with the precondition checks, but harmless).
    task_ids.extend(J.keys())

    for task_id in task_ids:
        if task_id in driver.stats_initialized_task_ids:
            continue
        driver.stats_initialized_task_ids.add(task_id)
        for fn_name in (
            "add_actual_distance",
            "add_actual_pickup_distance",
            "add_actual_duration",
            "add_actual_pickup_duration",
        ):
            fn = getattr(S, fn_name, None)
            if fn is None:
                continue
            try:
                fn(task_id)
            except Exception as exc:  # noqa: BLE001
                _LOG.debug(
                    "Stats.%s failed for task %d: %s",
                    fn_name,
                    task_id,
                    exc,
                )
