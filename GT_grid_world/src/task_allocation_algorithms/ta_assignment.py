"""TA-Hybrid task-assignment stage (paper Section 3).

Faithful re-implementation of Section 3 of
Liu, Ma, Li, Koenig, "Task and Path Planning for Multi-Agent Pickup and
Delivery," *AAMAS 2019*, pp. 1152-1160.

This module implements **only** the task-assignment stage of TA-Hybrid (and
TA-Prioritized -- both papers share this stage). The path-planning stage
(Section 4 for TA-Prioritized, Section 5 for TA-Hybrid) lives in a separate
module (``ta_hybrid.py``) wired up later in the roadmap.

The TA-assignment stage:

1.  Reads the full offline task schedule (released and unreleased).
2.  Materialises a representative ``(pickup_cell, delivery_cell)`` pair for
    every scheduled task (the schedule by itself does not pin cells -- it
    only names a SKU and a task type).
3.  Builds a directed weighted graph ``G' = (V', E')`` with ``V' = A union T``
    (one vertex per agent, one per task) with the four edge-weight rules
    from paper Section 3.1.
4.  Solves a special TSP on ``G'`` with LKH-3 (Helsgaun 2017) via the
    ``elkai`` Python wrapper, producing a Hamiltonian cycle.
5.  Partitions the cycle at agent vertices into ``M`` ordered task
    sequences, one per agent.
6.  Computes per-sequence execution times per Section 3.2 (used as the
    secondary objective / tie-breaker, and surfaced for diagnostics).

Online-vs-offline bridge (per Ethan's `task_schedule_creation` merge,
2026-06-17): we require the simulator to be running in
``--use-precomputed-schedule`` mode so the full task pool is known at
``t = 0``. Without that, TA-Hybrid would have to re-solve the TSP every
time a new task arrived, which destroys the algorithm's runtime profile
and is not what the paper describes.

Cell materialisation (Step 2) is the only place this module makes
representative choices, because the schedule names a SKU and a task type
but does not commit to a specific warehouse cell or driveway cell. We pick
the lowest-(row, col) candidate for reproducibility:

- Outbound (type 0): pickup is a representative cell holding the SKU in
  the warehouse at ``t = 0``; delivery is a representative empty
  driveway cell.
- Inbound (type 1):  pickup is a representative empty driveway cell;
  delivery is a representative empty warehouse cell.

If the warehouse state at ``t = 0`` lacks a SKU that an outbound task
needs (which can happen for a far-future outbound that depends on an
intermediate inbound) we fall back to a warehouse-region centroid -- the
TSP only consumes distances and the *actual* pickup cell is resolved at
path-planning time anyway. This matches the paper's framing of TSP edge
weights as **estimates**.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import elkai
import numpy as np
from numpy.typing import NDArray

from ..agent import Agent, AgentLoader
from ..graph import Graph
from ..import_schedule import (
    SCHEDULE_DEADLINE,
    SCHEDULE_RELEASE_TIME,
    SCHEDULE_SKU_ID,
    SCHEDULE_SKU_ID_OFFSET,
    SCHEDULE_TASK_TYPE,
)
from ..logging_config import get_logger


_LOG = get_logger("alloc.ta_hybrid")

Loc = Tuple[int, int]

TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1


# ---------------------------------------------------------------------------
# Materialised task representation
# ---------------------------------------------------------------------------
class MaterializedTask:
    """A schedule row resolved into concrete pickup/delivery cells.

    Carries the original ``schedule_row_index`` so the outer TA-Hybrid driver
    can match this back to the live ``J`` task id that ``add_tasks_from_schedule``
    assigns when the task is released. With the deterministic ordering used by
    ``materialize_schedule``, the schedule-row index also equals the position
    of the task within the per-tick release sequence, which makes the
    materialised plan stable across simulator ticks.
    """

    __slots__ = (
        "schedule_row_index",
        "release_time",
        "deadline",
        "sku_id",
        "task_type",
        "pickup",
        "delivery",
    )

    def __init__(
        self,
        schedule_row_index: int,
        release_time: int,
        deadline: int,
        sku_id: int,
        task_type: int,
        pickup: Loc,
        delivery: Loc,
    ) -> None:
        self.schedule_row_index = schedule_row_index
        self.release_time = release_time
        self.deadline = deadline
        self.sku_id = sku_id
        self.task_type = task_type
        self.pickup = pickup
        self.delivery = delivery

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"MaterializedTask(idx={self.schedule_row_index}, r={self.release_time}, "
            f"d={self.deadline}, sku={self.sku_id}, type={self.task_type}, "
            f"s={self.pickup}, g={self.delivery})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaterializedTask):
            return NotImplemented
        return (
            self.schedule_row_index == other.schedule_row_index
            and self.release_time == other.release_time
            and self.deadline == other.deadline
            and self.sku_id == other.sku_id
            and self.task_type == other.task_type
            and self.pickup == other.pickup
            and self.delivery == other.delivery
        )


# ---------------------------------------------------------------------------
# Step 1: representative-cell helpers
# ---------------------------------------------------------------------------
def _first_sorted(locations) -> Optional[Loc]:
    """Return the lexicographically smallest (row, col) in ``locations`` or ``None``."""
    if not locations:
        return None
    return min(tuple(loc) for loc in locations)


def _warehouse_region_centroid(G: Graph) -> Loc:
    """Approximate centroid of warehouse aisle cells (fallback for outbound when
    the requested SKU isn't in the warehouse at ``t = 0``)."""
    aisles = G.get_aisle_locations()
    if not aisles:
        raise ValueError("Graph has no aisle locations; cannot pick a warehouse fallback")
    rows = [r for r, _ in aisles]
    cols = [c for _, c in aisles]
    centroid = (int(round(sum(rows) / len(rows))), int(round(sum(cols) / len(cols))))
    # snap to the nearest actual aisle cell (so distance lookups don't see a hole)
    return min(aisles, key=lambda loc: abs(loc[0] - centroid[0]) + abs(loc[1] - centroid[1]))


def _driveway_region_centroid(G: Graph) -> Loc:
    """Approximate centroid of driveway cells (fallback for inbound)."""
    stations = G.get_station_locations()
    if not stations:
        raise ValueError("Graph has no station locations; cannot pick a driveway fallback")
    rows = [r for r, _ in stations]
    cols = [c for _, c in stations]
    centroid = (int(round(sum(rows) / len(rows))), int(round(sum(cols) / len(cols))))
    return min(stations, key=lambda loc: abs(loc[0] - centroid[0]) + abs(loc[1] - centroid[1]))


def _materialize_one(
    schedule_row_index: int,
    row: NDArray,
    G: Graph,
    sku_id_offset: int,
) -> MaterializedTask:
    """Pick a representative ``(pickup, delivery)`` pair for a single schedule row.

    Choices are deterministic (lowest ``(row, col)`` wins) so that two runs with
    the same schedule produce the same task sequences. Fallbacks for the edge
    cases (SKU not currently in warehouse / no empty cells) are documented
    inline.
    """
    release_time = int(row[SCHEDULE_RELEASE_TIME])
    deadline = int(row[SCHEDULE_DEADLINE])
    sku_id = int(row[SCHEDULE_SKU_ID]) + sku_id_offset
    task_type = int(row[SCHEDULE_TASK_TYPE])

    if task_type == TASK_TYPE_OUTBOUND:
        instances = G.warehouse.get_sku_instances(sku_id) if hasattr(G, "warehouse") else set()
        pickup = _first_sorted(instances)
        if pickup is None:
            # SKU not in warehouse at t=0 (e.g. depends on a future inbound).
            # Use the warehouse centroid as a distance proxy.
            pickup = _warehouse_region_centroid(G)
            _LOG.debug(
                "outbound schedule row %d wants sku %d not in warehouse at t=0; "
                "using warehouse centroid %s as pickup proxy",
                schedule_row_index, sku_id, pickup,
            )
        delivery = _first_sorted(G.driveway.get_empty_locations()) if hasattr(G, "driveway") else None
        if delivery is None:
            delivery = _driveway_region_centroid(G)
    elif task_type == TASK_TYPE_INBOUND:
        # Canonical inbound pickup cell: the lowest-(row, col) driveway cell
        # over *all* driveway cells (empty OR full). This must match
        # ``import_schedule.add_tasks_from_schedule(deterministic=True)``,
        # which always places inbound SKUs at this same cell so the
        # pre-committed task tuple stays valid throughout the run.
        if hasattr(G, "driveway"):
            empty = set(G.driveway.get_empty_locations())
            full = set(G.driveway.get_full_locations()) if hasattr(G.driveway, "get_full_locations") else set()
            all_driveway_cells = empty | full
            pickup = _first_sorted(all_driveway_cells) if all_driveway_cells else None
        else:
            pickup = None
        if pickup is None:
            pickup = _driveway_region_centroid(G)
        delivery = _first_sorted(G.warehouse.get_empty_locations()) if hasattr(G, "warehouse") else None
        if delivery is None:
            delivery = _warehouse_region_centroid(G)
    else:
        raise ValueError(
            f"Unsupported task_type={task_type} in schedule row {schedule_row_index}; "
            f"TA-Hybrid expects task_type in {{0, 1}}."
        )

    return MaterializedTask(
        schedule_row_index=schedule_row_index,
        release_time=release_time,
        deadline=deadline,
        sku_id=sku_id,
        task_type=task_type,
        pickup=pickup,
        delivery=delivery,
    )


def materialize_schedule(
    schedule: NDArray,
    G: Graph,
    sku_id_offset: int = SCHEDULE_SKU_ID_OFFSET,
) -> List[MaterializedTask]:
    """Materialise every schedule row into a ``MaterializedTask``.

    Rows are sorted by ``release_time`` (stable) so the resulting list order
    matches the order in which ``add_tasks_from_schedule`` will release the
    tasks at runtime. The list index ``i`` corresponds to ``task_id = i + 1``
    in M2M's running ``J`` (CRG / ``add_tasks_from_schedule`` use
    ``last_task_id += 1`` before assigning, starting from ``last_task_id = 0``).
    """
    if schedule is None or schedule.size == 0:
        return []

    sched = np.atleast_2d(schedule)
    # Stable sort by release time so ties keep their original schedule-row
    # order, matching add_tasks_from_schedule's `_due_schedule_tasks` exactly.
    order = np.argsort(sched[:, SCHEDULE_RELEASE_TIME], kind="stable")
    materialized: List[MaterializedTask] = []
    for i, original_row_idx in enumerate(order):
        materialized.append(
            _materialize_one(
                schedule_row_index=int(original_row_idx),
                row=sched[original_row_idx],
                G=G,
                sku_id_offset=sku_id_offset,
            )
        )
    return materialized


# ---------------------------------------------------------------------------
# Step 2: G' assignment-graph construction (paper Section 3.1)
# ---------------------------------------------------------------------------
def _dist_int(G: Graph, a: Loc, b: Loc) -> int:
    """Shortest-path distance from ``a`` to ``b``, cast to int.

    ``Graph.get_distance`` returns ``len(path) - 1`` which is naturally
    integral, but typed as ``float``; LKH-3 / elkai require ``int`` edge
    weights so we coerce here in one place.
    """
    d = G.get_distance(a, b)
    if d == float("inf") or (isinstance(d, float) and np.isinf(d)):
        # Treat unreachable as a very large but finite penalty so the TSP
        # solver still converges. In well-formed MAPD instances this should
        # never fire; we surface it via a warning.
        _LOG.warning("infinite distance between %s and %s; using 10**6 penalty", a, b)
        return 10**6
    return int(d)


def build_assignment_matrix(
    agents: Sequence[Agent],
    tasks: Sequence[MaterializedTask],
    G: Graph,
) -> NDArray:
    """Build the ``(M+N) x (M+N)`` asymmetric cost matrix for paper Section 3.1.

    Indexing convention:

    - Row / column ``i in [0, M)`` represents agent ``agents[i]``.
    - Row / column ``M + j``     represents task ``tasks[j]``.

    Edge weights (paper Section 3.1):

    -  ``w(alpha_i, tau_j) = max(dist(p_i, s_j), r_j)`` -- agent to task.
    -  ``w(tau_i, tau_j)   = dist(s_i, g_i) + dist(g_i, s_j)`` -- task to task.
    -  ``w(tau_i, alpha_j) = dist(s_i, g_i)`` -- task to agent (independent of
       which agent ``alpha_j`` is -- the cycle just needs *some* exit edge).
    -  ``w(alpha_i, alpha_j) = 0`` -- agent to agent (zero so an unassigned
       agent can be "skipped" by the Hamiltonian cycle without cost).

    Diagonal is zero (every node trivially reaches itself). LKH-3 ignores
    self-loops in the TSP formulation.
    """
    M = len(agents)
    N = len(tasks)
    size = M + N
    cost = np.zeros((size, size), dtype=np.int64)

    # Cache per-task pickup-cell and pickup-to-delivery distance.
    task_pickup = [t.pickup for t in tasks]
    task_delivery = [t.delivery for t in tasks]
    task_release = [t.release_time for t in tasks]
    task_traverse = [
        _dist_int(G, tasks[j].pickup, tasks[j].delivery) for j in range(N)
    ]

    # alpha_i -> tau_j  (top-right block)
    for i, agent in enumerate(agents):
        for j in range(N):
            travel = _dist_int(G, agent.home, task_pickup[j])
            cost[i, M + j] = max(travel, task_release[j])

    # tau_i -> tau_j   (bottom-right block, excluding diagonal)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cost[M + i, M + j] = task_traverse[i] + _dist_int(
                G, task_delivery[i], task_pickup[j]
            )

    # tau_i -> alpha_j (bottom-left block)
    for i in range(N):
        for j in range(M):
            cost[M + i, j] = task_traverse[i]

    # alpha_i -> alpha_j (top-left block) is already zero.
    return cost


# ---------------------------------------------------------------------------
# Step 3: special TSP via LKH-3
# ---------------------------------------------------------------------------
def solve_special_tsp(cost: NDArray, runs: int = 1) -> List[int]:
    """Solve the asymmetric TSP on ``cost`` and return the tour.

    The returned tour is a list of vertex indices that starts and ends at
    vertex 0. ``elkai`` rotates the tour so that vertex 0 is first; we keep
    that convention here so callers can rely on a fixed start.

    Args:
        cost: ``(N, N)`` integer cost matrix (asymmetric is fine).
        runs: Number of LKH restarts. The default of 1 matches the
            ``elkai`` default. The paper uses LKH for up to 1000 seconds
            (small warehouse) or 6000 seconds (large warehouse); we
            expose ``runs`` so the M2M caller can tune this.

    Notes:
        ``elkai.DistanceMatrix`` accepts asymmetric matrices since v2.0
        (Helsgaun's LKH-3 natively handles ATSP). Each row of ``cost``
        must be a list of ``int``-castable values.
    """
    if cost.shape[0] == 0:
        return []
    if cost.shape[0] == 1:
        return [0]

    matrix = elkai.DistanceMatrix(cost.astype(int).tolist())
    if runs == 1:
        tour = matrix.solve_tsp()
    else:
        tour = matrix.solve_tsp(runs=runs)
    # elkai returns a tour that closes back on the start vertex.
    # Strip the trailing closing index so partitioning is easier.
    if len(tour) > 1 and tour[0] == tour[-1]:
        tour = tour[:-1]
    return list(tour)


# ---------------------------------------------------------------------------
# Step 4: partition the Hamiltonian cycle into per-agent task sequences
# ---------------------------------------------------------------------------
def partition_tour(
    tour: Sequence[int],
    num_agents: int,
) -> Dict[int, List[int]]:
    """Partition a Hamiltonian cycle on ``V' = A union T`` into per-agent task lists.

    Args:
        tour: The cycle as produced by :func:`solve_special_tsp` (start vertex
            first, *not* repeated at the end).
        num_agents: ``M = |A|``. Vertices ``[0, M)`` are agents, vertices
            ``[M, M + N)`` are tasks.

    Returns:
        Dict mapping each agent index ``i in [0, M)`` to the ordered list of
        task indices (``j``, relative to the materialised-task list) assigned
        to that agent. Agents with no tasks get ``[]``.

    M2M-specific re-balancing -- the paper's special-TSP cost matrix
    (alpha_i->alpha_j = 0) lets LKH cluster every alpha vertex back-to-back
    in the cycle and dump all tasks onto a single agent, which is
    technically optimal for the all-pairs travel objective but produces a
    catastrophic load imbalance when run on M2M's continuous-release
    schedules where 30+ agents are expected to share work. To preserve
    paper-style cycle semantics for the *common* case while degrading
    gracefully on the *pathological* cluster case, we:

    1.  First compute the standard "attribute each task to the most-recent
        agent vertex" partition.
    2.  If the resulting assignment is unbalanced beyond
        ``2 * ceil(N / M)`` for any agent, fall back to a chunked
        round-robin redistribution: extract tasks in tour order and
        slice them into ``M`` contiguous blocks of size ``ceil(N / M)``.

    The fallback preserves intra-chunk TSP locality (consecutive tasks in
    a chunk are spatially correlated by LKH's tour) while ensuring every
    agent has roughly equal work, which is what makes 30+ agent
    parallelism actually deliver on its promise.
    """
    sequences: Dict[int, List[int]] = {i: [] for i in range(num_agents)}
    if not tour:
        return sequences

    # The cycle visits each agent vertex exactly once. We walk the tour and
    # attribute every task vertex to the most-recently-seen agent. The cycle
    # is closed (last agent's task tail belongs to that last agent before the
    # cycle wraps back to the first), which we handle by starting the walk at
    # the first agent vertex and treating tasks before it as belonging to the
    # last agent vertex.
    first_agent_pos = None
    for pos, vertex in enumerate(tour):
        if vertex < num_agents:
            first_agent_pos = pos
            break

    if first_agent_pos is None:
        # Pathological: tour contains no agent vertices. Shouldn't happen
        # because every Hamiltonian cycle on G' must include every alpha_i.
        raise ValueError("tour contains no agent vertices; not a valid Hamiltonian cycle on G'")

    # Rotate so the tour begins at an agent vertex.
    rotated = list(tour[first_agent_pos:]) + list(tour[:first_agent_pos])
    current_agent: Optional[int] = None
    for vertex in rotated:
        if vertex < num_agents:
            current_agent = vertex
        else:
            assert current_agent is not None
            sequences[current_agent].append(vertex - num_agents)

    # Imbalance check + chunked-round-robin fallback.
    num_tasks = sum(len(s) for s in sequences.values())
    if num_tasks == 0 or num_agents == 0:
        return sequences
    target = -(-num_tasks // num_agents)  # ceil(N / M)
    max_load = max(len(s) for s in sequences.values())
    if max_load <= 2 * target:
        return sequences

    _LOG.info(
        "partition_tour: max_load=%d > 2 * ceil(N/M)=%d (N=%d, M=%d); "
        "redistributing via chunked round-robin",
        max_load,
        2 * target,
        num_tasks,
        num_agents,
    )

    # Tasks in tour order (skip agent vertices entirely).
    tasks_in_tour: List[int] = [v - num_agents for v in rotated if v >= num_agents]
    rebalanced: Dict[int, List[int]] = {i: [] for i in range(num_agents)}
    for i in range(num_agents):
        chunk = tasks_in_tour[i * target : (i + 1) * target]
        rebalanced[i] = chunk
    return rebalanced


# ---------------------------------------------------------------------------
# Step 5: per-sequence execution times (paper Section 3.2)
# ---------------------------------------------------------------------------
def compute_execution_times(
    sequences: Dict[int, List[int]],
    agents: Sequence[Agent],
    tasks: Sequence[MaterializedTask],
    G: Graph,
) -> Dict[int, int]:
    """Compute the execution time ``M_i`` for each agent's task sequence.

    Per paper Section 3.2:

    -  ``start(t_{i,1}) = w(alpha_i, tau_{i,1}) = max(dist(p_i, s_1), r_1)``
    -  For ``k >= 2``:
       ``start(t_{i,k}) = max(start(t_{i,k-1}) + w(tau_{k-1}, tau_k), r_{i,k})``
    -  ``M_i = start(t_{i,L_i}) + w(tau_{L_i}, alpha_{i+1})
                = start(t_{i,L_i}) + dist(s_{i,L_i}, g_{i,L_i})``

    The trailing ``w(tau, alpha)`` term is just the pickup-to-delivery
    distance of the last task (the alpha-to-alpha edge that closes the cycle
    is zero, and tau-to-alpha is by definition the task's own traversal
    distance).

    Returns a dict ``agent_idx -> execution_time`` for every agent in
    ``sequences`` (including agents whose ``sequences[i]`` is empty -- their
    execution time is 0 since they never move).
    """
    exec_times: Dict[int, int] = {}
    for agent_idx, task_indices in sequences.items():
        if not task_indices:
            exec_times[agent_idx] = 0
            continue
        agent = agents[agent_idx]
        first = tasks[task_indices[0]]
        travel_to_first = _dist_int(G, agent.home, first.pickup)
        start = max(travel_to_first, first.release_time)
        for k in range(1, len(task_indices)):
            prev = tasks[task_indices[k - 1]]
            curr = tasks[task_indices[k]]
            prev_traverse = _dist_int(G, prev.pickup, prev.delivery)
            transition = _dist_int(G, prev.delivery, curr.pickup)
            start = max(start + prev_traverse + transition, curr.release_time)
        last = tasks[task_indices[-1]]
        last_traverse = _dist_int(G, last.pickup, last.delivery)
        exec_times[agent_idx] = start + last_traverse
    return exec_times


# ---------------------------------------------------------------------------
# Step 6: top-level entry point
# ---------------------------------------------------------------------------
def ta_assignment_plan(
    G: Graph,
    Rs: AgentLoader,
    schedule: NDArray,
    sku_id_offset: int = SCHEDULE_SKU_ID_OFFSET,
    runs: int = 1,
) -> Tuple[Dict[int, List[MaterializedTask]], Dict[int, int]]:
    """Run the full TA-assignment stage on a schedule.

    Args:
        G: M2M ``Graph`` (used for distances and warehouse / driveway state).
        Rs: ``AgentLoader`` carrying agents with ``home`` parking locations.
        schedule: ``(N, 4)`` NDArray of
            ``(release_time, deadline, sku_id, task_type)`` rows, as loaded by
            :func:`import_schedule.import_schedule`.
        sku_id_offset: Schedule files are 0-indexed; warehouse SKUs are
            1-indexed (see :mod:`import_schedule`).
        runs: LKH-3 restarts (forwarded to :func:`solve_special_tsp`).

    Returns:
        ``(plan, exec_times)`` where

        -   ``plan`` is a dict ``agent_id -> List[MaterializedTask]`` giving
            each agent its TSP-ordered task sequence. The outer TA-Hybrid
            driver uses each task's ``schedule_row_index + 1`` to map back to
            the live ``J`` task id at runtime.
        -   ``exec_times`` is a dict ``agent_id -> int`` of per-sequence
            execution times (Section 3.2) for diagnostics / logging.

    Raises:
        ValueError: if the schedule contains unsupported task types or if
            the graph has no aisle / driveway cells (which would make
            materialisation impossible).
    """
    agents = list(Rs.agents)
    materialized = materialize_schedule(schedule, G, sku_id_offset=sku_id_offset)
    if not materialized:
        plan = {a.id: [] for a in agents}
        return plan, {a.id: 0 for a in agents}

    cost = build_assignment_matrix(agents, materialized, G)
    tour = solve_special_tsp(cost, runs=runs)
    sequences = partition_tour(tour, num_agents=len(agents))
    exec_times_by_idx = compute_execution_times(sequences, agents, materialized, G)

    plan: Dict[int, List[MaterializedTask]] = {}
    exec_times_by_id: Dict[int, int] = {}
    for agent_idx, task_indices in sequences.items():
        agent_id = agents[agent_idx].id
        plan[agent_id] = [materialized[j] for j in task_indices]
        exec_times_by_id[agent_id] = exec_times_by_idx[agent_idx]

    _LOG.info(
        "TA-assignment: %d agents, %d tasks, makespan=%d (sum=%d)",
        len(agents),
        len(materialized),
        max(exec_times_by_id.values()) if exec_times_by_id else 0,
        sum(exec_times_by_id.values()),
    )
    return plan, exec_times_by_id


def compute_per_task_L_bounds(
    plan: Dict[int, List[MaterializedTask]],
    exec_times: Dict[int, int],
    G: Graph,
) -> Dict[Tuple[int, int], int]:
    """Per-task latest-arrival deadline ``L_{i,k}`` (paper Section 3.2).

    For agent ``i``'s ``k``-th task ``t_{i,k}`` the paper defines the
    pickup-arrival deadline as

        L_{i,k} = L - tail_{i,k}

    where ``L`` is the global makespan target (we use
    ``max_i M_i`` from :func:`compute_execution_times`) and
    ``tail_{i,k}`` is the time from the moment the agent arrives at
    ``s_{i,k}`` until it finishes ``t_{i,L_i}`` (the agent's last
    task), assuming no release-time waits beyond ``k``:

        tail_{i,L_i} = w(tau_{i,L_i}, alpha_{i+1})  -- pickup -> delivery
                                                    of the last task
        tail_{i,k}   = w(tau_{i,k}, alpha_{i,k+1})  -- pickup -> delivery
                       + w(alpha_{i,k+1}, tau_{i,k+1})  -- transition
                       + tail_{i,k+1}

    AMAPF uses ``L_{i,k}`` as the upper bound of the meta-vertex
    window for ``s_{i,k}``: the walker assigned to agent ``i``'s
    head task must arrive at ``s_{i,k}`` no later than
    ``L_{i,k}``, otherwise the agent cannot finish its remaining
    tasks within the global makespan. This is strictly tighter than
    using a single global ``L`` for every (agent, task) pair (the
    pre-existing ``driver.initial_L`` path), and matches the paper's
    definition exactly.

    Returns a dict ``(agent_id, schedule_row_index) -> L_{i,k}`` for
    every (agent, task) pair in ``plan``. Tasks not in any agent's
    plan are simply absent from the result.

    Note: bounds are conservative -- they do not credit any
    release-time slack the agent might absorb between ``k`` and
    ``L_i`` (such absorption would let the agent miss ``L_{i,k}``
    and still finish on time). Conservatism is the safe direction;
    a looser bound could let AMAPF return a flow that pushes the
    realised makespan above ``L``.
    """
    if not exec_times:
        return {}
    global_L = max(exec_times.values())

    bounds: Dict[Tuple[int, int], int] = {}
    for agent_id, tasks in plan.items():
        if not tasks:
            continue
        # Backward DP over the agent's task order.
        K = len(tasks)
        # tail for the last task = its own pickup -> delivery distance
        tail = _dist_int(G, tasks[K - 1].pickup, tasks[K - 1].delivery)
        bounds[(agent_id, tasks[K - 1].schedule_row_index)] = global_L - tail
        for k in range(K - 2, -1, -1):
            curr = tasks[k]
            nxt = tasks[k + 1]
            curr_traverse = _dist_int(G, curr.pickup, curr.delivery)
            transition = _dist_int(G, curr.delivery, nxt.pickup)
            tail = curr_traverse + transition + tail
            bounds[(agent_id, curr.schedule_row_index)] = global_L - tail
    return bounds
