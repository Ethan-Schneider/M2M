"""Min-Cost Max-Flow AMAPF solver for TA-Hybrid Group 2 (Section 5.4).

The paper Liu, Ma, Li, Koenig (AAMAS 2019), Section 5.4, defines a
time-extended directed flow network whose integer min-cost max-flow,
when feasible at flow value ``|A'|``, corresponds to collision-free
sub-paths for the agents in subgroup ``A'`` from their current
locations to their currently-assigned pickup locations.

This module builds and solves that network. It does **not** do subgroup
partitioning, edge-collision resolution, or dummy-path planning -- those
are Phase C.2's job (see :mod:`ta_hybrid`).

Paper-exact network construction (per Section 5.4 + Figure 3)
-------------------------------------------------------------

Given subgroup ``A' = {a_1, ..., a_k}`` with current locations ``c_i``,
assigned pickups ``s_i`` (pairwise distinct across the subgroup --
guaranteed by partitioning in Section 5.3), release times ``r_i``, and
per-task deadlines ``L_i``; given current timestep ``t0``, makespan
bound ``L = max_i L_i``, and a set of other-agent sub-paths to avoid:

-   For each open grid cell ``v`` at ``t = t0``: a single node
    ``v_out_{t0}``.
-   For each open grid cell ``v`` at each ``t in [t0+1, L]``: two nodes
    ``v_in_t`` and ``v_out_t``, connected by a vertex-capacity edge
    ``(v_in_t, v_out_t)`` with cost 0, capacity 1.
-   For each grid edge ``(u, v)`` and ``t in [t0, L-1]``: a move edge
    ``(u_out_t, v_in_{t+1})`` with cost 1, capacity 1.
-   For each open cell ``v`` and ``t in [t0, L-1]``: a wait edge
    ``(v_out_t, v_in_{t+1})`` with cost 1, capacity 1.
-   Source ``S`` connected to ``c_i_out_{t0}`` for each agent ``i in A'``
    with cost 0, capacity 1.
-   For each agent ``i in A'``, a "meta" node ``m_i``:

    -   ``(s_i_out_t, m_i)`` cost 0, capacity 1, for every
        ``t in [max(r_i, r'_i + 1), L_i]``, where ``r'_i`` is the latest
        timestep at which any out-of-subgroup agent's sub-path passes
        through ``s_i`` (``r'_i = t0 - 1`` if none does).
    -   ``(m_i, T)`` cost 0, capacity 1.

-   **Pickup-hold constraint:** for each agent ``i in A'`` and each
    ``t in [r_i, L-1]``, remove every move edge ``(s_i_out_t,
    u_in_{t+1})`` for grid neighbours ``u`` of ``s_i``. Wait edges from
    ``s_i`` to itself are **not** removed -- the holder waits there.
-   **Other-agent vertex avoidance:** for each other-agent path
    ``P = (p_{t0}, p_{t0+1}, ...)`` and each ``t in [t0, L]``, remove
    the vertex-capacity edge ``(p_t_in_t, p_t_out_t)`` (no removal at
    ``t = t0`` since ``v_in_{t0}`` does not exist).
-   **Other-agent edge avoidance:** for each consecutive pair
    ``(p_t, p_{t+1})`` in an other-agent path and each
    ``t in [t0, L-1]``, remove the swap-conflict move edge
    ``(p_{t+1}_out_t, p_t_in_{t+1})``.

Solve via ``networkx.max_flow_min_cost`` which gives first priority to
maximum flow and second priority to minimum total cost -- exactly the
paper's lexicographic objective.

Outputs
-------

Per-walker results: ``AmapfWalker`` records ``start_agent_id`` (the
agent whose current_loc was at the trace's start), ``assigned_task_id``
(the agent whose meta vertex the trace ended at), and the absolute
sequence of cells visited from ``t0`` through ``t_arrive``. When the
two ids differ, the original task assignment has been swapped by the
flow solver (Section 5.3: "if an agent is assigned the current pickup
location of a different agent, TA-Hybrid replaces its current task
sequence with the task sequence of this different agent"). Phase C.2
applies that swap.

Edge collisions between walkers can still occur in the returned paths
since the flow network only enforces vertex collisions; Section 5.4's
post-hoc swap-resolution step is Phase C.2's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

# Optional fast MCMF backend. We default to OR-tools' ``SimpleMinCostFlow``
# (cost-scaling push-relabel in C++, ~100-1000x faster than the pure-Python
# ``networkx.max_flow_min_cost`` on AMAPF-shaped graphs). Falling back to
# networkx if the import fails keeps the test suite runnable on machines
# that haven't installed ortools (and lets us keep the legacy code path
# around for cross-validation when debugging).
try:
    from ortools.graph.python import min_cost_flow as _ortools_mcf
    _ORTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised by the fallback path
    _ortools_mcf = None  # type: ignore[assignment]
    _ORTOOLS_AVAILABLE = False

from ..graph import Graph
from ..logging_config import get_logger


_LOG = get_logger("alloc.ta_hybrid.amapf")

Loc = Tuple[int, int]
Node = Any   # one of: 'S', 'T', ('in', loc, t), ('out', loc, t), ('meta', agent_id)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AmapfAgent:
    """One agent in a single AMAPF subgroup.

    All agents in a subgroup have **pairwise distinct** ``pickup_loc``
    (precondition enforced by the C.2 partitioner; this module does not
    re-check it). Deadlines ``L_i`` are derived externally from the
    paper's makespan bound and the agent's remaining estimated execution
    time (Section 5.4 paragraph 1).
    """

    agent_id: int
    current_loc: Loc       # c_i at t0
    pickup_loc: Loc        # s_i for the agent's current task
    release_time: int      # r_i (absolute timestep)
    deadline: int          # L_i (absolute timestep, the latest arrival at s_i)


@dataclass(frozen=True)
class AmapfWalker:
    """One unit of flow traced from source to sink.

    ``subpath[0] == start_loc``, ``subpath[-1] == pickup_loc`` of the
    walker's *assigned* (post-swap) task. ``subpath[i]`` is the cell
    occupied at absolute timestep ``t0 + i``.
    """

    start_agent_id: int          # agent whose current_loc began this walk
    assigned_task_agent_id: int  # agent whose meta vertex this walk hit
    subpath: List[Loc]


@dataclass(frozen=True)
class AmapfResult:
    """Result of a successful AMAPF solve.

    ``walkers`` is in the same order the trace was performed. The C.2
    caller is responsible for re-assigning task sequences when
    ``start_agent_id != assigned_task_agent_id`` and for resolving any
    edge collisions between walkers' subpaths.
    """

    walkers: List[AmapfWalker]
    makespan_bound_used: int
    total_cost: int


# ---------------------------------------------------------------------------
# Network construction (private)
# ---------------------------------------------------------------------------
def _open_cells_and_neighbors(G: Graph) -> Tuple[List[Loc], Dict[Loc, List[Loc]]]:
    """Enumerate non-obstacle cells and their 4-connected neighbours."""
    rows, cols = G.get_graph_size()
    cells: List[Loc] = []
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if not G.get_if_obstacle(cell):
                cells.append(cell)
    cell_set = set(cells)
    neighbors: Dict[Loc, List[Loc]] = {}
    for cell in cells:
        r, c = cell
        nbs: List[Loc] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cell_set:
                nbs.append(nb)
        neighbors[cell] = nbs
    return cells, neighbors


def _expand_path(
    entry: Tuple,
    horizon_t: int,
) -> Tuple[List[Tuple[Loc, int]], bool]:
    """Expand a path entry into a list of ``(cell, abs_t)`` occupancy.

    Accepts both 3-tuples ``(start_loc, path, start_t)`` and 4-tuples
    ``(start_loc, path, start_t, is_permanent_terminal)``. When the
    permanent-terminal flag is set, the agent's terminal cell is
    repeated through ``horizon_t`` (inclusive) so the
    network-construction routines see it as an indefinite reservation.

    Returns ``(occupancy_list, is_permanent_terminal)`` where the
    occupancy list is ``[(start_loc, start_t), (path[0], start_t+1),
    ..., (terminal, horizon_t)]`` when permanent, or truncated to the
    natural path end otherwise.
    """
    if len(entry) == 4:
        start_loc, path, start_t, is_perm = entry
    elif len(entry) == 3:
        start_loc, path, start_t = entry
        is_perm = False
    else:
        raise ValueError(f"unexpected other-agent path tuple shape: {entry}")

    occ: List[Tuple[Loc, int]] = [(start_loc, start_t)]
    for i, p in enumerate(path):
        occ.append((p, start_t + 1 + i))
    if is_perm:
        terminal = path[-1] if path else start_loc
        last_t = start_t + len(path)
        # Extend through horizon_t inclusive (we want the terminal cell
        # to occupy every timestep from last_t + 1 up through horizon_t).
        for t in range(last_t + 1, horizon_t + 1):
            occ.append((terminal, t))
    return occ, is_perm


def _last_other_path_visit(
    other_agent_paths: Sequence[Tuple],
    cell: Loc,
    t0: int,
    horizon_t: int,
) -> Optional[int]:
    """Latest absolute timestep ``t >= t0`` at which any other-agent
    path occupies ``cell``. Returns ``None`` if no other path passes
    through ``cell`` at or after ``t0``.

    Permanent-terminal entries are honoured: an agent indefinitely
    parked at ``cell`` from time ``t_p`` onwards contributes
    ``min(horizon_t, ...)``. ``horizon_t`` should be the planner's L.
    """
    last = None
    for entry in other_agent_paths:
        occ, _ = _expand_path(entry, horizon_t)
        for c, t in occ:
            if c == cell and t >= t0:
                last = t if last is None else max(last, t)
    return last


def build_amapf_network(
    G: Graph,
    agents: Sequence[AmapfAgent],
    other_agent_paths: Sequence[Tuple],
    t0: int,
    L: int,
    enforce_pickup_hold: bool = True,
) -> nx.DiGraph:
    """Build the time-extended flow network of paper Section 5.4.

    Returns a networkx ``DiGraph`` with capacity / weight attributes on
    every edge, and demand ``-len(agents)`` on ``'S'`` and ``+len(agents)``
    on ``'T'`` (the rest are 0). Pass it to
    :func:`networkx.max_flow_min_cost` or to :func:`solve_amapf_single_L`.

    Args:
        G: M2M ``Graph`` (provides obstacle layout via ``get_if_obstacle``
            and grid dimensions via ``get_graph_size``).
        agents: Subgroup ``A'``. All ``pickup_loc`` are assumed
            pairwise distinct.
        other_agent_paths: Sub-paths of agents not in ``A'`` (in
            absolute time). Each element is ``(start_loc, path,
            start_t)`` where occupancy is ``start_loc`` at ``start_t``
            and ``path[i]`` at ``start_t + 1 + i``. Used for vertex /
            edge avoidance.
        t0: Current timestep (start of the AMAPF planning window).
        L: Upper bound timestep -- the network extends through ``t = L``.
        enforce_pickup_hold: When ``True`` (the default, paper Section
            5.4) move-out edges from each pickup cell ``s_i`` are
            removed for ``t in [r_i, L-1]`` so no walker can leave the
            cell after the corresponding task's release. When ``False``
            (paper Section 5.5 "first try without holds" optimisation)
            those edges are kept, the network is strictly less
            constrained, and the caller must post-check the returned
            walkers for pickup-cell overlaps and retry with
            ``enforce_pickup_hold=True`` if any are found.

    Raises:
        ValueError: ``L < t0``, or any agent's current/pickup cell is
            an obstacle.
    """
    if L < t0:
        raise ValueError(f"L ({L}) must be >= t0 ({t0})")
    if not agents:
        raise ValueError("AMAPF subgroup must be non-empty")

    cells, neighbors_of = _open_cells_and_neighbors(G)
    cell_set = set(cells)

    for ag in agents:
        if ag.current_loc not in cell_set:
            raise ValueError(
                f"agent {ag.agent_id} current_loc {ag.current_loc} is an obstacle"
            )
        if ag.pickup_loc not in cell_set:
            raise ValueError(
                f"agent {ag.agent_id} pickup_loc {ag.pickup_loc} is an obstacle"
            )

    N = nx.DiGraph()
    N.add_node("S", demand=-len(agents))
    N.add_node("T", demand=len(agents))

    # ----- vertex-capacity edges (cost 0, cap 1) for t in [t0+1, L] -----
    # These also implicitly create v_in_t and v_out_t for those (cell, t).
    for cell in cells:
        for t in range(t0 + 1, L + 1):
            N.add_edge(("in", cell, t), ("out", cell, t), capacity=1, weight=0)

    # ----- move and wait edges for t in [t0, L-1] -----
    # Each grid edge (u, v) -> move edge (u_out_t, v_in_{t+1}).
    # Each cell v -> wait edge (v_out_t, v_in_{t+1}).
    # Together these implicitly create v_out_{t0} (when t = t0).
    for t in range(t0, L):
        for cell in cells:
            out_node = ("out", cell, t)
            N.add_edge(out_node, ("in", cell, t + 1), capacity=1, weight=1)
            for nb in neighbors_of[cell]:
                N.add_edge(out_node, ("in", nb, t + 1), capacity=1, weight=1)

    # ----- source edges -----
    for ag in agents:
        N.add_edge("S", ("out", ag.current_loc, t0), capacity=1, weight=0)

    # ----- meta vertices and sink edges -----
    for ag in agents:
        meta = ("meta", ag.agent_id)
        r_prime = _last_other_path_visit(
            other_agent_paths, ag.pickup_loc, t0, horizon_t=L
        )
        if r_prime is None:
            # r'_j = t0 - 1 sentinel (Section 5.4 paragraph 4).
            r_prime = t0 - 1
        window_start = max(ag.release_time, r_prime + 1, t0)
        window_end = min(ag.deadline, L)
        # Always add the meta -> sink edge so the demand at T is correctly
        # absorbed if the agent's task IS feasible; if not, the unit of
        # flow simply cannot reach this meta and max_flow_min_cost will
        # report sub-saturation.
        N.add_edge(meta, "T", capacity=1, weight=0)
        if window_start > window_end:
            # Section 5.4: "If max(rj, r'_j+1) > Lj, then a makespan of L
            # is unachievable." We leave the meta with no incoming edges;
            # the outer L-increment loop will retry.
            continue
        for t in range(window_start, window_end + 1):
            N.add_edge(("out", ag.pickup_loc, t), meta, capacity=1, weight=0)

    # ----- pickup-hold constraints: remove move-out edges from s_i for
    # t in [r_i, L-1]. Wait edges (s_i -> s_i) are kept.
    # Skipped when enforce_pickup_hold is False (paper Section 5.5).
    if enforce_pickup_hold:
        for ag in agents:
            s = ag.pickup_loc
            for t in range(max(ag.release_time, t0), L):
                out_node = ("out", s, t)
                for nb in neighbors_of[s]:
                    edge = (out_node, ("in", nb, t + 1))
                    if N.has_edge(*edge):
                        N.remove_edge(*edge)

    # ----- other-agent constraints -----
    for entry in other_agent_paths:
        full, _is_perm = _expand_path(entry, horizon_t=L)

        # Vertex collisions: remove (p_t_in_t, p_t_out_t) for each (p_t, t)
        # in [t0+1, L]. (At t = t0 there is no v_in_{t0} so nothing to remove.)
        for p, t in full:
            if t < t0 + 1 or t > L:
                continue
            edge = (("in", p, t), ("out", p, t))
            if N.has_edge(*edge):
                N.remove_edge(*edge)

        # Edge collisions: remove (p_{t+1}_out_t, p_t_in_{t+1}) for each
        # consecutive (p_t, p_{t+1}). t is the time at p_t.
        for i in range(len(full) - 1):
            p_t, t = full[i]
            p_t1, t1 = full[i + 1]
            assert t1 == t + 1, "non-consecutive other-agent path entries"
            if t < t0 or t > L - 1:
                continue
            edge = (("out", p_t1, t), ("in", p_t, t + 1))
            if N.has_edge(*edge):
                N.remove_edge(*edge)

    return N


# ---------------------------------------------------------------------------
# Flow -> sub-paths transformation (private)
# ---------------------------------------------------------------------------
def _trace_one_walker(
    flow_dict: Dict[Node, Dict[Node, int]],
    start_node: Tuple[str, Loc, int],
    t0: int,
) -> Tuple[List[Loc], Optional[int]]:
    """Follow one unit of flow from ``start_node`` through the network
    until a meta vertex (then to sink) is reached.

    Mutates ``flow_dict`` in place, decrementing each consumed edge so
    subsequent traces don't re-use it.

    Returns ``(cells_per_tick, meta_agent_id)``. ``cells_per_tick[i]`` is
    the cell occupied at absolute timestep ``t0 + i`` (``cells_per_tick[0]``
    is the start cell). ``meta_agent_id`` is ``None`` only on an
    inconsistent trace (which would indicate a bug in the network or
    flow solver).
    """
    _kind, start_cell, t_start = start_node
    assert _kind == "out" and t_start == t0, (
        f"trace must start at an out-node at t0, got {start_node}"
    )
    cells_per_tick: List[Loc] = [start_cell]

    current: Node = start_node
    while True:
        outgoing = flow_dict.get(current, {})
        nxt: Optional[Node] = None
        for cand, f in outgoing.items():
            if f >= 1:
                nxt = cand
                outgoing[cand] = f - 1
                break
        if nxt is None:
            return cells_per_tick, None

        # Meta vertex terminates this walker.
        if isinstance(nxt, tuple) and len(nxt) == 2 and nxt[0] == "meta":
            # Also consume the meta -> T edge so it's not re-used.
            meta_out = flow_dict.get(nxt, {})
            if meta_out.get("T", 0) >= 1:
                meta_out["T"] -= 1
            return cells_per_tick, nxt[1]

        kind, cell, t = nxt
        if kind == "in":
            # Crossing into a new (cell, t) -> the agent is now at `cell` at
            # absolute time t. Record it.
            cells_per_tick.append(cell)
        # Otherwise nxt is the same cell's out-node (vertex-capacity hop);
        # nothing to record.
        current = nxt


def extract_walkers(
    flow_dict: Dict[Node, Dict[Node, int]],
    agents: Sequence[AmapfAgent],
    t0: int,
) -> List[AmapfWalker]:
    """Trace every unit of source-outgoing flow into an ``AmapfWalker``.

    Mutates ``flow_dict`` by decrementing consumed edges. Callers who
    want to reuse the flow should pass a deep copy.
    """
    # agent_id lookup by current cell at t0 (current_locs are unique per
    # subgroup so this is bijective).
    agent_at_loc: Dict[Loc, AmapfAgent] = {ag.current_loc: ag for ag in agents}

    walkers: List[AmapfWalker] = []
    source_out = flow_dict.get("S", {})
    for out_node, f in list(source_out.items()):
        if f < 1 or not isinstance(out_node, tuple) or out_node[0] != "out":
            continue
        _, start_cell, _t_start = out_node
        if start_cell not in agent_at_loc:
            # Should not happen -- the source connects only to agent start cells.
            continue
        # Consume this many units (capacity=1 so at most one per edge, but
        # keep the loop defensive).
        for _ in range(f):
            source_out[out_node] -= 1
            cells, meta_id = _trace_one_walker(flow_dict, out_node, t0)
            if meta_id is None:
                # Should never happen on a saturated flow.
                raise RuntimeError(
                    f"walker starting at {start_cell} reached a dead-end "
                    "without hitting a meta vertex"
                )
            walkers.append(
                AmapfWalker(
                    start_agent_id=agent_at_loc[start_cell].agent_id,
                    assigned_task_agent_id=meta_id,
                    subpath=cells,
                )
            )
    return walkers


# ---------------------------------------------------------------------------
# MCMF backends
# ---------------------------------------------------------------------------
def _solve_mcmf_ortools(
    network: nx.DiGraph,
    n_units: int,
    source: str = "S",
    sink: str = "T",
) -> Optional[Tuple[Dict[Any, Dict[Any, int]], int]]:
    """Solve min-cost max-flow on ``network`` via OR-tools.

    The same input graph that ``networkx.max_flow_min_cost`` accepts
    (with integer ``capacity`` and ``weight`` edge attributes) is
    translated arc-by-arc into OR-tools' ``SimpleMinCostFlow``. Source
    supply is set to ``+n_units`` and sink supply to ``-n_units``,
    which forces an exact saturating flow -- if no such flow exists
    the solver returns ``INFEASIBLE`` and we propagate ``None`` so the
    outer L-increment loop in :func:`solve_amapf` can retry at a
    larger horizon.

    Returns ``(flow_dict, total_cost)`` where ``flow_dict`` matches
    the nested-dict shape that ``nx.max_flow_min_cost`` returns
    (``flow_dict[u][v] == flow``). This lets :func:`extract_walkers`
    consume the result without modification.
    """
    smcf = _ortools_mcf.SimpleMinCostFlow()
    # OR-tools nodes are integers; map our tuple-named nodes to ids
    # via a dict + parallel list.
    node_id: Dict[Any, int] = {}
    nodes: List[Any] = []

    def _id_of(name: Any) -> int:
        idx = node_id.get(name)
        if idx is None:
            idx = len(nodes)
            node_id[name] = idx
            nodes.append(name)
        return idx

    arcs: List[Tuple[Any, Any, int]] = []  # (u, v, arc_id) for back-mapping
    for u, v, data in network.edges(data=True):
        cap = int(data.get("capacity", 0))
        wt = int(data.get("weight", 0))
        if cap <= 0:
            continue
        arc_id = smcf.add_arc_with_capacity_and_unit_cost(
            _id_of(u), _id_of(v), cap, wt
        )
        arcs.append((u, v, arc_id))

    # Make sure source / sink exist as nodes even if isolated.
    _id_of(source)
    _id_of(sink)
    smcf.set_node_supply(node_id[source], n_units)
    smcf.set_node_supply(node_id[sink], -n_units)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        # INFEASIBLE / UNBALANCED / BAD_RESULT all mean "no saturating
        # flow exists at this network shape"; same semantic as a
        # max_flow < n_units result from networkx.
        return None

    flow_dict: Dict[Any, Dict[Any, int]] = {n: {} for n in nodes}
    for u, v, arc_id in arcs:
        f = smcf.flow(arc_id)
        if f > 0:
            flow_dict[u][v] = flow_dict[u].get(v, 0) + f

    return flow_dict, int(smcf.optimal_cost())


def _solve_mcmf_networkx(
    network: nx.DiGraph,
    n_units: int,
    source: str = "S",
    sink: str = "T",
) -> Optional[Tuple[Dict[Any, Dict[Any, int]], int]]:
    """Legacy networkx backend -- kept for fallback when ortools is
    unavailable and for cross-validation in tests.
    """
    try:
        flow_dict = nx.max_flow_min_cost(
            network, source, sink, capacity="capacity", weight="weight"
        )
    except nx.NetworkXUnfeasible:
        return None
    except nx.NetworkXError as exc:
        _LOG.debug("AMAPF networkx error: %s", exc)
        return None

    total_flow = sum(flow_dict.get(source, {}).values())
    if total_flow < n_units:
        return None

    total_cost = 0
    for u, out_edges in flow_dict.items():
        for v, f in out_edges.items():
            if f == 0:
                continue
            w = network[u][v].get("weight", 0)
            total_cost += f * w
    return flow_dict, int(total_cost)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def solve_amapf_single_L(
    G: Graph,
    agents: Sequence[AmapfAgent],
    other_agent_paths: Sequence[Tuple],
    t0: int,
    L: int,
    enforce_pickup_hold: bool = True,
) -> Optional[AmapfResult]:
    """Solve the AMAPF instance for a single fixed makespan bound ``L``.

    Returns ``None`` if a feasible flow of ``len(agents)`` units cannot
    be found at this ``L``; the caller should retry with a larger
    ``L``. Returns an ``AmapfResult`` on success.

    See :func:`build_amapf_network` for the meaning of
    ``enforce_pickup_hold``. The MCMF solve is dispatched to OR-tools
    when available (much faster than networkx on graphs of the size
    AMAPF produces), with a transparent fallback to networkx.
    """
    network = build_amapf_network(
        G, agents, other_agent_paths, t0, L,
        enforce_pickup_hold=enforce_pickup_hold,
    )

    n_units = len(agents)
    if _ORTOOLS_AVAILABLE:
        result = _solve_mcmf_ortools(network, n_units)
    else:
        result = _solve_mcmf_networkx(network, n_units)
    if result is None:
        _LOG.debug(
            "AMAPF at L=%d infeasible (no saturating flow of %d units)",
            L,
            n_units,
        )
        return None
    flow_dict, total_cost = result

    walkers = extract_walkers(flow_dict, agents, t0)
    if len(walkers) != len(agents):
        # Defensive: tracing produced fewer walkers than agents, which
        # would indicate a bug in extract_walkers.
        raise RuntimeError(
            f"extract_walkers produced {len(walkers)} walkers for "
            f"{len(agents)} agents (n_units={n_units})"
        )

    return AmapfResult(
        walkers=walkers,
        makespan_bound_used=L,
        total_cost=int(total_cost),
    )


def detect_pickup_hold_violations(
    walkers: Sequence[AmapfWalker],
    agents: Sequence[AmapfAgent],
    t0: int,
) -> List[Tuple[int, int, Loc, int]]:
    """Identify pickup-cell overlaps in a without-holds AMAPF solution.

    Paper Section 5.5 specifies that when AMAPF is solved without the
    pickup-hold constraint, the algorithm must afterwards check whether
    any walker's sub-path passes through another agent's terminal
    pickup cell at a timestep at or after that agent's arrival
    (i.e., once the holder has reached its pickup and the task is
    being executed). If any such overlap exists, the solve is retried
    with ``enforce_pickup_hold=True``.

    A walker is considered to "hold" the pickup cell of the agent whose
    meta vertex it hit -- i.e. its ``assigned_task_agent_id`` -- from
    that walker's arrival time onwards. Walker w_a is in violation if
    there exists another walker w_b (``b.start_agent_id != a.start_agent_id``)
    whose sub-path passes through ``a``'s endpoint cell at any timestep
    strictly after ``a``'s arrival.

    Args:
        walkers: AMAPF walkers from a no-holds solve.
        agents: The subgroup that was solved.
        t0: Absolute simulator timestep ``subpath[0]`` corresponds to.

    Returns:
        List of violations as ``(holder_start_agent_id,
        intruder_start_agent_id, pickup_cell, abs_t)`` tuples. Empty
        when there are no violations (i.e. the no-holds solution is
        already collision-free with respect to the pickup constraint
        and can be used directly).
    """
    if not walkers:
        return []
    # Arrival time + assigned pickup cell of each walker.
    holder_info: List[Tuple[int, int, Loc]] = []
    for w in walkers:
        if not w.subpath:
            continue
        arrival_t = t0 + len(w.subpath) - 1
        # The walker arrives at its assigned task's pickup -- subpath[-1]
        # equals the pickup of the agent whose meta vertex was hit.
        holder_info.append((w.start_agent_id, arrival_t, w.subpath[-1]))

    violations: List[Tuple[int, int, Loc, int]] = []
    for w in walkers:
        for holder_id, arrival_t, pickup_cell in holder_info:
            if holder_id == w.start_agent_id:
                continue
            for i, cell in enumerate(w.subpath):
                abs_t = t0 + i
                if cell == pickup_cell and abs_t > arrival_t:
                    violations.append(
                        (holder_id, w.start_agent_id, pickup_cell, abs_t)
                    )
                    break  # one violation per (holder, intruder) pair is enough
    return violations


def solve_amapf(
    G: Graph,
    agents: Sequence[AmapfAgent],
    other_agent_paths: Sequence[Tuple],
    t0: int,
    initial_L: int,
    max_L: int = 500,
    use_5_5_optimization: bool = True,
) -> Optional[AmapfResult]:
    """Solve the AMAPF instance, incrementing the makespan bound until
    feasibility is reached (paper Section 5.4 paragraph 1).

    When ``use_5_5_optimization=True`` (the default), each ``L`` is
    first attempted with ``enforce_pickup_hold=False`` (paper Section
    5.5): if the resulting walkers have no pickup-cell violations the
    no-holds solution is returned directly; otherwise the same ``L``
    is retried with the hold constraint enabled. Only if the
    with-holds retry also fails does the loop bump ``L`` upward.

    Args:
        G: M2M graph.
        agents: Subgroup ``A'`` (pairwise distinct pickups).
        other_agent_paths: Sub-paths of agents not in ``A'``.
        t0: Current timestep.
        initial_L: Initial makespan bound (paper uses
            ``max_i estimated_execution_time(task_seq_of_a_i)``).
        max_L: Hard upper bound; the loop gives up beyond this. Default
            500 is chosen so we error out long before hitting the
            simulator's 600-tick study horizon.
        use_5_5_optimization: Toggle the no-holds-first behaviour.
            Setting this to ``False`` reverts to the canonical Section
            5.4 path (always solve with holds) and is mainly useful
            for tests and ablations.

    Returns:
        ``AmapfResult`` once feasible; ``None`` if no feasible solution
        is found by ``max_L``.
    """
    L = max(initial_L, t0)  # L must dominate t0; the network is empty otherwise
    while L <= max_L:
        if use_5_5_optimization:
            no_holds = solve_amapf_single_L(
                G, agents, other_agent_paths, t0, L,
                enforce_pickup_hold=False,
            )
            if no_holds is not None:
                if not detect_pickup_hold_violations(
                    no_holds.walkers, agents, t0
                ):
                    return no_holds
                # Pickup violation -> retry with the hold constraint.
                with_holds = solve_amapf_single_L(
                    G, agents, other_agent_paths, t0, L,
                    enforce_pickup_hold=True,
                )
                if with_holds is not None:
                    return with_holds
            # else: no-holds itself was infeasible. With-holds is strictly
            # more restrictive (fewer edges) so it would also be
            # infeasible at this L; skip the retry and bump L.
        else:
            result = solve_amapf_single_L(
                G, agents, other_agent_paths, t0, L,
                enforce_pickup_hold=True,
            )
            if result is not None:
                return result
        L += 1
    _LOG.warning(
        "solve_amapf: gave up after L reached %d for %d-agent subgroup at t0=%d",
        max_L,
        len(agents),
        t0,
    )
    return None
