import numpy as np
import time

from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...reallocation_tasks.jr_consumer import shuffle_reference_location

# Task type codes mirror those in case_request_generator.py / simulate.py:
#   0 = outbound (warehouse -> driveway)
#   1 = inbound  (driveway  -> warehouse)
#   2 = shuffle  (warehouse -> warehouse) -- the M2M name for a rearrangement task
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1
TASK_TYPE_SHUFFLE = 2

# Task types whose pickup is a warehouse SKU instance. Used by simulate.py's
# `_refresh_tasks_after_warehouse_change` to know which start_locs need to
# be re-derived when warehouse occupancy changes. We *do not* use this set to
# decide which SKU-distribution matrix to populate any more (1.6 split shuffle
# out of the outbound matrix into its own rearrangement matrix), but the
# physical-pickup-from-warehouse semantics it captures are unchanged.
WAREHOUSE_PICKUP_TASK_TYPES = frozenset({TASK_TYPE_OUTBOUND, TASK_TYPE_SHUFFLE})

# Task types whose dropoff is a warehouse-empty cell. Same comment as above:
# kept for refresh-side bookkeeping; the cost matrices now branch by exact
# task type rather than by membership in this set.
WAREHOUSE_DROPOFF_TASK_TYPES = frozenset({TASK_TYPE_INBOUND, TASK_TYPE_SHUFFLE})


def manhattan_distance(loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two locations."""
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

def calculate_deadline_cost(deadline: int, current_time: int) -> int:
    """Deadline-based urgency cost.

    Piecewise linear (NOT quadratic, despite earlier docstring claims):

    - ``time_until_deadline > 60``      -> 0 (no urgency yet)
    - ``0 < time_until_deadline <= 60`` -> ``60 - time_until_deadline`` (linear ramp;
      reaches 60 right at the deadline)
    - ``time_until_deadline <= 0``      -> ``60 + |time_until_deadline|`` (continues
      linearly past the deadline)

    The 60-timestep window and the linear shape match Ethan's
    ``local_task_reallocation`` design; the math here is ported, not
    redesigned. Roadmap 1.6 wires this into the allocator's ``total_costs``
    via the ``deadline_weight`` knob; the proper non-linear / per-task-type
    tardiness shaping (if any) is left to a future iteration with Ethan.
    """
    time_until_deadline = deadline - current_time

    if time_until_deadline > 60:
        return 0
    elif time_until_deadline > 0:
        return int(60 - time_until_deadline)
    else:
        overdue_time = abs(time_until_deadline)
        return int(60 + overdue_time)


def per_task_type_sku_distribution_term(
    task_type: int,
    n: int,
    valid_p: np.ndarray,
    valid_q: np.ndarray,
    *,
    inbound_sku_distribution_costs: np.ndarray,
    outbound_sku_distribution_costs: np.ndarray,
    rearrangement_sku_distribution_costs: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Return the per-task-type SKU-distribution cost vector for task ``n``.

    This is the single source of truth for the per-task-type objective dispatch
    (roadmap section 1.6 / plan 3.4). Each task type pulls its placement-quality
    term from a different matrix:

    - ``TASK_TYPE_INBOUND``: use the inbound-style matrix indexed over goal cells.
      Returns shape ``(|valid_q|,)`` and axis tag ``"goals"``.
    - ``TASK_TYPE_OUTBOUND``: use the outbound-style matrix indexed over start
      cells. Returns shape ``(|valid_p|,)`` and axis tag ``"starts"``.
    - ``TASK_TYPE_SHUFFLE``: use the rearrangement matrix indexed over start
      cells. The rearrangement matrix is currently a placeholder mirroring the
      outbound formula because the proper rearrangement objective is TBD with
      Ethan; the dispatch is named explicitly so a future math change lands
      in *one* place. Returns shape ``(|valid_p|,)`` and axis tag ``"starts"``.

    The caller is responsible for shape-broadcasting the vector against its
    own ``base_costs`` tensor: ``"goals"`` broadcasts along the Q axis,
    ``"starts"`` along the P axis.

    Raises ``ValueError`` for unknown task types -- intentional: the
    "uniform penalization problem" the per-task-type architecture exists to
    solve (handoff section 13) means silently falling through to a default
    matrix would be a regression.
    """
    if task_type == TASK_TYPE_INBOUND:
        return inbound_sku_distribution_costs[n, valid_q], "goals"
    if task_type == TASK_TYPE_OUTBOUND:
        return outbound_sku_distribution_costs[n, valid_p], "starts"
    if task_type == TASK_TYPE_SHUFFLE:
        return rearrangement_sku_distribution_costs[n, valid_p], "starts"
    raise ValueError(
        f"Unknown task type {task_type!r} in per-task-type objective dispatch. "
        f"Expected one of: {{TASK_TYPE_INBOUND={TASK_TYPE_INBOUND}, "
        f"TASK_TYPE_OUTBOUND={TASK_TYPE_OUTBOUND}, "
        f"TASK_TYPE_SHUFFLE={TASK_TYPE_SHUFFLE}}}."
    )

# Default crM2M (concatenated rearrangement) objective hyperparameters.
# ``lambda`` weights the detour cost against the placement benefit in the
# rearrangement utility ``U = b - lambda * Delta`` (lambda >= 1). 1.5 matches
# Ethan's MILP-branch insertion experiments. The detour cutoff rejects any
# rearrangement whose *weighted* detour ``lambda * Delta`` reaches the cutoff
# (so e.g. Delta=5, lambda=2 is rejected even though Delta alone is < 10).
CRM2M_DEFAULT_LAMBDA = 1.5
CRM2M_DEFAULT_DETOUR_CUTOFF = 10.0
# Return margin (ticks) added to the effective delay in the utility penalty ONLY
# (not the slack cap, not the reject-gate). It requires a shuffle to finish with
# ``m`` ticks of spare buffer-time before its slack "pays off": a shuffle is only
# credited as free when ``slack >= Delta + c_pp + m``. This targets the observed
# failure mode where a shuffle finishes just as a buffer slot opens, but the slot
# is re-consumed by other agents' outbound arrivals before this agent can return
# to place -- so the move was not actually worthwhile. ``0.0`` reproduces the
# pre-margin behaviour exactly.
CRM2M_DEFAULT_RETURN_MARGIN = 0.0
# Floor on the net-clearing-rate denominator of the ambient slack formula
# (items/tick). Keeps slack finite as arrivals approach the drain rate; the
# real bound on slack is ``slack_cap`` (see ``compute_ambient_slack``), so this
# only needs to be a small positive number to avoid divide-by-zero / negatives.
CRM2M_DEFAULT_SLACK_EPS = 1e-3


def compute_ambient_slack(
    buffer_level: float,
    capacity: float,
    drain_per_tick: float,
    arrival_per_tick: float,
    lambda_: float,
    detour_cutoff: float,
    c_pp: float = 0.0,
    eps: float = CRM2M_DEFAULT_SLACK_EPS,
) -> float:
    """System-wide ambient slack scalar ``slack(t)`` for buffer-aware crM2M.

    ::

        slack(t) = min( slack_cap , max(0, B(t) - (K-1)) / max(mu - alpha, eps) )
        slack_cap = c_pp + detour_cutoff / lambda_

    - ``buffer_level`` (``B``), ``capacity`` (``K``): current / max output-buffer
      occupancy in items.
    - ``drain_per_tick`` (``mu``): buffer consumption rate, items/tick.
    - ``arrival_per_tick`` (``alpha``): estimated outbound arrival rate, items/tick
      (already converted from the queue's items/min).
    - ``c_pp``: the shuffle pick+place service time (one pick + one place).

    The numerator ``max(0, B - (K-1))`` is the *overshoot* -- how many items must
    drain before one free slot opens. The denominator ``mu - alpha`` is the *net*
    clearing rate (drain minus the refill from new outbound arrivals competing
    for slots), so overshoot / net-rate converts items into a predicted wait in
    ticks.

    ``slack_cap`` ceilings the creditable slack so a saturated buffer cannot
    excuse an unbounded free move. It is ``c_pp + detour_cutoff / lambda_``, NOT
    ``detour_cutoff / lambda_`` alone: the utility penalizes the *effective delay*
    ``Delta + c_pp``, so to let a fully-congested buffer credit a physically-allowed
    shuffle (raw detour ``Delta`` up to the gate ceiling ``detour_cutoff / lambda_``,
    plus the fixed pick+place ``c_pp``) the cap must cover both terms. Pinning the
    cap at ``detour_cutoff / lambda_`` alone -- when that is *smaller* than ``c_pp``
    -- means slack can never fully offset even a zero-detour shuffle's pick+place
    cost, so shuffles almost never clear the ``U>0`` gate (the collision that made
    rearrangements vanishingly rare). Decoupling the two restores the intended
    behavior: at max slack, the penalty on any gate-passing shuffle is
    ``lambda_ * max(0, (Delta + c_pp) - (c_pp + detour_cutoff/lambda_))
    = lambda_ * max(0, Delta - detour_cutoff/lambda_) = 0`` (since the gate already
    forces ``Delta < detour_cutoff / lambda_``), i.e. a saturated buffer judges a
    shuffle purely on its benefit.

    Returns ``0.0`` when the buffer is disabled (``capacity <= 0``) so callers can
    invoke it unconditionally; with ``slack == 0`` the utility reduces to the
    original ``b - lambda_ * Delta``.
    """
    if capacity <= 0:
        return 0.0
    overshoot = max(0.0, float(buffer_level) - (float(capacity) - 1.0))
    if overshoot == 0.0:
        return 0.0
    net_rate = max(float(drain_per_tick) - float(arrival_per_tick), float(eps))
    slack = overshoot / net_rate
    slack_cap = (
        float(c_pp) + detour_cutoff / lambda_ if lambda_ > 0 else float("inf")
    )
    return float(min(slack_cap, slack))


def _crm2m_distance(G: Graph, a: Tuple[int, int], b: Tuple[int, int], method: str) -> float:
    """Distance between two cells using the allocator's configured metric."""
    if method == "manhattan":
        return float(manhattan_distance(a, b))
    if method == "shortest_path":
        return float(G.get_distance(a, b))
    raise ValueError(f"Invalid cost calculation method: {method}")


def _crm2m_distances_to_point(
    G: Graph,
    locs: List[Tuple[int, int]],
    point: Tuple[int, int],
    method: str = "manhattan",
    indices: np.ndarray = None,
    locs_arr: np.ndarray = None,
) -> np.ndarray:
    """Distances from ``point`` to locations, vectorized for Manhattan.

    Returns a full-length ``(len(locs),)`` float array. When ``indices`` is
    given, only those entries are filled (others stay 0); callers must only
    read filled indices (e.g. a shuffle task's ``valid_p`` / ``valid_q``).

    Optional ``locs_arr`` is a prebuilt ``(N, 2)`` int array of ``locs`` so
    callers that hit many references avoid re-converting the tuple list.
    """
    n = len(locs)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    if indices is None:
        idx = np.arange(n, dtype=np.intp)
    else:
        idx = np.asarray(indices, dtype=np.intp)
        if idx.size == 0:
            return out

    if method == "manhattan":
        if locs_arr is None:
            locs_arr = np.asarray(locs, dtype=np.int32)
        pts = locs_arr[idx]
        dists = (
            np.abs(pts[:, 0] - point[0]) + np.abs(pts[:, 1] - point[1])
        ).astype(float)
    elif method == "shortest_path":
        dists = np.array(
            [float(G.get_distance(locs[i], point)) for i in idx], dtype=float
        )
    else:
        raise ValueError(f"Invalid cost calculation method: {method}")

    if indices is None:
        return dists
    out[idx] = dists
    return out


def compute_crm2m_terms(
    Rs: AgentLoader,
    G: Graph,
    start_locs: List[Tuple[int, int]],
    goal_locs: List[Tuple[int, int]],
    method: str = "manhattan",
    start_indices: np.ndarray = None,
    goal_indices: np.ndarray = None,
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Static terms for the crM2M rearrangement utility (plan section 3.4).

    The rearrangement utility is ``U(a_m, s_p, d_q) = b(s_p, d_q) - lambda * Delta(a_m, s_p)``
    where (all distances reference a single fixed dummy ``h_0 = Rs.agents[0].home``
    unless a per-task driveway ``reference_location`` overrides the benefit term):

    - ``Delta(a_m, s_p) = dist(g^m_{i-1}, s_p) + dist(s_p, h_0) - dist(a_m, h_0)`` --
      the marginal detour of inserting a pickup at ``s_p`` on the agent's way
      out of the aisle toward ``h_0``. ``dist(g^m_{i-1}, s_p)`` is supplied live
      by the allocator's ``agent_start_cost_tensor`` (it already equals the
      distance from the agent's previous-task goal / current anchor to ``s_p``,
      and is updated in place as the agent picks up more tasks in a batch).
    - ``b(s_p, d_q) = dist(s_p, h_0) - dist(d_q, h_0)`` by default -- how much
      closer to the aisle exit the item moved. When a shuffle carries a
      driveway ``reference_location`` (from ``generate_reallocation_tasks``),
      allocators instead use ``b = dist(s_p, ref) - dist(d_q, ref)`` matching
      irM2M / ``fast_optimal_insertion.benefit``.

    This function returns only the agent-independent / anchor-dependent pieces
    so the allocator never densifies the full ``(M, N, P, Q, K)`` tensor: the
    ``(M, P)`` detour and ``(P, Q)`` benefit are assembled on demand per task
    from these vectors, and the ``K`` coupling is the ``(P, Q)`` same-aisle mask.

    Optional ``start_indices`` / ``goal_indices`` restrict distance fills to a
    subset (e.g. locations that appear on any shuffle task). Unfilled entries
    stay 0 and must not be read.

    Returns:
        - ``home``: the dummy reference cell ``h_0``.
        - ``start_home``: ``(P,)`` distances ``dist(s_p, h_0)`` (detour + default benefit).
        - ``goal_home``: ``(Q,)`` distances ``dist(d_q, h_0)`` (default benefit).
        - ``agent_home``: ``(M,)`` distances ``dist(a_m, h_0)`` from each agent's
          current anchor (last task goal, else live state). Kept as its own
          vector -- deliberately NOT aliased to ``g^m_{i-1}`` -- so downstream
          changes to the agent anchor stay separable from the previous-goal term.
        - ``coupling_mask``: ``(P, Q)`` boolean, ``True`` where ``s_p`` and ``d_q``
          share an aisle (column). This reproduces the generator's per-aisle
          ``C_i`` coupling (``s_p in S^k_n`` and ``d_q in D^k_n``) as a single
          global mask, since coupling is "same column" for every shuffle task.
          Allocators that score a shuffle with a driveway ``reference_location``
          may replace this with an all-True mask (inter-aisle moves allowed).
    """
    home = Rs.agents[0].home

    start_arr = np.asarray(start_locs, dtype=np.int32) if start_locs else None
    goal_arr = np.asarray(goal_locs, dtype=np.int32) if goal_locs else None
    start_home = _crm2m_distances_to_point(
        G, start_locs, home, method, indices=start_indices, locs_arr=start_arr
    )
    goal_home = _crm2m_distances_to_point(
        G, goal_locs, home, method, indices=goal_indices, locs_arr=goal_arr
    )

    M = len(Rs.agents)
    agent_home = np.empty(M, dtype=float)
    for m, agent in enumerate(Rs.agents):
        if len(agent.task_sequence) == 0:
            anchor = agent.state
        else:
            anchor = agent.task_sequence[-1][2]
        agent_home[m] = _crm2m_distance(G, anchor, home, method)

    if start_arr is None or start_arr.size == 0:
        start_cols = np.empty(0, dtype=np.int32)
    else:
        start_cols = start_arr[:, 1]
    if goal_arr is None or goal_arr.size == 0:
        goal_cols = np.empty(0, dtype=np.int32)
    else:
        goal_cols = goal_arr[:, 1]
    if len(start_cols) == 0 or len(goal_cols) == 0:
        coupling_mask = np.zeros((len(start_locs), len(goal_locs)), dtype=bool)
    else:
        coupling_mask = start_cols[:, None] == goal_cols[None, :]

    return home, start_home, goal_home, agent_home, coupling_mask


def benefit_distances_to_reference(
    G: Graph,
    start_locs: List[Tuple[int, int]],
    goal_locs: List[Tuple[int, int]],
    reference_location: Tuple[int, int],
    method: str = "manhattan",
    start_indices: np.ndarray = None,
    goal_indices: np.ndarray = None,
    start_arr: np.ndarray = None,
    goal_arr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(P,)`` / ``(Q,)`` distances from each start/goal to a driveway reference.

    Used as the benefit term ``b(s, g) = dist(s, ref) - dist(g, ref)`` so crM2M
    matches ``optimal_insertion_gurobi.benefit`` / ``fast_optimal_insertion``.

    Optional ``start_indices`` / ``goal_indices`` fill only a subset (typically
    locations allowed by shuffle tasks); other entries stay 0. Optional
    ``start_arr`` / ``goal_arr`` avoid re-converting location lists.
    """
    start_ref = _crm2m_distances_to_point(
        G, start_locs, reference_location, method,
        indices=start_indices, locs_arr=start_arr,
    )
    goal_ref = _crm2m_distances_to_point(
        G, goal_locs, reference_location, method,
        indices=goal_indices, locs_arr=goal_arr,
    )
    return start_ref, goal_ref


def shuffle_location_indices(
    J: Dict,
    idx_to_task_id: Dict[int, int],
    task_start_mask: np.ndarray,
    task_goal_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Union of start/goal indices allowed by any shuffle task, plus shuffle ``n``s."""
    shuffle_ns = [
        n for n, task_id in idx_to_task_id.items()
        if J[task_id][4] == TASK_TYPE_SHUFFLE
    ]
    if not shuffle_ns:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, shuffle_ns
    shuffle_rows = np.asarray(shuffle_ns, dtype=np.intp)
    start_indices = np.flatnonzero(task_start_mask[shuffle_rows].any(axis=0))
    goal_indices = np.flatnonzero(task_goal_mask[shuffle_rows].any(axis=0))
    return start_indices, goal_indices, shuffle_ns


def precompute_shuffle_benefit_by_ref(
    G: Graph,
    J: Dict,
    idx_to_task_id: Dict[int, int],
    start_locs: List[Tuple[int, int]],
    goal_locs: List[Tuple[int, int]],
    shuffle_ns: List[int],
    start_indices: np.ndarray,
    goal_indices: np.ndarray,
    method: str = "manhattan",
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """One vectorized benefit distance pair per unique shuffle ``reference_location``.

    Restricts fills to ``start_indices`` / ``goal_indices`` (shuffle-allowed
    locations only), so M2M-only cells are skipped.
    """
    start_arr = np.asarray(start_locs, dtype=np.int32) if start_locs else None
    goal_arr = np.asarray(goal_locs, dtype=np.int32) if goal_locs else None
    benefit_by_ref: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for n in shuffle_ns:
        ref = shuffle_reference_location(J[idx_to_task_id[n]])
        if ref is None or ref in benefit_by_ref:
            continue
        benefit_by_ref[ref] = benefit_distances_to_reference(
            G, start_locs, goal_locs, ref, method,
            start_indices=start_indices, goal_indices=goal_indices,
            start_arr=start_arr, goal_arr=goal_arr,
        )
    return benefit_by_ref


def committed_shuffle_terms(
    G: Graph,
    prev_cell: Tuple[int, int],
    s_p: Tuple[int, int],
    d_q: Tuple[int, int],
    home: Tuple[int, int],
    slack: float = 0.0,
    c_pp: float = 0.0,
    lambda_: float = CRM2M_DEFAULT_LAMBDA,
    method: str = "manhattan",
    return_margin: float = 0.0,
    benefit_reference: Tuple[int, int] = None,
) -> Tuple[float, float, float]:
    """Reproduce the rearrangement objective's ``(benefit, detour, utility)`` for a
    single *committed* shuffle, evaluated from live post-allocation state.

    This mirrors :func:`rearrangement_cost_cube` exactly for one winning
    ``(agent, s_p, d_q)`` triple, so a completed shuffle can be logged with the
    same numbers the allocator scored it with (Ethan's request: record the
    benefit, the actual detour cost, and the actual utility of each completed
    rearrangement). ``prev_cell`` is the agent's anchor immediately before the
    shuffle in its task sequence (``g^m_{i-1}``, or the agent's state when the
    shuffle is the first queued task) -- the same anchor the allocator's
    ``agent_start_cost_tensor`` / ``agent_home`` encode:

        Delta   = dist(prev_cell, s_p) + dist(s_p, h0) - dist(prev_cell, h0)
        benefit = dist(s_p, ref) - dist(d_q, ref)
        U       = benefit - lambda_ * max(0, (Delta + c_pp + return_margin) - slack)

    ``ref`` defaults to ``home`` (``h_0``). Pass ``benefit_reference`` (the
    shuffle's driveway ``reference_location``) so benefit matches irM2M /
    ``fast_optimal_insertion`` while detour still uses ``home``.

    ``return_margin`` (>= 0) raises the free-move bar so the shuffle must finish
    with that many ticks of spare buffer-time; ``0.0`` reproduces the original.

    Returns ``(benefit, detour, utility)``.
    """
    ref = home if benefit_reference is None else benefit_reference
    d_prev_sp = _crm2m_distance(G, prev_cell, s_p, method)
    d_sp_h0 = _crm2m_distance(G, s_p, home, method)
    d_prev_h0 = _crm2m_distance(G, prev_cell, home, method)
    d_sp_ref = _crm2m_distance(G, s_p, ref, method)
    d_dq_ref = _crm2m_distance(G, d_q, ref, method)

    detour = d_prev_sp + d_sp_h0 - d_prev_h0
    benefit = d_sp_ref - d_dq_ref
    discounted_delay = max(0.0, (detour + c_pp + return_margin) - slack)
    utility = benefit - lambda_ * discounted_delay
    return float(benefit), float(detour), float(utility)


def rearrangement_U_by_start(
    agent_start_cost_tensor: np.ndarray,
    agent_home: np.ndarray,
    start_home: np.ndarray,
    goal_home: np.ndarray,
    coupling_mask: np.ndarray,
    valid_p: np.ndarray,
    valid_q: np.ndarray,
    lambda_: float,
    detour_cutoff: float,
    slack: float = 0.0,
    c_pp: float = 0.0,
    return_margin: float = 0.0,
    benefit_start: np.ndarray = None,
    benefit_goal: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gated per-(agent, start) utility using each start's best coupled goal.

    Detour / delay do not depend on the goal, so for fixed ``(m, p)`` the ``q``
    that maximizes ``U = b(p,q) - lambda * delay(m,p)`` is simply the coupled
    ``q`` with largest benefit. This collapses the ``(M, P, Q)`` search to an
    ``(M, P)`` matrix plus a ``(|valid_p|,)`` best-goal index vector.

    Returns:
        - ``U_mp``: ``(M, |valid_p|)`` utilities; invalid / gated entries are
          ``-inf``.
        - ``best_q_local``: ``(|valid_p|,)`` local goal indices into ``valid_q``.
    """
    vp = np.asarray(valid_p, dtype=np.intp)
    vq = np.asarray(valid_q, dtype=np.intp)
    n_p = int(vp.size)
    n_q = int(vq.size)
    M = int(agent_start_cost_tensor.shape[0])
    if n_p == 0 or n_q == 0:
        return np.full((M, n_p), -np.inf), np.zeros(n_p, dtype=np.intp)

    detour = (
        agent_start_cost_tensor[:, vp]
        + start_home[vp][None, :]
        - agent_home[:, None]
    )
    discounted_delay = np.maximum(0.0, detour + c_pp + return_margin - slack)
    ok_detour = (lambda_ * detour) < detour_cutoff

    b_start = start_home if benefit_start is None else benefit_start
    b_goal = goal_home if benefit_goal is None else benefit_goal
    benefit = b_start[vp][:, None] - b_goal[vq][None, :]
    coupling = coupling_mask[np.ix_(vp, vq)]
    if not np.all(coupling):
        benefit = np.where(coupling, benefit, -np.inf)

    best_q_local = np.argmax(benefit, axis=1)
    best_b = benefit[np.arange(n_p), best_q_local]

    U_mp = best_b[None, :] - lambda_ * discounted_delay
    U_mp = np.where(ok_detour & (best_b[None, :] > -np.inf) & (U_mp > 0), U_mp, -np.inf)
    return U_mp, best_q_local


def rearrangement_best_assignment(
    agent_start_cost_tensor: np.ndarray,
    agent_home: np.ndarray,
    start_home: np.ndarray,
    goal_home: np.ndarray,
    coupling_mask: np.ndarray,
    valid_p: np.ndarray,
    valid_q: np.ndarray,
    lambda_: float,
    detour_cutoff: float,
    slack: float = 0.0,
    c_pp: float = 0.0,
    return_margin: float = 0.0,
    benefit_start: np.ndarray = None,
    benefit_goal: np.ndarray = None,
    agent_allowed: np.ndarray = None,
) -> Tuple[float, Tuple[int, int, int] | None]:
    """Argmin of gated ``-U`` without materializing an ``(M, P, Q)`` cube.

    Returns ``(cost, (m, p_local, q_local))`` or ``(inf, None)`` when every
    candidate is gated. Optional ``agent_allowed`` (length ``M`` bool) forces
    disallowed agents to ``-inf`` utility (task-sequence limits).
    """
    U_mp, best_q_local = rearrangement_U_by_start(
        agent_start_cost_tensor, agent_home, start_home, goal_home,
        coupling_mask, valid_p, valid_q, lambda_, detour_cutoff,
        slack=slack, c_pp=c_pp, return_margin=return_margin,
        benefit_start=benefit_start, benefit_goal=benefit_goal,
    )
    if agent_allowed is not None:
        U_mp = np.where(np.asarray(agent_allowed, dtype=bool)[:, None], U_mp, -np.inf)

    max_u = float(np.max(U_mp)) if U_mp.size else -np.inf
    if not np.isfinite(max_u) or max_u == -np.inf:
        return float("inf"), None

    ms, ps = np.where(U_mp == max_u)
    pick = int(np.random.randint(ms.size)) if ms.size > 1 else 0
    m = int(ms[pick])
    p_local = int(ps[pick])
    q_local = int(best_q_local[p_local])
    return -max_u, (m, p_local, q_local)


def rearrangement_cost_cube(
    agent_start_cost_tensor: np.ndarray,
    agent_home: np.ndarray,
    start_home: np.ndarray,
    goal_home: np.ndarray,
    coupling_mask: np.ndarray,
    valid_p: np.ndarray,
    valid_q: np.ndarray,
    lambda_: float,
    detour_cutoff: float,
    slack: float = 0.0,
    c_pp: float = 0.0,
    return_margin: float = 0.0,
    benefit_start: np.ndarray = None,
    benefit_goal: np.ndarray = None,
) -> np.ndarray:
    """Per-task crM2M cost cube ``-U`` over ``(M, |valid_p|, |valid_q|)``.

    Allocators minimize cost, so the rearrangement cost is ``-U`` (maximizing
    utility). For a type=2 (shuffle) task this *replaces* the base + SKU cost
    entirely (per plan section 3.4 -- the rearrangement objective is a complete
    utility, not an additive placement term).

    Buffer-aware utility (ambient-slack model):

        effective_delay = Delta + c_pp + return_margin
        U = b - lambda_ * max(0, effective_delay - slack)

    ``return_margin`` (>= 0, default 0) is a safety headroom added to the penalty
    term ONLY -- not to ``slack_cap`` and not to the reject-gate. It requires a
    shuffle to finish with ``return_margin`` ticks of spare buffer-time before its
    slack fully "pays off" (free when ``slack >= Delta + c_pp + return_margin``),
    which shrinks the creditable-detour window from ``Delta <= detour_cutoff/lambda_``
    to ``Delta <= detour_cutoff/lambda_ - return_margin``. This suppresses shuffles
    that would finish just as a buffer slot opens only for competing outbound
    arrivals to re-consume it before this agent returns to place. ``0.0`` leaves the
    cube byte-for-byte identical to the pre-margin behaviour.

    where ``c_pp`` is the pick+place service time of the shuffle (a scalar in the
    same tick/cell currency as ``Delta``; one pick plus one place) and ``slack``
    is the single system-wide ambient slack scalar ``slack(t)`` -- the predicted
    idle time an outbound delivery would wait for a free output-buffer slot. Both
    default to ``0.0``, in which case ``U`` collapses exactly to the original
    ``b - lambda_ * Delta`` and this function is byte-for-byte unchanged.

    Slack only *softens the penalty*; it does not unlock longer physical moves.
    The hard reject-gate acts on the *raw travel detour* ``Delta`` alone -- not on
    the effective delay and not on the discounted overflow. ``c_pp`` is a fixed
    service cost, not a travel distance, so it is excluded from the physical-length
    ceiling (matching Ethan's insertion model, which caps the raw detour before
    adding the pick+place adjustment):

    Gating (entry set to ``+inf`` -> never chosen by ``argmin``):
        - ``lambda_ * Delta >= detour_cutoff`` (raw-detour cutoff -- a
          physical-sanity ceiling that ignores c_pp and slack),
        - ``U <= 0`` (no net benefit; rearrangement is opportunistic/droppable),
        - ``coupling_mask`` False (``s_p`` / ``d_q`` not in the same aisle group).

    The ``s_p in V_alloc`` / ``d_q in V_alloc`` / ``tau_n in T_alloc`` gates are
    already enforced upstream (allocated locations are removed from
    ``valid_p`` / ``valid_q`` via the task masks, and allocated tasks are
    excluded from the unallocated-task set), so they need no handling here.

    Optional ``benefit_start`` / ``benefit_goal`` override the benefit distances
    (default: ``start_home`` / ``goal_home``, i.e. distances to ``h_0``). Pass
    distances to a shuffle's driveway ``reference_location`` so benefit matches
    irM2M / ``fast_optimal_insertion``. Detour still uses ``start_home`` (``h_0``).

    Prefer :func:`rearrangement_best_assignment` / :func:`rearrangement_U_by_start`
    in hot allocator loops -- they avoid materializing this cube.
    """
    vp = np.asarray(valid_p, dtype=np.intp)
    vq = np.asarray(valid_q, dtype=np.intp)
    n_p = int(vp.size)
    n_q = int(vq.size)
    M = int(agent_start_cost_tensor.shape[0])
    cost = np.full((M, n_p, n_q), np.inf, dtype=float)
    if n_p == 0 or n_q == 0:
        return cost

    # Delta / delay: (M, |valid_p|) -- independent of goal.
    detour = (
        agent_start_cost_tensor[:, vp]
        + start_home[vp][None, :]
        - agent_home[:, None]
    )
    discounted_delay = np.maximum(0.0, detour + c_pp + return_margin - slack)
    ok_detour = (lambda_ * detour) < detour_cutoff  # (M, P)
    if not ok_detour.any():
        return cost

    b_start = start_home if benefit_start is None else benefit_start
    b_goal = goal_home if benefit_goal is None else benefit_goal
    benefit = b_start[vp][:, None] - b_goal[vq][None, :]  # (P, Q)
    coupling = coupling_mask[np.ix_(vp, vq)]
    if not coupling.any():
        return cost

    # Preprune starts with no coupled positive-benefit goal (U <= b always).
    ok_p = (coupling & (benefit > 0)).any(axis=1)  # (P,)
    ok_mp = ok_detour & ok_p[None, :]  # (M, P)
    if not ok_mp.any():
        return cost

    # Single broadcast of U; write -U only into surviving entries (no np.where copies).
    U = benefit[None, :, :] - (lambda_ * discounted_delay)[:, :, None]
    valid = ok_mp[:, :, None] & coupling[None, :, :] & (U > 0)
    cost[valid] = -U[valid]
    return cost


def construct_cost_elements(J: Dict[int, Tuple], Rs: AgentLoader, G: Graph, current_time: int, method : str = "manhattan") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Compute the cost elements needed for allocation.
    Args:
        - J: Dict[task_id, (start_loc, goal_loc, deadline, sku_id, inbound)]
        - Rs: AgentLoader
        - G: Graph
        - current_time: Current timestep for deadline calculations
        - method: str
    Returns:
        - agent_start_cost_tensor: (M, P)
        - start_goal_dist: (P, Q)
        - task_start_mask: (N, P)
        - task_goal_mask: (N, Q)
        - start_locs: List[Tuple[int, int]]
        - goal_locs: List[Tuple[int, int]]
    """
    allocated_task_ids = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_task_ids.add(task[0])

    unallocated_task_ids = [task_id for task_id in J.keys() if task_id not in allocated_task_ids]

    M = len(Rs.agents)  # Number of agents
    N = len(unallocated_task_ids) # Number of tasks

    # Get all possible start and goal locations from unallocated tasks
    all_start_locs = set()
    all_goal_locs = set()
    for task_id in unallocated_task_ids:
        all_start_locs.update(J[task_id][0])  # start_locations_frozenset
        all_goal_locs.update(J[task_id][1])  # goal_locations_frozenset

    # idx to task_id mapping
    idx_to_task_id = {idx: task_id for idx, task_id in enumerate(unallocated_task_ids)}
    task_id_to_idx = {task_id: idx for idx, task_id in enumerate(unallocated_task_ids)}
    
    # Convert to sorted lists for consistent indexing
    start_locs = sorted(list(all_start_locs))
    goal_locs = sorted(list(all_goal_locs))
    P = len(start_locs)
    Q = len(goal_locs)

    # print(f"Number of start locations: {P}")
    # print(f"Number of goal locations: {Q}")
    # print(f"Number of tasks: {N}")
    # print(f"Number of agents: {M}")
    
    # Create location to index mappings
    start_loc_to_idx = {loc: idx for idx, loc in enumerate(start_locs)}
    goal_loc_to_idx = {loc: idx for idx, loc in enumerate(goal_locs)}

    # Get all locations currently allocated to tasks
    allocated_locs = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_locs.add(task[1])
            allocated_locs.add(task[2])

    # Get all locations occupied by items in warehouse and driveway
    warehouse_occupied_locs = set(G.warehouse.get_full_locations())
    driveway_occupied_locs = set(G.driveway.get_full_locations())

    # Combine all unusable locations
    unusable_locs = allocated_locs | warehouse_occupied_locs | driveway_occupied_locs

    # 1. Build (P, Q) distance matrix between all start and goal locations
    if method == "manhattan":
        start_goal_dist = np.array([[manhattan_distance(s, g) for g in goal_locs] for s in start_locs])
    elif method == "shortest_path":
        start_goal_dist = np.array([[G.get_distance(s, g) for g in goal_locs] for s in start_locs])
    else:
        raise ValueError(f"Invalid cost calculation method: {method}")
    
    # print(f"Start Goal Distance Matrix: {start_goal_dist} with shape {start_goal_dist.shape}")
    # exit()

    # 2. Build (N, P) task-start membership matrix (1 if task n has start_loc p and p not unusable, else 0)
    task_start_mask = np.zeros((N, P), dtype=np.float32)
    for n, task_id in enumerate(unallocated_task_ids):
        for s in J[task_id][0]:
            if s not in allocated_locs:
                i = start_loc_to_idx[s]
                task_start_mask[n, i] = 1.0

    # 3. Build (N, Q) task-goal membership matrix (1 if task n has goal_loc q and q not unusable, else 0)
    task_goal_mask = np.zeros((N, Q), dtype=np.float32)
    for n, task_id in enumerate(unallocated_task_ids):
        for g in J[task_id][1]:
            if g not in unusable_locs:
                j = goal_loc_to_idx[g]
                task_goal_mask[n, j] = 1.0

    # 4. Build (M, P) agent-start cost matrix with deadline urgency costs
    agent_start_cost_tensor = np.full((M, P), np.inf)
    for m in range(M):
        # Determine agent's current position
        if len(Rs.agents[m].task_sequence) == 0:
            agent_pos = Rs.agents[m].state
        else:
            agent_pos = Rs.agents[m].task_sequence[-1][2]  # goal location of most recent task
        
        for i, s in enumerate(start_locs):
            # Base cost (negative for argmax logic)
            if method == "manhattan":
                agent_start_cost_tensor[m, i] = manhattan_distance(agent_pos, s)
            elif method == "shortest_path":
                agent_start_cost_tensor[m, i] = G.get_distance(agent_pos, s)
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            
            # Find tasks that can use this start location and calculate deadline urgency
            # deadline_urgency_cost = 0
            # for n, task in enumerate(unallocated_tasks):
            #     if s in task[1] and task_start_mask[n, i] == 1.0:
            #         # This task can use this start location, add its deadline urgency
            #         task_deadline = task[3]
            #         urgency_cost = calculate_deadline_cost(task_deadline, current_time)
            #         deadline_urgency_cost = max(deadline_urgency_cost, urgency_cost)
            
            # Combine base cost with deadline urgency (positive urgency cost increases the negative base cost)
            # agent_start_cost_tensor[m, i] = 0.3*base_cost + 0.7*deadline_urgency_cost
            
    # print(f"Agent Start Cost Matrix: {agent_start_cost_tensor} with shape {agent_start_cost_tensor.shape}")
    
    # 5. Build (N) vector of task deadline costs
    task_deadline_costs = np.zeros(N)
    for n, task_id in enumerate(unallocated_task_ids):
        task_deadline_costs[n] = calculate_deadline_cost(J[task_id][2], current_time)

    # 6. Build (N, Q) inbound-style SKU-distribution cost matrix.
    # Populated for inbound tasks ONLY. Defaults to np.inf elsewhere so the
    # per-task-type dispatch in `per_task_type_sku_distribution_term` plus
    # the allocator's argmin will skip invalid combos. (1.6 split: shuffle
    # used to be populated here too in 1.4 as a skeleton; it now has its
    # own rearrangement matrix below.)
    inbound_sku_distribution_costs = np.full((N, Q), np.inf)
    for n, task_id in enumerate(unallocated_task_ids):
        if J[task_id][4] != TASK_TYPE_INBOUND:
            continue
        for q in J[task_id][1]:
            if q in unusable_locs:
                continue
            j = goal_loc_to_idx[q]
            inbound_sku_distribution_costs[n, j] = -1 * G.query_sku_KD_trees(J[task_id][3], goal_locs[j], 1)[0]

    # 7. Build (N, P) outbound-style SKU-distribution cost matrix.
    # Populated for outbound tasks ONLY (1.6 split, see above).
    outbound_sku_distribution_costs = np.full((N, P), np.inf)
    for n, task_id in enumerate(unallocated_task_ids):
        if J[task_id][4] != TASK_TYPE_OUTBOUND:
            continue
        for p in J[task_id][0]:
            if p in allocated_locs:
                continue
            i = start_loc_to_idx[p]
            # Second-closest because the closest is the start cell itself.
            outbound_sku_distribution_costs[n, i] = G.query_sku_KD_trees(J[task_id][3], start_locs[i], 2)[0][1]

    # 7b. Build (N, P) rearrangement SKU-distribution cost matrix (type=2).
    # PLACEHOLDER for the 1.6 deliverable: this matrix exists so the
    # per-task-type dispatch has a dedicated branch for shuffle, but the
    # math currently mirrors the outbound formula (second-nearest same-SKU
    # neighbour distance) because the proper rearrangement objective is TBD
    # with Ethan (handoff sections 2 & 13, plan section 3.4 -- "the
    # rearrangement objective function math is TBD with Ethan; the
    # infrastructure (per-type if-clauses, cost tensor construction) can be
    # built now"). When Ethan's math lands, only the BODY of this loop
    # changes; every consumer already routes type=2 through this matrix via
    # `per_task_type_sku_distribution_term`.
    rearrangement_sku_distribution_costs = np.full((N, P), np.inf)
    for n, task_id in enumerate(unallocated_task_ids):
        if J[task_id][4] != TASK_TYPE_SHUFFLE:
            continue
        for p in J[task_id][0]:
            if p in allocated_locs:
                continue
            i = start_loc_to_idx[p]
            rearrangement_sku_distribution_costs[n, i] = G.query_sku_KD_trees(J[task_id][3], start_locs[i], 2)[0][1]

    # 8. Build vector of size (M) which includes the estimated time for the agent
    # to complete its already-allocated task sequence. As of 1.6 this is added
    # to base_costs so already-busy agents look more expensive than idle ones
    # (plan 3.4: execution time accounting). The vector is updated in-place by
    # `fast_greedy_allocation` after each batch assignment.
    agent_task_sequence_time = np.zeros(M)
    for m in range(M):
        if len(Rs.agents[m].task_sequence) == 0:
            continue
        agent_task_sequence_time[m] = G.get_distance(Rs.agents[m].state, Rs.agents[m].task_sequence[0][1])
        for i in range(1, len(Rs.agents[m].task_sequence)):
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].task_sequence[i-1][2], Rs.agents[m].task_sequence[i][1])
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].task_sequence[i][1], Rs.agents[m].task_sequence[i][2])

    return (agent_start_cost_tensor, start_goal_dist, task_start_mask, task_goal_mask,
            start_locs, goal_locs, idx_to_task_id, task_id_to_idx, task_deadline_costs,
            inbound_sku_distribution_costs, outbound_sku_distribution_costs,
            rearrangement_sku_distribution_costs, agent_task_sequence_time)
