import numpy as np

from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader

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


def _crm2m_distance(G: Graph, a: Tuple[int, int], b: Tuple[int, int], method: str) -> float:
    """Distance between two cells using the allocator's configured metric."""
    if method == "manhattan":
        return float(manhattan_distance(a, b))
    if method == "shortest_path":
        return float(G.get_distance(a, b))
    raise ValueError(f"Invalid cost calculation method: {method}")


def compute_crm2m_terms(
    Rs: AgentLoader,
    G: Graph,
    start_locs: List[Tuple[int, int]],
    goal_locs: List[Tuple[int, int]],
    method: str = "manhattan",
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Static terms for the crM2M rearrangement utility (plan section 3.4).

    The rearrangement utility is ``U(a_m, s_p, d_q) = b(s_p, d_q) - lambda * Delta(a_m, s_p)``
    where (all distances reference a single fixed dummy ``h_0 = Rs.agents[0].home``):

    - ``Delta(a_m, s_p) = dist(g^m_{i-1}, s_p) + dist(s_p, h_0) - dist(a_m, h_0)`` --
      the marginal detour of inserting a pickup at ``s_p`` on the agent's way
      out of the aisle toward ``h_0``. ``dist(g^m_{i-1}, s_p)`` is supplied live
      by the allocator's ``agent_start_cost_tensor`` (it already equals the
      distance from the agent's previous-task goal / current anchor to ``s_p``,
      and is updated in place as the agent picks up more tasks in a batch).
    - ``b(s_p, d_q) = dist(s_p, h_0) - dist(d_q, h_0)`` -- how much closer to the
      aisle exit the item moved (agent-independent placement quality).

    This function returns only the agent-independent / anchor-dependent pieces
    so the allocator never densifies the full ``(M, N, P, Q, K)`` tensor: the
    ``(M, P)`` detour and ``(P, Q)`` benefit are assembled on demand per task
    from these vectors, and the ``K`` coupling is the ``(P, Q)`` same-aisle mask.

    Returns:
        - ``home``: the dummy reference cell ``h_0``.
        - ``start_home``: ``(P,)`` distances ``dist(s_p, h_0)``.
        - ``goal_home``: ``(Q,)`` distances ``dist(d_q, h_0)``.
        - ``agent_home``: ``(M,)`` distances ``dist(a_m, h_0)`` from each agent's
          current anchor (last task goal, else live state). Kept as its own
          vector -- deliberately NOT aliased to ``g^m_{i-1}`` -- so downstream
          changes to the agent anchor stay separable from the previous-goal term.
        - ``coupling_mask``: ``(P, Q)`` boolean, ``True`` where ``s_p`` and ``d_q``
          share an aisle (column). This reproduces the generator's per-aisle
          ``C_i`` coupling (``s_p in S^k_n`` and ``d_q in D^k_n``) as a single
          global mask, since coupling is "same column" for every shuffle task.
    """
    home = Rs.agents[0].home

    start_home = np.array(
        [_crm2m_distance(G, s, home, method) for s in start_locs], dtype=float
    )
    goal_home = np.array(
        [_crm2m_distance(G, g, home, method) for g in goal_locs], dtype=float
    )

    M = len(Rs.agents)
    agent_home = np.empty(M, dtype=float)
    for m, agent in enumerate(Rs.agents):
        if len(agent.task_sequence) == 0:
            anchor = agent.state
        else:
            anchor = agent.task_sequence[-1][2]
        agent_home[m] = _crm2m_distance(G, anchor, home, method)

    start_cols = np.array([s[1] for s in start_locs])
    goal_cols = np.array([g[1] for g in goal_locs])
    if len(start_cols) == 0 or len(goal_cols) == 0:
        coupling_mask = np.zeros((len(start_locs), len(goal_locs)), dtype=bool)
    else:
        coupling_mask = start_cols[:, None] == goal_cols[None, :]

    return home, start_home, goal_home, agent_home, coupling_mask


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
) -> np.ndarray:
    """Per-task crM2M cost cube ``-U`` over ``(M, |valid_p|, |valid_q|)``.

    Allocators minimize cost, so the rearrangement cost is ``-U`` (maximizing
    utility). For a type=2 (shuffle) task this *replaces* the base + SKU cost
    entirely (per plan section 3.4 -- the rearrangement objective is a complete
    utility, not an additive placement term).

    Gating (entry set to ``+inf`` -> never chosen by ``argmin``):
        - ``lambda_ * Delta >= detour_cutoff`` (weighted-detour cutoff),
        - ``U <= 0`` (no net benefit; rearrangement is opportunistic/droppable),
        - ``coupling_mask`` False (``s_p`` / ``d_q`` not in the same aisle group).

    The ``s_p in V_alloc`` / ``d_q in V_alloc`` / ``tau_n in T_alloc`` gates are
    already enforced upstream (allocated locations are removed from
    ``valid_p`` / ``valid_q`` via the task masks, and allocated tasks are
    excluded from the unallocated-task set), so they need no handling here.
    """
    # Delta: (M, |valid_p|). dist(g^m_{i-1}, s_p) is the live agent_start cost.
    detour = (
        agent_start_cost_tensor[:, valid_p]
        + start_home[valid_p][None, :]
        - agent_home[:, None]
    )
    # b: (|valid_p|, |valid_q|).
    benefit = start_home[valid_p][:, None] - goal_home[valid_q][None, :]
    # U: (M, |valid_p|, |valid_q|).
    U = benefit[None, :, :] - lambda_ * detour[:, :, None]
    cost = -U

    weighted_detour = lambda_ * detour  # (M, |valid_p|)
    cost = np.where(weighted_detour[:, :, None] >= detour_cutoff, np.inf, cost)
    cost = np.where(U <= 0, np.inf, cost)
    coupling = coupling_mask[np.ix_(valid_p, valid_q)]  # (|valid_p|, |valid_q|)
    cost = np.where(coupling[None, :, :], cost, np.inf)
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
