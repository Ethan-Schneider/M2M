import random
import numpy as np
from typing import List, Tuple

from ..agent import AgentLoader
from ..graph import Graph

# queue_generation writes 0-indexed SKU ids; GT warehouse SKUs are 1-indexed.
QUEUE_SKU_ID_OFFSET = 1
QUEUE_SKU_ID = 0
QUEUE_TASK_TYPE = 1
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_SHUFFLE = 2

# crM2M queue look-ahead is demand-driven (position, not time), so a crM2M
# shuffle has no meaningful time deadline. We stamp this sentinel (matching
# ``get_deadline("none")``) so the deadline field stays structurally present
# but never expires a shuffle.
NO_DEADLINE = 9999999


# ---------------------------------------------------------------------------
# irM2M (Insertion Rearrangement) reallocation-task generation.
#
# Produces the 5-tuple candidate set consumed by the optimal-insertion MILP
# (``optimal_insertion_gurobi`` / ``fast_optimal_insertion``): each candidate
# carries a driveway-column ``reference_location`` used as the benefit
# reference point. ``merge_reallocation_tasks_into_J`` folds a chosen set into
# ``J`` for simultaneous allocation.
# ---------------------------------------------------------------------------


def merge_reallocation_tasks_into_J(
    tasks_a: dict,
    J: dict,
    G: Graph,
    last_task_id: int,
) -> int:
    """Add candidate rearrangement tasks to ``J`` for simultaneous (crM2M) allocation."""
    for _task_key, (C_i, _release, deadline, task_type, _aisle_reference) in tasks_a.items():
        if not C_i:
            continue
        starts = frozenset(start for start, _goal in C_i)
        goals = frozenset(goal for _start, goal in C_i)
        sample_start = next(iter(starts))
        sku = G.warehouse.get_sku_at_location(sample_start)
        sku_id = int(sku.sku_id) if sku is not None else 0
        last_task_id += 1
        J[last_task_id] = (starts, goals, int(deadline), sku_id, int(task_type))
    return last_task_id


def _lookahead_outbound_tasks(queue: np.ndarray, B: int, W: int) -> List[Tuple[np.ndarray, int]]:
    """Return outbound queue rows in ``queue[B:W]`` with their 0-based queue indices."""
    if queue.size == 0 or B >= W:
        return []

    window = np.atleast_2d(queue[B:W])
    flagged: List[Tuple[np.ndarray, int]] = []
    for offset, task in enumerate(window):
        if int(task[QUEUE_TASK_TYPE]) == TASK_TYPE_OUTBOUND:
            flagged.append((task, B + offset))
    return flagged


def generate_reallocation_tasks(
    queue: np.ndarray,
    J: dict,
    G: Graph,
    Rs: AgentLoader,
    B: int,
    W: int,
    t: int,
) -> dict:
    """
    Look at outbound tasks in the queue window ``queue[B:W]`` and build shuffle
    reallocation candidates for each.

    ``B`` and ``W`` are 0-based queue indices into the remaining (unpopped)
    queue. For example, ``W=100`` inspects the next 100 tasks ``queue[0:100]``;
    ``B=30, W=100`` inspects the 70 tasks ``queue[30:100]``.

    Args:
        queue: Remaining task queue with rows ``(sku_id, task_type)``.
        J: The dictionary of tasks.
        G: The graph of the warehouse.
        Rs: The dictionary of agents.
        B: Start index (inclusive) of the lookahead window in the queue.
        W: End index (exclusive) of the lookahead window in the queue.
        t: Current simulation timestep.

    Returns:
        A dictionary of reallocation tasks.

        Reallocation task is defined as: tau=(C_i, r_i, d_i, sigma_i, ref_i), where
        C_i = {(S_i^k, D_i^k)}_{k=1}^{|C_i|} is the set of (start_loc, end_loc) pairs for task i.
        r_i, d_i are the release time and deadline of the reallocation task
        sigma_i is the task type, 0 for outbound, 1 for inbound, 2 for shuffle.
        ref_i is a representative driveway vertex in the aisle that the
        corresponding real outbound task has been pre-committed to, used as
        the benefit reference point instead of each candidate goal's own
        column (see ``optimal_insertion_gurobi.benefit``).
    """
    flagged_real_tasks = _lookahead_outbound_tasks(queue, B, W)

    reallocation_tasks = {}
    if not flagged_real_tasks:
        return reallocation_tasks

    # Aisles (driveway columns) currently available for an outbound task's
    # eventual delivery. Empty for every flagged task if the driveway is full.
    driveway_columns = sorted({loc[1] for loc in G.driveway.get_empty_locations()})
    if not driveway_columns:
        return reallocation_tasks

    dist_matrix = G.get_distance_matrix()
    empty_locations = list(G.warehouse.get_empty_locations())
    empty_idx = np.array([G.location_index(loc) for loc in empty_locations], dtype=int)

    for reallocation_task_id, (task, queue_index) in enumerate(flagged_real_tasks):
        sku_id = int(task[QUEUE_SKU_ID]) + QUEUE_SKU_ID_OFFSET
        sku_locations = list(G.warehouse.get_sku_instances(sku_id))
        if not sku_locations or empty_idx.size == 0:
            continue

        # Pre-commit this (not-yet-released) outbound task to one driveway
        # aisle, the same way import_schedule._add_outbound_task will when it
        # is actually released, and use any cell in that aisle as "the real
        # goal location" -- driveway cells in the same column are close
        # enough to each other that it doesn't matter which one. Candidate
        # goals span the whole warehouse (not just the SKU's own aisle) so
        # inter-aisle rearrangements are possible, but are pruned to those
        # that are actually closer to that reference than the SKU already
        # is -- a move that isn't is never chosen by the insertion MILP
        # anyway (see optimal_insertion_gurobi.solve_insertion), so skipping
        # it here keeps the candidate set from blowing up now that every
        # empty warehouse location is in play.
        chosen_column = random.choice(driveway_columns)
        reference_location = G.get_driveway_column_reference(chosen_column)
        dist_to_reference = dist_matrix[G.location_index(reference_location)]

        C_i = set()
        for start_loc in sku_locations:
            d_start = dist_to_reference[G.location_index(start_loc)]
            closer = dist_to_reference[empty_idx] < d_start
            for goal_loc, is_closer in zip(empty_locations, closer):
                if is_closer and goal_loc != start_loc:
                    C_i.add((start_loc, goal_loc))

        if not C_i:
            continue

        r_i = t
        # Shuffle should finish before this outbound reaches the front of the queue.
        d_i = t + queue_index + 1
        sigma_i = TASK_TYPE_SHUFFLE
        reallocation_tasks[reallocation_task_id] = (C_i, r_i, d_i, sigma_i, reference_location)

    return reallocation_tasks


# ---------------------------------------------------------------------------
# crM2M (Concatenated Rearrangement) reallocation-task generation.
#
# Produces the 4-tuple candidate set consumed by ``jr_consumer`` (J_a
# lifecycle) and, ultimately, the LNS allocator's rearrangement cost cube.
# Candidates are strictly same-aisle pre-staging moves with no time deadline
# (the queue look-ahead only decides *which* SKUs to pre-stage, by position).
# Kept byte-identical to the symbotic_2026 implementation so crM2M behaviour
# is unchanged by the task_queue merge; only the function names are suffixed
# to coexist with the irM2M generator above.
# ---------------------------------------------------------------------------


def _lookahead_outbound_tasks_crm2m(queue: np.ndarray, B: int, W: int) -> List[np.ndarray]:
    """Return the outbound rows in the index window ``queue[B:W]``.

    The queue has no release times, so the look-ahead is over *positions* in the
    remaining (unpopped) queue rather than a time window. ``B``/``W`` are 0-based
    indices: ``B=30, W=100`` inspects the 70 rows ``queue[30:100]``. Position only
    selects *which* upcoming SKUs to pre-stage; it does not impose a deadline.
    """
    if queue.size == 0 or B >= W:
        return []

    window = np.atleast_2d(queue[B:W])
    return [task for task in window if int(task[QUEUE_TASK_TYPE]) == TASK_TYPE_OUTBOUND]


def generate_crm2m_reallocation_tasks(queue: np.ndarray, J: dict, G: Graph, Rs: AgentLoader, B: int, W: int, t: int) -> dict:
    """Build shuffle candidates for each outbound task in the queue look-ahead.

    For every outbound task in the index window ``queue[B:W]`` we build the
    coupled set ``C_i`` of same-aisle ``(sku_instance_cell, empty_cell)`` pairs:
    the move that pre-positions that SKU toward the aisle exit before its demand
    arrives. Per-aisle coupling (``s_p`` and ``d_q`` in the same column) is
    encoded by only pairing cells that share a column.

    Args:
        queue: Remaining task queue with rows ``(sku_id, task_type)``.
        J: The dictionary of live (real) tasks.
        G: The warehouse graph.
        Rs: The agent loader.
        B: Start index (inclusive) of the look-ahead window in the queue.
        W: End index (exclusive) of the look-ahead window in the queue.
        t: Current simulation timestep.

    Returns:
        ``{reallocation_task_id: (C_i, r_i, d_i, sigma_i)}`` where
        ``C_i = {(start_loc, goal_loc)}`` are the coupled same-aisle moves,
        ``r_i`` the release time, ``d_i`` the deadline (always ``NO_DEADLINE`` in
        the queue model -- shuffles are not time-bounded), and ``sigma_i`` the
        task type (2 = shuffle).
    """
    flagged_real_tasks = _lookahead_outbound_tasks_crm2m(queue, B, W)

    reallocation_tasks = {}
    aisle_columns = sorted({loc[1] for loc in G.get_aisle_locations()})

    for reallocation_task_id, task in enumerate(flagged_real_tasks):
        sku_id = int(task[QUEUE_SKU_ID]) + QUEUE_SKU_ID_OFFSET
        sku_locations = set(G.warehouse.get_sku_instances(sku_id))
        empty_locations = set(G.warehouse.get_empty_locations())

        C_i = set()
        for column in aisle_columns:
            aisle_locations = [
                loc for loc in G.get_aisle_locations() if loc[1] == column
            ]
            start_locations = [
                loc for loc in aisle_locations if loc in sku_locations
            ]
            goal_locations = [
                loc for loc in aisle_locations if loc in empty_locations
            ]
            if not start_locations or not goal_locations:
                continue
            for start_loc in start_locations:
                for goal_loc in goal_locations:
                    C_i.add((start_loc, goal_loc))

        if not C_i:
            continue

        r_i = t
        # No time deadline: the queue look-ahead only decides *which* SKUs to
        # pre-stage (by position), not *when* the move must finish.
        d_i = NO_DEADLINE
        sigma_i = TASK_TYPE_SHUFFLE
        reallocation_tasks[reallocation_task_id] = (C_i, r_i, d_i, sigma_i)

    return reallocation_tasks
