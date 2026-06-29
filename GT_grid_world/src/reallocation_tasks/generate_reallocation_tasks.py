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

# Queue arrival is demand-driven, not time-driven, so a shuffle has no meaningful
# time deadline. We stamp this sentinel (matching ``get_deadline("none")``) so the
# deadline field stays structurally present but never expires a shuffle.
NO_DEADLINE = 9999999


def _lookahead_outbound_tasks(queue: np.ndarray, B: int, W: int) -> List[np.ndarray]:
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


def generate_reallocation_tasks(queue: np.ndarray, J: dict, G: Graph, Rs: AgentLoader, B: int, W: int, t: int) -> dict:
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
    flagged_real_tasks = _lookahead_outbound_tasks(queue, B, W)

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
