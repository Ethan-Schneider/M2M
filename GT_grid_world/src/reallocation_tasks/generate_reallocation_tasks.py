import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from munkres import Munkres

from ..agent import AgentLoader
from ..graph import Graph

# queue_generation writes 0-indexed SKU ids; GT warehouse SKUs are 1-indexed.
QUEUE_SKU_ID_OFFSET = 1
QUEUE_SKU_ID = 0
QUEUE_TASK_TYPE = 1
QUEUE_DEADLINE = 2
QUEUE_DRIVEWAY_COLUMN = 3
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_SHUFFLE = 2

Location = Tuple[int, int]

# (task_id, deadline, reference_location); task_id is ``None`` for a
# not-yet-released (future, tentative) queue lookahead task.
OutboundTaskRef = Tuple[Optional[int], int, Location]


def merge_reallocation_tasks_into_J(
    tasks_a: dict,
    J: dict,
    G: Graph,
    last_task_id: int,
) -> int:
    """Add candidate rearrangement tasks to ``J`` for simultaneous (crM2M) allocation."""
    for _task_key, (C_i, _release, deadline, task_type, _aisle_reference, _target_task_id) in tasks_a.items():
        if not C_i:
            continue
        starts = frozenset(start for start, _goal in C_i)
        goals = frozenset(goal for _start, goal in C_i)
        sample_start = next(iter(starts))
        sku = G.warehouse.get_sku_at_location(sample_start)
        sku_id = int(sku.sku_id) if sku is not None else 0
        last_task_id += 1
        if reference_location is not None:
            J[last_task_id] = (
                starts,
                goals,
                int(deadline),
                sku_id,
                int(task_type),
                reference_location,
            )
        else:
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


def _future_outbound_tasks_by_sku(
    queue: np.ndarray, B: int, W: int
) -> Dict[int, List[Tuple[np.ndarray, int]]]:
    """Group not-yet-released outbound queue rows in ``queue[B:W]`` by SKU id, in queue order."""
    by_sku: Dict[int, List[Tuple[np.ndarray, int]]] = defaultdict(list)
    for task, queue_index in _lookahead_outbound_tasks(queue, B, W):
        sku_id = int(task[QUEUE_SKU_ID]) + QUEUE_SKU_ID_OFFSET
        by_sku[sku_id].append((task, queue_index))
    return by_sku


def _driveway_reference_for_goals(goals: frozenset, G: Graph) -> Optional[Location]:
    """Representative driveway reference point for a task's (single-column) goal set."""
    if not goals:
        return None
    return G.get_driveway_column_reference(next(iter(goals))[1])


def _reconcile_locks(
    item_task_locks: Dict[Location, int],
    pending_task_targets: Dict[int, Location],
    J: dict,
    G: Graph,
) -> None:
    """Reconcile item->task locks against physical warehouse state.

    A lock (``item_task_locks[loc] == task_id``) records that the item
    physically sitting at ``loc`` right now is committed to fulfilling
    outbound task ``task_id`` and must not be proposed as a rearrangement
    candidate for any other task.

    ``pending_task_targets[task_id] == goal`` records that a *committed but
    not yet executed* shuffle is carrying the locked item for ``task_id`` to
    ``goal``. It exists because ``item_task_locks`` can only describe an
    item's *current* physical location, and there's a window -- from the
    moment the shuffle is inserted into an agent's task_sequence to the
    moment it's actually picked up and placed -- where the item isn't at its
    old location, its new location, or anywhere in the warehouse at all
    (it's on the agent). Tracking the destination separately lets us follow
    the *same physical item instance* across that gap instead of losing the
    lock and letting a fresh Munkres round re-pick (possibly) a different
    instance for the task.

    Three passes, each over current state, so ordering within a tick doesn't
    matter:
      1. A confirmed lock whose task has completed (left ``J``) is released;
         if its item is still sitting there it wasn't the one used to fulfill
         the task, so it becomes free again. Any pending target for that task
         is dropped too -- moot now.
      2. A pending target whose task completed without ever arriving (should
         not normally happen, but keeps state from leaking) is dropped.
      3. A pending target whose item has now physically arrived is promoted
         to a confirmed lock at its new location.

    A confirmed lock whose item has physically left ``loc`` but has *no*
    matching pending target (e.g. it was picked up directly for the real
    outbound task rather than shuffled) is released -- there's nothing to
    track it to.
    """
    for loc in list(item_task_locks):
        task_id = item_task_locks[loc]
        if task_id not in J:
            del item_task_locks[loc]
            pending_task_targets.pop(task_id, None)
        elif G.warehouse.get_sku_at_location(loc) is None:
            del item_task_locks[loc]

    for task_id in list(pending_task_targets):
        if task_id not in J:
            del pending_task_targets[task_id]

    for task_id, goal in list(pending_task_targets.items()):
        if G.warehouse.get_sku_at_location(goal) is not None:
            item_task_locks[goal] = task_id
            del pending_task_targets[task_id]


def _released_outbound_tasks_for_sku(
    sku_id: int, J: dict, G: Graph, exclude_task_ids: set
) -> List[OutboundTaskRef]:
    """(task_id, deadline, driveway reference location) of every released
    outbound task in ``J`` for ``sku_id``, excluding tasks already locked to
    another item."""
    tasks: List[OutboundTaskRef] = []
    for task_id, (_starts, goals, deadline, entry_sku_id, task_type) in J.items():
        if task_type != TASK_TYPE_OUTBOUND or entry_sku_id != sku_id or task_id in exclude_task_ids:
            continue
        reference_location = _driveway_reference_for_goals(goals, G)
        if reference_location is not None:
            tasks.append((task_id, int(deadline), reference_location))
    return tasks


def _outbound_tasks_for_sku(
    sku_id: int,
    needed: int,
    J: dict,
    future_by_sku: Dict[int, List[Tuple[np.ndarray, int]]],
    driveway_columns: List[int],
    G: Graph,
    t: int,
    locked_task_ids: set,
) -> List[OutboundTaskRef]:
    """Released outbound tasks for ``sku_id`` (excluding ones already locked
    to another item), topped up with future (not yet released) queue
    lookahead tasks in queue order until ``needed`` tasks are collected or the
    lookahead window is exhausted.

    Future tasks carry ``task_id=None`` -- they have no stable identity in
    ``J`` yet, so any match against one is tentative and not locked.
    """
    tasks = _released_outbound_tasks_for_sku(sku_id, J, G, locked_task_ids)

    for task, queue_index in future_by_sku.get(sku_id, []):
        if len(tasks) >= needed:
            break

        driveway_column = (
            int(task[QUEUE_DRIVEWAY_COLUMN]) if task.shape[0] > QUEUE_DRIVEWAY_COLUMN else None
        )
        if driveway_column is None or driveway_column not in driveway_columns:
            driveway_column = random.choice(driveway_columns)
        reference_location = G.get_driveway_column_reference(driveway_column)
        if reference_location is None:
            continue

        deadline = (
            int(task[QUEUE_DEADLINE]) if task.shape[0] > QUEUE_DEADLINE else t + queue_index + 1
        )
        tasks.append((None, deadline, reference_location))

    return tasks


def generate_reallocation_tasks(
    queue: np.ndarray,
    J: dict,
    G: Graph,
    Rs: AgentLoader,
    B: int,
    W: int,
    t: int,
    item_task_locks: Dict[Location, int],
    pending_task_targets: Dict[int, Location],
    use_item_task_locks: bool = True,
) -> dict:
    """
    Build shuffle reallocation candidates by assigning each current instance
    of a SKU in warehouse storage to a current (released) or future outbound
    task for that same SKU.

    For each SKU type:
      1. Gather every current warehouse location holding that SKU. Locations
         already locked (see below) to a task keep that pairing; only
         unlocked ("free") locations are up for a new assignment.
      2. Gather every released outbound task for that SKU already in ``J``
         that isn't already locked to some other item. If there are fewer
         such tasks than free locations, step through the queue lookahead
         window ``queue[B:W]`` (``B``/``W`` are 0-based indices into the
         remaining, unpopped queue) and add future outbound tasks of the same
         SKU, in queue order, until the counts match or the window is
         exhausted.
      3. Compute a minimum-cost assignment between free item locations and
         task reference (driveway aisle) locations, cost = grid distance,
         solved with the Munkres algorithm. Munkres pads non-square matrices
         with zero-cost cells internally, so surplus items or surplus tasks
         are simply left unmatched (the "pad the matrix" edge case falls out
         of this for free).
      4. For each matched (item, task) pair, build the candidate rearrangement
         goals for that one item -- empty warehouse locations strictly closer
         to the task's reference point than the item's current location.

    Item/task locking: once a free item is matched to a *released* outbound
    task (one with a stable id in ``J``), that pairing is recorded in
    ``item_task_locks`` (mutated in place) and treated as fixed on every
    subsequent call -- the item is never proposed as a candidate for a
    different task, and the task is excluded from the assignment pool for
    other items. This follows the specific physical item instance, not just
    its current location: if a rearrangement candidate for a locked item is
    later chosen and executed (moving it from ``v1`` to ``v2``), the
    ``pending_task_targets`` bookkeeping (populated by
    ``fast_optimal_insertion.apply_insertions``) carries the lock across that
    move so ``v2`` inherits it rather than the item becoming a fresh,
    up-for-grabs candidate -- see ``_reconcile_locks``. The lock is released
    once the task leaves ``J`` (completed). Matches against not-yet-released
    future tasks are tentative and are not locked, since those tasks have no
    stable id until they're actually released.

    Setting ``use_item_task_locks=False`` disables all of the above: every
    call treats every current instance of a SKU as free and recomputes a
    fresh min-cost assignment against that SKU's outbound tasks from
    scratch, so the same physical item can end up matched to a different
    task from one call to the next. ``item_task_locks`` and
    ``pending_task_targets`` are ignored (neither read nor written) in this
    mode.

    Args:
        queue: Remaining task queue with rows ``(sku_id, task_type[,
            deadline[, driveway_column]])``.
        J: The dictionary of live real tasks (outbound/inbound).
        G: The graph of the warehouse.
        Rs: The dictionary of agents.
        B: Start index (inclusive) of the lookahead window in the queue.
        W: End index (exclusive) of the lookahead window in the queue.
        t: Current simulation timestep.
        item_task_locks: Persistent ``{item_location: task_id}`` mapping,
            owned by the caller and mutated in place across calls.
        pending_task_targets: Persistent ``{task_id: goal_location}`` mapping
            of committed-but-not-yet-executed shuffles for locked items,
            owned by the caller and mutated in place across calls (written by
            ``apply_insertions``, consumed here).
        use_item_task_locks: When ``True`` (default), preserve item->task
            pairings across calls as described above. When ``False``, ignore
            ``item_task_locks``/``pending_task_targets`` entirely and
            recompute a fresh min-cost assignment over every current
            instance of each SKU on every call.

    Returns:
        A dictionary of reallocation tasks.

        Reallocation task is defined as: tau=(C_i, r_i, d_i, sigma_i, ref_i, task_id), where
        C_i = {(S_i^k, D_i^k)}_{k=1}^{|C_i|} is the set of (start_loc, end_loc) pairs for task i.
        r_i, d_i are the release time and deadline of the reallocation task
        sigma_i is the task type, 0 for outbound, 1 for inbound, 2 for shuffle.
        ref_i is a representative driveway vertex in the aisle that the
        corresponding real outbound task has been (or will be) committed to,
        used as the benefit reference point instead of each candidate goal's
        own column (see ``optimal_insertion_gurobi.benefit``).
        task_id is the real outbound task in ``J`` this candidate is locked
        to, or ``None`` if it's a tentative match against a future task.
    """
    reallocation_tasks: dict = {}

    # Aisles (driveway columns) currently available for an outbound task's
    # eventual delivery. Empty if the driveway is full.
    driveway_columns = sorted({loc[1] for loc in G.driveway.get_empty_locations()})
    if not driveway_columns:
        return reallocation_tasks

    dist_matrix = G.get_distance_matrix()
    empty_locations = list(G.warehouse.get_empty_locations())
    empty_idx = np.array([G.location_index(loc) for loc in empty_locations], dtype=int)
    if empty_idx.size == 0:
        return reallocation_tasks

    if use_item_task_locks:
        _reconcile_locks(item_task_locks, pending_task_targets, J, G)

    future_by_sku = _future_outbound_tasks_by_sku(queue, B, W)
    munkres = Munkres()
    next_id = 0

    for sku_id in G.warehouse.get_all_skus():
        sku_locations = G.warehouse.get_sku_instances(sku_id)
        if not sku_locations:
            continue

        if use_item_task_locks:
            free_locations = [loc for loc in sku_locations if loc not in item_task_locks]
            locked_locations = [loc for loc in sku_locations if loc in item_task_locks]
            locked_task_ids = {item_task_locks[loc] for loc in locked_locations}
        else:
            free_locations = list(sku_locations)
            locked_locations = []
            locked_task_ids = set()

        # (start_loc, deadline, reference_location, task_id) for every item
        # this round, whether its task pairing is already locked or freshly
        # assigned below.
        matched: List[Tuple[Location, int, Location, Optional[int]]] = []

        for loc in locked_locations:
            task_id = item_task_locks[loc]
            task_entry = J.get(task_id)
            if task_entry is None:
                continue
            _starts, goals, deadline, _sku, _type = task_entry
            reference_location = _driveway_reference_for_goals(goals, G)
            if reference_location is None:
                continue
            matched.append((loc, int(deadline), reference_location, task_id))

        if free_locations:
            outbound_tasks = _outbound_tasks_for_sku(
                sku_id, len(free_locations), J, future_by_sku, driveway_columns, G, t, locked_task_ids
            )
            if outbound_tasks:
                item_idx = np.array([G.location_index(loc) for loc in free_locations], dtype=int)
                ref_idx = np.array(
                    [G.location_index(ref) for _tid, _d, ref in outbound_tasks], dtype=int
                )
                cost_matrix = dist_matrix[np.ix_(item_idx, ref_idx)].tolist()
                assignment = munkres.compute(cost_matrix)

                for item_row, task_col in assignment:
                    start_loc = free_locations[item_row]
                    task_id, deadline, reference_location = outbound_tasks[task_col]
                    matched.append((start_loc, deadline, reference_location, task_id))
                    if use_item_task_locks and task_id is not None:
                        item_task_locks[start_loc] = task_id

        for start_loc, deadline, reference_location, task_id in matched:
            dist_to_reference = dist_matrix[G.location_index(reference_location)]
            d_start = dist_to_reference[G.location_index(start_loc)]
            closer = dist_to_reference[empty_idx] < d_start
            C_i = {
                (start_loc, goal_loc)
                for goal_loc, is_closer in zip(empty_locations, closer)
                if is_closer and goal_loc != start_loc
            }
            if not C_i:
                continue

            r_i = t
            d_i = deadline
            sigma_i = TASK_TYPE_SHUFFLE
            reallocation_tasks[next_id] = (C_i, r_i, d_i, sigma_i, reference_location, task_id)
            next_id += 1

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
