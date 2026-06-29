"""Bridge generator reallocation candidates into the separate ``J_a`` pool.

``generate_reallocation_tasks`` emits, per flagged future demand, a 4-tuple
``tau = (C_i, r_i, d_i, sigma_i)`` where ``C_i`` is a set of coupled
``(start, goal)`` pairs (each pair is same-aisle by construction).

For crM2M (concatenated rearrangement) the shuffles are scored *alongside* real
tasks by the allocator, but they are kept in their own dictionary ``J_a`` rather
than merged into the persistent real-task pool ``J``. Keeping them separate gives
shuffles their own lifecycle (expiry / pruning) and their own completion track in
stats, and prevents proactive moves from polluting ``J``. Each tick the loop
hands the allocator a transient union ``{**J, **J_a}`` so the cost cube can score
both; ``J`` and ``J_a`` themselves stay the canonical stores.

This module converts each 4-tuple into a single 5-tuple ``J_a`` entry whose
``start_frozenset`` is every candidate SKU-instance cell and whose
``goal_frozenset`` is every candidate empty cell. The per-aisle coupling
(``s_p in S^k_n`` / ``d_q in D^k_n``) is enforced at scoring time by the
same-column mask in ``construct_cost_elements.compute_crm2m_terms`` rather than
by exploding ``C_i`` into one task per pair (which would blow up ``N``).
"""

from typing import Dict, Tuple

from ..agent import AgentLoader
from ..graph import Graph

TASK_TYPE_SHUFFLE = 2

# Rearrangement task ids live above this base so they never collide with real
# task ids.
REARRANGEMENT_TASK_ID_BASE = 100000


def _active_rearrangement_skus(J_a: Dict[int, Tuple]) -> set:
    """SKU ids that already have a live rearrangement task in ``J_a``."""
    return {
        task[3]
        for task in J_a.values()
        if task[4] == TASK_TYPE_SHUFFLE
    }


def add_reallocation_tasks_to_J_a(
    Ta: Dict[int, Tuple],
    J_a: Dict[int, Tuple],
    G: Graph,
    next_rearrangement_task_id: int,
) -> int:
    """Add generator reallocation candidates to ``J_a`` as crM2M entries.

    Each candidate becomes one 5-tuple entry
    ``(start_frozenset, goal_frozenset, deadline, sku_id, TASK_TYPE_SHUFFLE)``.
    Candidates whose SKU already has a live rearrangement task in ``J_a`` are
    skipped (one rearrangement per SKU in flight at a time).

    Args:
        Ta: ``{n: (C_i, r_i, d_i, sigma_i)}`` from ``generate_reallocation_tasks``.
        J_a: Separate rearrangement dictionary; mutated in place with new entries.
        G: Warehouse graph (used to read the SKU id at a start cell).
        next_rearrangement_task_id: First id to assign; ids increase from here.

    Returns:
        The next free rearrangement task id after assignment.
    """
    if not Ta:
        return next_rearrangement_task_id

    active_skus = _active_rearrangement_skus(J_a)
    current_id = max(next_rearrangement_task_id, REARRANGEMENT_TASK_ID_BASE)

    for _n, task_data in Ta.items():
        C_i, _release, deadline, sigma = task_data
        if sigma != TASK_TYPE_SHUFFLE or not C_i:
            continue

        start_locs = frozenset(s for s, _g in C_i)
        goal_locs = frozenset(g for _s, g in C_i)
        if not start_locs or not goal_locs:
            continue

        # All starts are instances of the same flagged SKU (the generator builds
        # C_i from one SKU's instances), so read the SKU id from any start cell.
        sku_instance = G.warehouse.get_sku_at_location(next(iter(start_locs)))
        if sku_instance is None:
            continue
        sku_id = sku_instance.sku_id

        if sku_id in active_skus:
            continue

        J_a[current_id] = (start_locs, goal_locs, int(deadline), sku_id, TASK_TYPE_SHUFFLE)
        active_skus.add(sku_id)
        current_id += 1

    return current_id


def drop_expired_rearrangements(J_a: Dict[int, Tuple], Rs: AgentLoader, t: int) -> None:
    """Remove ``J_a`` shuffles whose look-ahead window has passed.

    A shuffle's deadline ``d_i`` is the (proxy) time the future demand it was
    meant to prepare for becomes imminent. Once ``t >= d_i`` the demand is here
    and the move is no longer a *rearrangement* -- the item will simply be served
    from wherever it currently sits, so we must not keep trying to shuffle it.
    Such tasks are dropped from ``J_a`` and from any agent's task sequence.

    The one exception: an agent already *carrying* the shuffled SKU (status 2) is
    allowed to finish the in-flight move to its (still-valid, empty) goal cell
    rather than being stranded holding an item with no task.
    """
    expired = {
        task_id
        for task_id, task in J_a.items()
        if task[4] == TASK_TYPE_SHUFFLE and t >= task[2]
    }
    if not expired:
        return

    # Keep any expired shuffle whose SKU is already in an agent's gripper.
    for agent in Rs.agents:
        if agent.status == 2 and agent.task_sequence:
            head_id = agent.task_sequence[0][0]
            expired.discard(head_id)

    if not expired:
        return

    for agent in Rs.agents:
        if not agent.task_sequence:
            continue
        head_expired = agent.task_sequence[0][0] in expired
        new_sequence = [task for task in agent.task_sequence if task[0] not in expired]
        if len(new_sequence) == len(agent.task_sequence):
            continue
        agent.task_sequence = new_sequence
        if head_expired:
            # The active head was aborted (agent was not carrying it -- carried
            # shuffles were excluded above), so reset execution state and let the
            # router recompute a path on the next routing pass.
            agent.path_sequence = []
            agent.status = 1 if new_sequence else 0

    for task_id in expired:
        J_a.pop(task_id, None)


def prune_uncommitted_rearrangements(J_a: Dict[int, Tuple], Rs: AgentLoader) -> None:
    """Drop ``J_a`` shuffles not present in any agent's task sequence.

    M2M re-allocates from scratch every tick (trailing task-sequence entries are
    cleared and re-decided), so rearrangement *candidates* that no agent ended up
    executing should not persist -- they are regenerated fresh next tick from the
    look-ahead window. Shuffles an agent is actually executing remain in a task
    sequence and are kept so the simulator can complete them.
    """
    in_sequence = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            in_sequence.add(task[0])

    stale = [
        task_id
        for task_id, task in J_a.items()
        if task[4] == TASK_TYPE_SHUFFLE and task_id not in in_sequence
    ]
    for task_id in stale:
        J_a.pop(task_id)
