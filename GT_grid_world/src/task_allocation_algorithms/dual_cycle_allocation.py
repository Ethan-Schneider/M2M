"""Allocator-side dual-cycle task pairing.

Dual cycling chains a same-aisle outbound onto an inbound (IB->OB at aisles)
or a same-driveway inbound onto an outbound (OB->IB at the driveway), so the
agent does not deadhead back across the warehouse after a single task.

Architectural note (vs the previous simulate-side post-completion hook):

The earlier 1.5-skeleton implementation chained tasks reactively in
``simulate.simulate``: when an agent finished a delivery and would otherwise
idle, the simulator searched ``J`` for an eligible follow-on and appended it
to that agent's ``task_sequence``. That hook fired *after* path planning was
already complete for the tick, so the chained task did not benefit from the
allocator's cost-aware reasoning, and the per-agent search had no visibility
into the global snapshot the allocator was working with.

The proactive-rearrangement design needs dual cycling to live inside the
allocation pipeline (it interacts with rearrangement task placement and
needs to participate in tour-cost reasoning), so this module re-implements
the pairing as a pipeline stage that runs at the end of ``TaskAllocation``,
after the inner allocator (``fast_greedy``, ``c_lns``, ``hbh_mla_star``, ...)
has produced its assignment. The stage:

1. Walks every agent whose ``task_sequence`` is non-empty after the allocator.
2. Looks at the *last* task in the sequence (the cell where the agent will
   naturally end up after honouring the allocator's commitments).
3. If that endpoint is dual-cycle-eligible, scans ``J`` for a complementary
   task that is (a) unallocated in any agent's sequence, (b) compatible with
   the endpoint (same-aisle / same-driveway, valid drop), and appends the
   best match to ``agent.task_sequence``.

Re-running the stage every tick is intentional: the snapshot-style allocators
in this codebase clear all but the head task on entry, so the chain has to be
re-established each call. The ``J`` membership check guarantees that a
chained task cannot drift across allocators, ticks, or agents.

The chain-finder helpers themselves are adapted from the simulate-side
``_find_aisle_dual_cycle_chain`` / ``_find_driveway_dual_cycle_chain`` (now
removed), with two differences: (1) the "completed goal" is the *last task's
goal*, not a just-finished delivery, and (2) the eligibility predicate
treats the warehouse / driveway occupancy snapshot as the post-allocator
projection rather than the live in-tick state.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..agent import AgentLoader
from ..graph import Graph
from ..analysis.statistics import Stats
from ..logging_config import get_logger

_log = get_logger("dual_cycle_allocation")


# Mirror of ``simulate.TASK_TYPE_*``. Repeated here (rather than imported from
# simulate) because the allocator package must not depend on the simulator
# package.
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1


def _collect_global_allocation_snapshot(
    Rs: AgentLoader,
) -> Tuple[Set[int], Set[Tuple[int, int]]]:
    """Return ``(allocated_task_ids, allocated_locs)`` across every agent's
    task sequence. Used by the chain finders so a candidate cannot be stolen
    from another agent's already-committed plan.
    """
    allocated_task_ids: Set[int] = set()
    allocated_locs: Set[Tuple[int, int]] = set()
    for ag in Rs.agents:
        for task_tuple in ag.task_sequence:
            allocated_task_ids.add(task_tuple[0])
            allocated_locs.add(task_tuple[1])
            allocated_locs.add(task_tuple[2])
    return allocated_task_ids, allocated_locs


def _find_aisle_dual_cycle_chain(
    agent_state: Tuple[int, int],
    completed_goal: Tuple[int, int],
    J: Dict[int, Tuple],
    allocated_task_ids: Set[int],
    allocated_locs: Set[Tuple[int, int]],
    G: Graph,
) -> Optional[Tuple[int, Tuple[int, int], Tuple[int, int], int]]:
    """Aisle dual cycling (IB -> OB).

    The agent will end its current commitments at ``completed_goal`` (a
    warehouse aisle cell). Search ``J`` for an unallocated outbound (type=0)
    task whose pickup is in the SAME aisle (same column) and whose dropoff
    has a valid driveway-empty cell. If one is found, return a concrete
    ``(task_id, chosen_start, chosen_goal, deadline)`` tuple ready to append
    to ``agent.task_sequence``. Otherwise return ``None``.

    The chosen ``(start, goal)`` minimises deadhead from ``completed_goal`` to
    ``start`` plus ``start -> goal`` travel; deadline tie-breaking is
    deferred to the allocator's deadline term.
    """
    if not G.is_warehouse_aisle_location(completed_goal):
        return None

    same_aisle_cells = set(G.get_same_aisle_locations(completed_goal))
    driveway_empty = set(G.driveway.get_empty_locations())
    warehouse_full = set(G.warehouse.get_full_locations())

    best: Optional[Tuple[int, Tuple[int, int], Tuple[int, int], int]] = None
    best_cost = float("inf")
    for task_id, (start_locs, goal_locs, deadline, _sku, type_) in J.items():
        if task_id in allocated_task_ids:
            continue
        if type_ != TASK_TYPE_OUTBOUND:
            continue
        eligible_starts = [
            s
            for s in start_locs
            if s in same_aisle_cells and s in warehouse_full and s not in allocated_locs
        ]
        if not eligible_starts:
            continue
        eligible_goals = [g for g in goal_locs if g in driveway_empty and g not in allocated_locs]
        if not eligible_goals:
            continue
        chosen_start = min(eligible_starts, key=lambda s: G.get_distance(completed_goal, s))
        chosen_goal = min(eligible_goals, key=lambda g: G.get_distance(chosen_start, g))
        cost = G.get_distance(completed_goal, chosen_start) + G.get_distance(
            chosen_start, chosen_goal
        )
        if cost < best_cost:
            best_cost = cost
            best = (task_id, chosen_start, chosen_goal, deadline)
    return best


def _find_driveway_dual_cycle_chain(
    agent_state: Tuple[int, int],
    completed_goal: Tuple[int, int],
    J: Dict[int, Tuple],
    allocated_task_ids: Set[int],
    allocated_locs: Set[Tuple[int, int]],
    G: Graph,
) -> Optional[Tuple[int, Tuple[int, int], Tuple[int, int], int]]:
    """Driveway dual cycling (OB -> IB at driveways).

    The agent will end its current commitments at ``completed_goal`` (a
    driveway cell). Search ``J`` for an unallocated inbound (type=1) task
    whose pickup is at any driveway cell currently holding a pre-placed
    SKU. If one is found and has a valid warehouse-empty dropoff cell,
    return a concrete ``(task_id, chosen_start, chosen_goal, deadline)``
    tuple ready to append to ``agent.task_sequence``. Otherwise return
    ``None``.

    On the current ``symbotic_2026`` branch the driveway is a single
    physical region, so "same driveway" is trivially "any driveway cell".
    When the per-cell I/O direction typing from ``local_task_reallocation``
    lands (DPS-style mixed layout), this helper will tighten the
    eligibility filter to the matching subregion.
    """
    if not G.is_driveway_location(completed_goal):
        return None

    driveway_full = set(G.driveway.get_full_locations())
    warehouse_empty = set(G.warehouse.get_empty_locations())

    best: Optional[Tuple[int, Tuple[int, int], Tuple[int, int], int]] = None
    best_cost = float("inf")
    for task_id, (start_locs, goal_locs, deadline, _sku, type_) in J.items():
        if task_id in allocated_task_ids:
            continue
        if type_ != TASK_TYPE_INBOUND:
            continue
        eligible_starts = [
            s for s in start_locs if s in driveway_full and s not in allocated_locs
        ]
        if not eligible_starts:
            continue
        eligible_goals = [
            g for g in goal_locs if g in warehouse_empty and g not in allocated_locs
        ]
        if not eligible_goals:
            continue
        chosen_start = min(eligible_starts, key=lambda s: G.get_distance(completed_goal, s))
        chosen_goal = min(eligible_goals, key=lambda g: G.get_distance(chosen_start, g))
        cost = G.get_distance(completed_goal, chosen_start) + G.get_distance(
            chosen_start, chosen_goal
        )
        if cost < best_cost:
            best_cost = cost
            best = (task_id, chosen_start, chosen_goal, deadline)
    return best


def apply_dual_cycle_pairing(
    S: Stats,
    G: Graph,
    Rs: AgentLoader,
    J: Dict[int, Tuple],
    t: int,
    *,
    aisle_dual_cycle: bool = False,
    driveway_dual_cycle: bool = False,
) -> AgentLoader:
    """Post-allocator dual-cycle pairing stage.

    Iterates over agents whose ``task_sequence`` was populated by the inner
    allocator and appends a same-aisle / same-driveway follow-on task to
    those whose tour ends at a dual-cycle-eligible cell. Returns ``Rs`` for
    pipeline composition; mutation is in place.

    ``aisle_dual_cycle`` and ``driveway_dual_cycle`` are independent
    ablation knobs and may both be on. They never inspect the *same* head
    task (an IB tail goes through aisle DC, an OB tail through driveway DC).

    Per-tick re-execution (cheap; O(|J| x |agents_with_tail|)) is required
    because M2M's snapshot-style allocators wipe trailing tasks on entry,
    so a chain established last tick is gone by the time control returns
    here this tick. Pairs naturally re-form if the underlying tasks are
    still in ``J``.
    """
    if not (aisle_dual_cycle or driveway_dual_cycle):
        return Rs
    if not J:
        return Rs

    chained_count = 0
    for ag in Rs.agents:
        if not ag.task_sequence:
            continue

        last_task = ag.task_sequence[-1]
        last_task_id = last_task[0]
        last_goal = last_task[2]

        last_task_entry = J.get(last_task_id)
        if last_task_entry is None:
            # The tail of the agent's sequence references a task that has
            # already left J (e.g. just-completed inbound delivery this
            # tick). Don't try to chain off a stale endpoint.
            continue
        last_task_type = last_task_entry[4]

        # ``allocated_task_ids`` / ``allocated_locs`` are recomputed per
        # agent so an earlier agent's freshly-chained tail is visible to
        # later agents in this same pass (stops two agents from chaining
        # the same OB on top of distinct aisle IBs in one tick).
        allocated_task_ids, allocated_locs = _collect_global_allocation_snapshot(Rs)

        chained_tuple: Optional[Tuple[int, Tuple[int, int], Tuple[int, int], int]] = None
        if aisle_dual_cycle and last_task_type == TASK_TYPE_INBOUND:
            chained_tuple = _find_aisle_dual_cycle_chain(
                ag.state, last_goal, J, allocated_task_ids, allocated_locs, G
            )
            if chained_tuple is not None:
                _log.debug(
                    "Aisle dual cycle (IB->OB) at t=%s: agent %s chained task %s after task %s",
                    t,
                    ag.id,
                    chained_tuple[0],
                    last_task_id,
                )
        elif driveway_dual_cycle and last_task_type == TASK_TYPE_OUTBOUND:
            chained_tuple = _find_driveway_dual_cycle_chain(
                ag.state, last_goal, J, allocated_task_ids, allocated_locs, G
            )
            if chained_tuple is not None:
                _log.debug(
                    "Driveway dual cycle (OB->IB) at t=%s: agent %s chained task %s after task %s",
                    t,
                    ag.id,
                    chained_tuple[0],
                    last_task_id,
                )

        if chained_tuple is not None:
            ag.task_sequence.append(chained_tuple)
            chained_count += 1

    if chained_count:
        _log.debug("apply_dual_cycle_pairing chained %d tasks at t=%s", chained_count, t)

    return Rs
