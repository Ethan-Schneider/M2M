"""Allocator-level integration tests for roadmap section 1.6.

Verifies that the cost-element refactor lands correctly in the live
allocator path (``fast_greedy_call`` and the LNS wrappers that call it).
The tests below run real allocations on small synthetic ``J`` dictionaries
and assert behaviour that depends on the 1.6 wiring:

* ``deadline_weight > 0`` increases the chosen task's contribution to
  total cost (tardiness term is actually added, not silently dropped).
* ``agent_task_sequence_time[m]`` is included in base cost so an idle agent
  is preferred over an already-busy one at otherwise-equal per-task cost.
* Shuffle (type=2) tasks pull their SKU-distribution term from the
  ``rearrangement_sku_distribution_costs`` matrix, not the OB matrix.
* The three-way dispatch never raises for the canonical task types and
  raises ``ValueError`` for unknown types.

These tests are deliberately minimal -- they don't simulate a full timestep
loop. They build a hand-crafted ``J`` and call the allocator directly so
the behaviour assertions are tight and the failure messages clear.
"""

from __future__ import annotations

import numpy as np
import pytest

from GT_grid_world.src.agent import Agent, AgentLoader
from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
    TASK_TYPE_INBOUND,
    TASK_TYPE_OUTBOUND,
    TASK_TYPE_SHUFFLE,
    construct_cost_elements,
)
from GT_grid_world.src.task_allocation_algorithms.initial_solutions.fast_greedy import (
    fast_greedy_call,
)


# ---------------------------------------------------------------------------
# Helpers for fixture-shaped allocator state
# ---------------------------------------------------------------------------
def _agents_at(graph, positions):
    """Build an ``AgentLoader`` with one agent per position. Positions must be
    valid (non-obstacle) cells in the graph."""
    agents = [Agent(agent_id=i, state=loc) for i, loc in enumerate(positions)]
    return AgentLoader(agents)


def _make_outbound_task(graph, sku, *, deadline=100):
    """Build a single OB task tuple from a populated warehouse cell to any
    free driveway cell. Returns ``(task_id, task_tuple)``."""
    full = graph.warehouse.get_sku_instances(sku)
    stations = graph.driveway.get_empty_locations()
    if not full or not stations:
        pytest.skip("graph fixture has no full warehouse cell or no empty driveway")
    return 1, (frozenset(full), frozenset(stations), deadline, sku, TASK_TYPE_OUTBOUND)


def _make_shuffle_task(graph, sku, *, deadline=100):
    """Build a single shuffle task tuple from a populated warehouse cell to
    any empty warehouse cell."""
    full = graph.warehouse.get_sku_instances(sku)
    empty = graph.warehouse.get_empty_locations()
    if not full or not empty:
        pytest.skip("graph fixture has no full warehouse cell or no empty cell")
    return 2, (frozenset(full), frozenset(empty), deadline, sku, TASK_TYPE_SHUFFLE)


# ---------------------------------------------------------------------------
# Tardiness wiring (deadline_weight)
# ---------------------------------------------------------------------------
def test_fast_greedy_deadline_weight_off_yields_pure_base_cost(
    populated_graph, minimal_stats, seeded_rng
):
    """With ``deadline_weight=0`` the recorded total cost equals the agent->
    start + start->goal Manhattan distance plus any agent_task_sequence_time
    (which is zero here -- agents start idle). We check that the returned
    total cost is finite, non-negative, and independent of the deadline
    value, which together imply the deadline term isn't silently leaking in."""
    sku = next(s for s in populated_graph.warehouse.get_all_skus()
               if populated_graph.warehouse.get_sku_instance_count(s) >= 1)

    Rs_a = _agents_at(populated_graph, [(17, 0)])
    Rs_b = _agents_at(populated_graph, [(17, 0)])
    task_id, t_far = _make_outbound_task(populated_graph, sku, deadline=10_000)
    _, t_near = _make_outbound_task(populated_graph, sku, deadline=5)

    _, _, cost_far = fast_greedy_call(
        minimal_stats, populated_graph, Rs_a, {task_id: t_far}, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.0,
        sku_distribution_weight=0.0,
    )
    _, _, cost_near = fast_greedy_call(
        minimal_stats, populated_graph, Rs_b, {task_id: t_near}, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.0,
        sku_distribution_weight=0.0,
    )

    assert cost_far == cost_near, \
        "with deadline_weight=0 the deadline value must not affect total cost"
    assert cost_far >= 0


def test_fast_greedy_deadline_weight_on_increases_total_cost_for_urgent_task(
    populated_graph, minimal_stats, seeded_rng
):
    """With ``deadline_weight=1`` an urgent task (deadline near current_time)
    must show a strictly larger total cost than the same task with deadline
    far in the future, because the tardiness term is now wired in."""
    sku = next(s for s in populated_graph.warehouse.get_all_skus()
               if populated_graph.warehouse.get_sku_instance_count(s) >= 1)

    Rs_a = _agents_at(populated_graph, [(17, 0)])
    Rs_b = _agents_at(populated_graph, [(17, 0)])
    task_id, t_far = _make_outbound_task(populated_graph, sku, deadline=10_000)
    _, t_urgent = _make_outbound_task(populated_graph, sku, deadline=0)

    _, _, cost_far = fast_greedy_call(
        minimal_stats, populated_graph, Rs_a, {task_id: t_far}, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=1.0,
        sku_distribution_weight=0.0,
    )
    _, _, cost_urgent = fast_greedy_call(
        minimal_stats, populated_graph, Rs_b, {task_id: t_urgent}, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=1.0,
        sku_distribution_weight=0.0,
    )

    assert cost_urgent > cost_far, (
        "with deadline_weight=1, an at-deadline OB task should cost more than the "
        "same task far from its deadline -- evidence the tardiness term is wired in"
    )


# ---------------------------------------------------------------------------
# Execution-time accounting (agent_task_sequence_time)
# ---------------------------------------------------------------------------
def test_fast_greedy_prefers_idle_agent_over_busy_one(
    populated_graph, minimal_stats, seeded_rng
):
    """Two agents at the SAME state, but agent 1 already has a long
    pre-allocated task in its sequence. The 1.6 base cost adds
    ``agent_task_sequence_time[m]``, so agent 0 (idle) must be chosen for
    the new task even though both agents are otherwise identical from the
    perspective of agent->start distance.
    """
    sku = next(s for s in populated_graph.warehouse.get_all_skus()
               if populated_graph.warehouse.get_sku_instance_count(s) >= 1)

    same_pos = (17, 0)
    Rs = _agents_at(populated_graph, [same_pos, same_pos])

    # Pre-load agent 1 with a long task (drives up agent_task_sequence_time[1]).
    full_cells = list(populated_graph.warehouse.get_sku_instances(sku))
    distant_full = max(full_cells, key=lambda c: abs(c[0] - same_pos[0]) + abs(c[1] - same_pos[1]))
    distant_drive = max(populated_graph.driveway.get_empty_locations(),
                        key=lambda c: abs(c[0] - distant_full[0]) + abs(c[1] - distant_full[1]))
    pre_allocated_id = 999
    Rs.agents[1].task_sequence.append((pre_allocated_id, distant_full, distant_drive, 100))

    # Build a fresh OB task whose pickup is somewhere reachable for either agent.
    task_id, t = _make_outbound_task(populated_graph, sku, deadline=100)
    # Drop the cell agent 1 is already heading to from the fresh task's eligible
    # starts/goals, otherwise the allocator would refuse to reuse them.
    fresh_starts = frozenset(s for s in t[0] if s != distant_full)
    fresh_goals = frozenset(g for g in t[1] if g != distant_drive)
    if not fresh_starts or not fresh_goals:
        pytest.skip("no eligible cells left after excluding the busy agent's pre-allocation")
    t = (fresh_starts, fresh_goals, t[2], t[3], t[4])

    # Include the pre-allocated task in J so fast_greedy_call's early-return
    # guard (len(J) == sum(task_sequence lengths)) does not fire. The task is
    # excluded from the unallocated set inside ``construct_cost_elements`` so
    # we don't actually re-allocate it.
    J = {
        pre_allocated_id: (frozenset({distant_full}), frozenset({distant_drive}),
                           100, sku, TASK_TYPE_OUTBOUND),
        task_id: t,
    }

    _, allocations, _ = fast_greedy_call(
        minimal_stats, populated_graph, Rs, J, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.0,
        sku_distribution_weight=0.0,
    )

    assert allocations, "expected the new task to be allocated"
    assigned_agent = allocations[0][0]
    assert assigned_agent == 0, (
        f"expected idle agent 0 to win over busy agent 1, but task was assigned to "
        f"agent {assigned_agent}. agent_task_sequence_time wiring may be missing."
    )


# ---------------------------------------------------------------------------
# Per-task-type SKU distribution dispatch in the live allocator
# ---------------------------------------------------------------------------
def test_fast_greedy_shuffle_is_droppable_when_detour_dominates(
    populated_graph, minimal_stats, seeded_rng
):
    """crM2M (roadmap 4.2) replaces the 1.6 type=2 SKU-matrix routing entirely
    with the rearrangement utility ``U = b - lambda*Delta`` (cost = -U, gated).

    Here the only agent is anchored at its home ``h_0 = (17, 0)``. The detour to
    fetch any SKU instance is then ``Delta ~= dist(h_0, s) + dist(s, h_0) -
    dist(h_0, h_0) = 2*dist(s, h_0)``, which swamps the placement benefit
    ``b <= dist(s, h_0)``. So ``U <= 0`` for every candidate and the (opportunistic,
    droppable) shuffle is correctly gated out -- the allocator simply assigns
    nothing rather than forcing a wasteful move. ``sku_distribution_weight`` no
    longer influences type=2 cost at all."""
    sku = next(s for s in populated_graph.warehouse.get_all_skus()
               if populated_graph.warehouse.get_sku_instance_count(s) >= 1)

    Rs = _agents_at(populated_graph, [(17, 0)])
    task_id, t = _make_shuffle_task(populated_graph, sku, deadline=100)

    _, allocations, _ = fast_greedy_call(
        minimal_stats, populated_graph, Rs, {task_id: t}, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.0,
        sku_distribution_weight=1.0,
    )

    assert not allocations, (
        "a shuffle whose detour dominates its benefit (agent at home) must be "
        "dropped under crM2M, not force-allocated"
    )


def test_fast_greedy_outbound_uses_outbound_matrix(
    populated_graph, minimal_stats, seeded_rng
):
    """Symmetric sanity check: an OB-only J still allocates with
    sku_distribution_weight>0 (its row in the OB matrix is populated)."""
    sku = next(s for s in populated_graph.warehouse.get_all_skus()
               if populated_graph.warehouse.get_sku_instance_count(s) >= 1)

    Rs = _agents_at(populated_graph, [(17, 0)])
    task_id, t = _make_outbound_task(populated_graph, sku, deadline=100)

    _, allocations, cost = fast_greedy_call(
        minimal_stats, populated_graph, Rs, {task_id: t}, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.0,
        sku_distribution_weight=1.0,
    )

    assert allocations, "OB task with sku_distribution_weight>0 should be allocated"
    assert cost != np.inf


def test_construct_cost_elements_handles_all_three_task_types_in_one_J(
    populated_graph, minimal_stats, seeded_rng
):
    """Mixed J: one of each type; allocator runs without raising.

    The mandatory inbound (20) and outbound (10) tasks are always allocated. The
    type=2 shuffle (30) is opportunistic under crM2M: with agents anchored at
    their homes its detour dominates its benefit (``U <= 0``), so it is gated out.
    The invariant we assert is that the two mandatory tasks are allocated and the
    integration path doesn't raise; the shuffle is allowed to be dropped."""
    sku = next(s for s in populated_graph.warehouse.get_all_skus()
               if populated_graph.warehouse.get_sku_instance_count(s) >= 1)

    Rs = _agents_at(populated_graph, [(17, 0), (17, 6), (17, 13)])
    full = list(populated_graph.warehouse.get_sku_instances(sku))
    empty = populated_graph.warehouse.get_empty_locations()
    stations = populated_graph.driveway.get_empty_locations()
    drive_full_cell = stations[0]
    populated_graph.driveway.add_sku_instance(sku, drive_full_cell)
    drive_full = populated_graph.driveway.get_full_locations()
    drive_empty = [s for s in stations if s != drive_full_cell]
    if not (full and empty and drive_full and drive_empty):
        pytest.skip("populated_graph not large enough for a 3-type mixed J")

    J = {
        10: (frozenset(full), frozenset(drive_empty), 100, sku, TASK_TYPE_OUTBOUND),
        20: (frozenset(drive_full), frozenset(empty), 100, sku, TASK_TYPE_INBOUND),
        30: (frozenset(full), frozenset(empty), 100, sku, TASK_TYPE_SHUFFLE),
    }

    _, allocations, cost = fast_greedy_call(
        minimal_stats, populated_graph, Rs, J, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.5,
        sku_distribution_weight=1.0,
    )

    allocated_task_ids = {a[1] for a in allocations}
    assert {10, 20} <= allocated_task_ids, (
        f"expected both mandatory real tasks (IB 20, OB 10) to be allocated, "
        f"got {allocated_task_ids}"
    )
