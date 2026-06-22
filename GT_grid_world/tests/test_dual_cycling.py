"""Unit and integration tests for dual cycling.

Dual cycling has been moved out of ``simulate.py`` into the allocation
pipeline (``task_allocation_algorithms.dual_cycle_allocation``). Pairing now
happens *after* the inner allocator (``fast_greedy``, ``c_lns``, ``py_lns``,
...) returns and *before* path planning, so chained IB->OB / OB->IB tails
participate in routing on the same tick they're created.

Covers:

* Graph aisle / driveway membership helpers (``is_warehouse_aisle_location``,
  ``is_driveway_location``, ``get_same_aisle_locations``).
* The two ``_find_*_dual_cycle_chain`` helpers in
  ``dual_cycle_allocation.py`` -- positive match, different-aisle / no-SKU
  rejection, already-allocated rejection, wrong endpoint-type rejection.
* End-to-end: ``apply_dual_cycle_pairing`` chains an OB onto an IB tail
  when the aisle flag is on, chains an IB onto an OB tail when the
  driveway flag is on, and is a no-op when both are off.
"""

from __future__ import annotations

import pytest

from GT_grid_world.src.agent import Agent, AgentLoader
from GT_grid_world.src.task_allocation_algorithms.dual_cycle_allocation import (
    TASK_TYPE_INBOUND,
    TASK_TYPE_OUTBOUND,
    _collect_global_allocation_snapshot,
    _find_aisle_dual_cycle_chain,
    _find_driveway_dual_cycle_chain,
    apply_dual_cycle_pairing,
)

# Type=2 (shuffle / rearrangement) is referenced in some tuple-shape negative
# tests below; reuse the simulate-side constant since both sides agree on
# the wire value.
TASK_TYPE_SHUFFLE = 2


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def test_is_warehouse_aisle_location_true_for_aisle_cell(populated_graph):
    aisle = populated_graph.aisle_locations[0]
    assert populated_graph.is_warehouse_aisle_location(aisle) is True


def test_is_warehouse_aisle_location_false_for_driveway_cell(populated_graph):
    driveway = populated_graph.station_locations[0]
    assert populated_graph.is_warehouse_aisle_location(driveway) is False


def test_is_driveway_location_true_for_station_cell(populated_graph):
    driveway = populated_graph.station_locations[0]
    assert populated_graph.is_driveway_location(driveway) is True


def test_is_driveway_location_false_for_aisle_cell(populated_graph):
    aisle = populated_graph.aisle_locations[0]
    assert populated_graph.is_driveway_location(aisle) is False


def test_get_same_aisle_locations_returns_same_column(populated_graph):
    aisle = populated_graph.aisle_locations[0]
    column = aisle[1]
    same = populated_graph.get_same_aisle_locations(aisle)
    assert aisle in same, "the input cell should be included in its own aisle"
    assert all(loc[1] == column for loc in same), \
        "all returned cells must share the input cell's column"
    expected = [a for a in populated_graph.aisle_locations if a[1] == column]
    assert set(same) == set(expected)


def test_get_same_aisle_locations_empty_for_non_aisle(populated_graph):
    driveway = populated_graph.station_locations[0]
    assert populated_graph.get_same_aisle_locations(driveway) == []


# ---------------------------------------------------------------------------
# Aisle dual cycle: _find_aisle_dual_cycle_chain
# ---------------------------------------------------------------------------
def _empty_snapshot():
    """Convenience: no other agent has any task allocated yet."""
    return set(), set()


def test_aisle_dc_finds_same_column_outbound(populated_graph):
    """A pending OB whose pickup includes a same-column warehouse-full cell
    must be returned, with the chosen start in that column."""
    completed_goal = next(
        a for a in populated_graph.aisle_locations
        if a in populated_graph.warehouse.get_full_locations()
    )
    column = completed_goal[1]
    same_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] == column
    ]
    assert same_column_full, "fixture should expose at least one full cell in this aisle"

    J = {
        42: (
            frozenset(same_column_full),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100,                # deadline
            7,                  # sku id (not validated by the chain finder)
            TASK_TYPE_OUTBOUND,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_aisle_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is not None
    task_id, chosen_start, chosen_goal, deadline = result
    assert task_id == 42
    assert chosen_start[1] == column, "chosen start must be in the same aisle as completed_goal"
    assert chosen_start in same_column_full
    assert chosen_goal in populated_graph.driveway.get_empty_locations()
    assert deadline == 100


def test_aisle_dc_rejects_different_aisle_outbound(populated_graph):
    """A pending OB with no same-column pickup must be skipped."""
    completed_goal = populated_graph.aisle_locations[0]
    completed_column = completed_goal[1]
    other_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] != completed_column
    ]
    if not other_column_full:
        pytest.skip("populated_graph happens to have all full cells in one column")

    J = {
        42: (
            frozenset(other_column_full),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100, 1, TASK_TYPE_OUTBOUND,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_aisle_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is None


def test_aisle_dc_skips_already_allocated_task(populated_graph):
    """If the candidate OB task is already in some other agent's task_sequence,
    the chain finder must not re-assign it. The caller is responsible for
    populating the snapshot from the live ``Rs``."""
    completed_goal = next(
        a for a in populated_graph.aisle_locations
        if a in populated_graph.warehouse.get_full_locations()
    )
    column = completed_goal[1]
    same_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] == column
    ]
    chosen_start = same_column_full[0]
    chosen_goal = list(populated_graph.driveway.get_empty_locations())[0]

    J = {
        42: (
            frozenset(same_column_full),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100, 1, TASK_TYPE_OUTBOUND,
        ),
    }
    # Simulate that some other agent has already committed to task 42.
    allocated_task_ids = {42}
    allocated_locs = {chosen_start, chosen_goal}

    result = _find_aisle_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is None


def test_aisle_dc_returns_none_when_completed_goal_not_aisle(populated_graph):
    """The aisle DC finder must no-op if the tour endpoint isn't a warehouse
    aisle cell (e.g., it was actually an OB drop at a driveway)."""
    completed_goal = populated_graph.station_locations[0]
    full_cells = list(populated_graph.warehouse.get_full_locations())
    J = {
        42: (
            frozenset(full_cells),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100, 1, TASK_TYPE_OUTBOUND,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_aisle_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is None


def test_aisle_dc_ignores_inbound_and_shuffle_tasks(populated_graph):
    """Only OB tasks are valid chains for aisle DC. IB / shuffle entries in J
    must be filtered out by the helper."""
    completed_goal = next(
        a for a in populated_graph.aisle_locations
        if a in populated_graph.warehouse.get_full_locations()
    )
    column = completed_goal[1]
    same_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] == column
    ]
    J = {
        1: (
            frozenset(populated_graph.driveway.get_empty_locations()),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            100, 1, TASK_TYPE_INBOUND,
        ),
        2: (
            frozenset(same_column_full),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            100, 1, TASK_TYPE_SHUFFLE,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_aisle_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is None


# ---------------------------------------------------------------------------
# Driveway dual cycle: _find_driveway_dual_cycle_chain
# ---------------------------------------------------------------------------
def test_driveway_dc_finds_inbound_with_driveway_sku(populated_graph):
    """A pending IB task whose pickup is at a driveway cell with a pre-placed
    SKU must be returned."""
    driveway_loc_for_pickup = populated_graph.station_locations[1]
    populated_graph.driveway.add_sku_instance(1, driveway_loc_for_pickup)

    completed_goal = populated_graph.station_locations[0]
    J = {
        7: (
            frozenset({driveway_loc_for_pickup}),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            150, 1, TASK_TYPE_INBOUND,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_driveway_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is not None
    task_id, chosen_start, chosen_goal, deadline = result
    assert task_id == 7
    assert chosen_start == driveway_loc_for_pickup
    assert chosen_goal in populated_graph.warehouse.get_empty_locations()
    assert deadline == 150


def test_driveway_dc_returns_none_when_no_sku_at_pickup(populated_graph):
    """If the IB task's pickup driveway cell is empty, no chain can be made."""
    completed_goal = populated_graph.station_locations[0]
    empty_driveway = populated_graph.station_locations[1]
    J = {
        7: (
            frozenset({empty_driveway}),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            150, 1, TASK_TYPE_INBOUND,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_driveway_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is None


def test_driveway_dc_returns_none_when_completed_goal_not_driveway(populated_graph):
    """The driveway DC finder must no-op if the tour endpoint isn't a
    driveway cell."""
    completed_goal = populated_graph.aisle_locations[0]
    populated_graph.driveway.add_sku_instance(1, populated_graph.station_locations[0])
    J = {
        7: (
            frozenset({populated_graph.station_locations[0]}),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            150, 1, TASK_TYPE_INBOUND,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_driveway_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is None


def test_driveway_dc_ignores_outbound_and_shuffle_tasks(populated_graph):
    completed_goal = populated_graph.station_locations[0]
    populated_graph.driveway.add_sku_instance(1, populated_graph.station_locations[1])
    full_cells = list(populated_graph.warehouse.get_full_locations())
    J = {
        1: (
            frozenset(full_cells),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100, 1, TASK_TYPE_OUTBOUND,
        ),
        2: (
            frozenset(full_cells),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            100, 1, TASK_TYPE_SHUFFLE,
        ),
    }
    allocated_task_ids, allocated_locs = _empty_snapshot()

    result = _find_driveway_dual_cycle_chain(
        completed_goal, completed_goal, J, allocated_task_ids, allocated_locs, populated_graph
    )

    assert result is None


# ---------------------------------------------------------------------------
# Integration: apply_dual_cycle_pairing chains tails when the flag is on
# ---------------------------------------------------------------------------
def _make_loader_with_agent(state):
    agent = Agent(agent_id=0, state=state)
    return AgentLoader([agent]), agent


def _agent_with_ib_tail(populated_graph, ib_task_id=100):
    """Agent has just been assigned (by the allocator) an inbound task that
    drops off at a warehouse aisle cell. ``apply_dual_cycle_pairing`` should
    chain a same-aisle OB onto its tail when the aisle flag is on.

    Returns ``(Rs, agent, J, ib_drop_loc, ib_pickup_loc)``.
    """
    drop_loc = next(
        a for a in populated_graph.aisle_locations
        if a in populated_graph.warehouse.get_empty_locations()
    )
    pickup_loc = populated_graph.station_locations[0]
    populated_graph.driveway.add_sku_instance(1, pickup_loc)

    J = {
        ib_task_id: (
            frozenset({pickup_loc}),
            frozenset({drop_loc}),
            100, 1, TASK_TYPE_INBOUND,
        ),
    }

    Rs, agent = _make_loader_with_agent(state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(ib_task_id, pickup_loc, drop_loc, 100)]
    return Rs, agent, J, drop_loc, pickup_loc


def test_apply_pairing_chains_aisle_ob_when_flag_on(
    populated_graph, minimal_stats, seeded_rng
):
    """Agent's tail is an IB ending at a warehouse aisle cell. With
    ``aisle_dual_cycle=True`` and a same-aisle OB available in J,
    ``apply_dual_cycle_pairing`` must append that OB onto the tail."""
    Rs, agent, J, ib_drop_loc, ib_pickup_loc = _agent_with_ib_tail(populated_graph)

    column = ib_drop_loc[1]
    same_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] == column
    ]
    if not same_column_full:
        pytest.skip("populated_graph has no full cells in the IB drop's aisle")
    ob_task_id = 200
    J[ob_task_id] = (
        frozenset(same_column_full),
        frozenset(populated_graph.driveway.get_empty_locations()),
        200, 2, TASK_TYPE_OUTBOUND,
    )

    apply_dual_cycle_pairing(
        minimal_stats, populated_graph, Rs, J, t=0,
        aisle_dual_cycle=True, driveway_dual_cycle=False,
    )

    assert len(agent.task_sequence) == 2, "OB should be chained onto the IB tail"
    head_id, _, _, _ = agent.task_sequence[0]
    tail_id, tail_start, tail_goal, _ = agent.task_sequence[1]
    assert head_id == 100
    assert tail_id == ob_task_id
    assert tail_start[1] == column, "chained pickup must be in the same aisle"
    assert tail_goal in populated_graph.driveway.get_empty_locations()


def test_apply_pairing_does_not_chain_when_aisle_flag_off(
    populated_graph, minimal_stats, seeded_rng
):
    """Same setup as above but with ``aisle_dual_cycle=False``: the agent's
    task_sequence must be untouched."""
    Rs, agent, J, ib_drop_loc, _ = _agent_with_ib_tail(populated_graph)

    column = ib_drop_loc[1]
    same_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] == column
    ]
    if not same_column_full:
        pytest.skip("populated_graph has no full cells in the IB drop's aisle")
    ob_task_id = 200
    J[ob_task_id] = (
        frozenset(same_column_full),
        frozenset(populated_graph.driveway.get_empty_locations()),
        200, 2, TASK_TYPE_OUTBOUND,
    )

    apply_dual_cycle_pairing(
        minimal_stats, populated_graph, Rs, J, t=0,
        aisle_dual_cycle=False, driveway_dual_cycle=False,
    )

    assert len(agent.task_sequence) == 1, "no chaining when both flags are off"


def _agent_with_ob_tail(populated_graph, ob_task_id=100):
    """Agent has been assigned an outbound that drops off at a driveway cell.
    ``apply_dual_cycle_pairing`` should chain a same-driveway IB onto its
    tail when the driveway flag is on.

    Returns ``(Rs, agent, J, ob_drop_loc, ob_pickup_loc, ob_sku)``.
    """
    pickup_loc = next(
        a for a in populated_graph.aisle_locations
        if a in populated_graph.warehouse.get_full_locations()
    )
    pickup_sku = populated_graph.warehouse.get_sku_at_location(pickup_loc).sku_id
    drop_loc = populated_graph.station_locations[0]

    J = {
        ob_task_id: (
            frozenset({pickup_loc}),
            frozenset({drop_loc}),
            100, pickup_sku, TASK_TYPE_OUTBOUND,
        ),
    }

    Rs, agent = _make_loader_with_agent(state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(ob_task_id, pickup_loc, drop_loc, 100)]
    return Rs, agent, J, drop_loc, pickup_loc, pickup_sku


def test_apply_pairing_chains_driveway_ib_when_flag_on(
    populated_graph, minimal_stats, seeded_rng
):
    """Agent's tail is an OB ending at a driveway cell. With
    ``driveway_dual_cycle=True`` and a driveway IB available in J,
    ``apply_dual_cycle_pairing`` must append that IB onto the tail."""
    Rs, agent, J, _, _, _ = _agent_with_ob_tail(populated_graph)

    ib_pickup = populated_graph.station_locations[1]
    populated_graph.driveway.add_sku_instance(3, ib_pickup)
    ib_task_id = 200
    J[ib_task_id] = (
        frozenset({ib_pickup}),
        frozenset(populated_graph.warehouse.get_empty_locations()),
        200, 3, TASK_TYPE_INBOUND,
    )

    apply_dual_cycle_pairing(
        minimal_stats, populated_graph, Rs, J, t=0,
        aisle_dual_cycle=False, driveway_dual_cycle=True,
    )

    assert len(agent.task_sequence) == 2
    head_id, _, _, _ = agent.task_sequence[0]
    tail_id, tail_start, tail_goal, _ = agent.task_sequence[1]
    assert head_id == 100
    assert tail_id == ib_task_id
    assert tail_start == ib_pickup, "chained IB must pick up at the driveway with the pre-placed SKU"
    assert tail_goal in populated_graph.warehouse.get_empty_locations()


def test_apply_pairing_does_not_chain_when_driveway_flag_off(
    populated_graph, minimal_stats, seeded_rng
):
    Rs, agent, J, _, _, _ = _agent_with_ob_tail(populated_graph)

    ib_pickup = populated_graph.station_locations[1]
    populated_graph.driveway.add_sku_instance(3, ib_pickup)
    ib_task_id = 200
    J[ib_task_id] = (
        frozenset({ib_pickup}),
        frozenset(populated_graph.warehouse.get_empty_locations()),
        200, 3, TASK_TYPE_INBOUND,
    )

    apply_dual_cycle_pairing(
        minimal_stats, populated_graph, Rs, J, t=0,
        aisle_dual_cycle=False, driveway_dual_cycle=False,
    )

    assert len(agent.task_sequence) == 1


def test_apply_pairing_idle_agent_not_chained(populated_graph, minimal_stats):
    """An agent with empty task_sequence (allocator gave them no work) must
    be left alone -- there's no tail to chain onto."""
    Rs, agent = _make_loader_with_agent(state=populated_graph.station_locations[0])
    populated_graph.driveway.add_sku_instance(1, populated_graph.station_locations[1])
    J = {
        1: (
            frozenset({populated_graph.station_locations[1]}),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            100, 1, TASK_TYPE_INBOUND,
        ),
    }

    apply_dual_cycle_pairing(
        minimal_stats, populated_graph, Rs, J, t=0,
        aisle_dual_cycle=True, driveway_dual_cycle=True,
    )

    assert agent.task_sequence == []


def test_apply_pairing_other_agents_committed_block_chain(
    populated_graph, minimal_stats
):
    """If another agent already has the candidate OB committed in their
    own task_sequence, ``apply_dual_cycle_pairing`` must not re-allocate
    it onto a same-aisle IB tail of a second agent."""
    Rs, ib_agent, J, ib_drop_loc, _ = _agent_with_ib_tail(populated_graph)

    column = ib_drop_loc[1]
    same_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] == column
    ]
    if not same_column_full:
        pytest.skip("populated_graph has no full cells in the IB drop's aisle")
    chosen_start = same_column_full[0]
    chosen_goal = list(populated_graph.driveway.get_empty_locations())[0]

    other_agent = Agent(agent_id=1, state=(chosen_start[0], chosen_start[1]))
    other_agent.task_sequence = [(200, chosen_start, chosen_goal, 200)]
    Rs.agents.append(other_agent)

    J[200] = (
        frozenset(same_column_full),
        frozenset(populated_graph.driveway.get_empty_locations()),
        200, 2, TASK_TYPE_OUTBOUND,
    )

    apply_dual_cycle_pairing(
        minimal_stats, populated_graph, Rs, J, t=0,
        aisle_dual_cycle=True, driveway_dual_cycle=False,
    )

    assert len(ib_agent.task_sequence) == 1, "OB already taken by other agent must not chain"


def test_collect_global_allocation_snapshot_aggregates_all_agents():
    a0 = Agent(agent_id=0, state=(0, 0))
    a0.task_sequence = [(1, (5, 5), (6, 6), 100)]
    a1 = Agent(agent_id=1, state=(0, 1))
    a1.task_sequence = [(2, (7, 7), (8, 8), 100), (3, (9, 9), (10, 10), 100)]
    Rs = AgentLoader([a0, a1])

    task_ids, locs = _collect_global_allocation_snapshot(Rs)

    assert task_ids == {1, 2, 3}
    assert locs == {(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)}
