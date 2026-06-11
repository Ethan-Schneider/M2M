"""Unit and integration tests for dual cycling (roadmap section 1.5).

Covers:

* Graph aisle / driveway membership helpers (``is_warehouse_aisle_location``,
  ``is_driveway_location``, ``get_same_aisle_locations``).
* The two ``_find_*_dual_cycle_chain`` helpers in ``simulate.py`` -- positive
  match, negative match (different aisle / no driveway SKU), already-allocated
  rejection, wrong completed-task-type rejection.
* End-to-end: the dual-cycle hook fires inside ``simulate()`` only when the
  matching CLI flag is on, and never when the flag is off.

The tests use the ``populated_graph`` fixture (30%-filled warehouse) so both
warehouse-full and warehouse-empty cells are available, which the dual-cycle
chain finder needs in order to pick valid start / goal cells for the chained
task.
"""

from __future__ import annotations

import pytest

from GT_grid_world.src.agent import Agent, AgentLoader
from GT_grid_world.src.simulate import (
    TASK_TYPE_INBOUND,
    TASK_TYPE_OUTBOUND,
    TASK_TYPE_SHUFFLE,
    _find_aisle_dual_cycle_chain,
    _find_driveway_dual_cycle_chain,
    simulate,
)


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
def _make_loader_with_agent(state):
    agent = Agent(agent_id=0, state=state)
    return AgentLoader([agent]), agent


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

    Rs, agent = _make_loader_with_agent(state=completed_goal)
    J = {
        42: (
            frozenset(same_column_full),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100,                # deadline
            7,                  # sku id (not validated by the chain finder)
            TASK_TYPE_OUTBOUND,
        ),
    }

    result = _find_aisle_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

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

    Rs, agent = _make_loader_with_agent(state=completed_goal)
    J = {
        42: (
            frozenset(other_column_full),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100, 1, TASK_TYPE_OUTBOUND,
        ),
    }

    result = _find_aisle_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

    assert result is None


def test_aisle_dc_skips_already_allocated_task(populated_graph):
    """If the candidate OB task is already in some other agent's task_sequence,
    the chain finder must not re-assign it."""
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

    main_agent = Agent(agent_id=0, state=completed_goal)
    other_agent = Agent(agent_id=1, state=(17, 0))
    other_agent.task_sequence = [(42, chosen_start, chosen_goal, 100)]
    Rs = AgentLoader([main_agent, other_agent])

    J = {
        42: (
            frozenset(same_column_full),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100, 1, TASK_TYPE_OUTBOUND,
        ),
    }

    result = _find_aisle_dual_cycle_chain(main_agent, completed_goal, J, Rs, populated_graph)

    assert result is None


def test_aisle_dc_returns_none_when_completed_goal_not_aisle(populated_graph):
    """The aisle DC hook must no-op if the just-completed dropoff isn't a
    warehouse aisle cell (e.g., it was actually an OB drop at a driveway)."""
    completed_goal = populated_graph.station_locations[0]
    Rs, agent = _make_loader_with_agent(state=completed_goal)
    full_cells = list(populated_graph.warehouse.get_full_locations())
    J = {
        42: (
            frozenset(full_cells),
            frozenset(populated_graph.driveway.get_empty_locations()),
            100, 1, TASK_TYPE_OUTBOUND,
        ),
    }

    result = _find_aisle_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

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
    Rs, agent = _make_loader_with_agent(state=completed_goal)
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

    result = _find_aisle_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

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
    Rs, agent = _make_loader_with_agent(state=completed_goal)
    J = {
        7: (
            frozenset({driveway_loc_for_pickup}),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            150, 1, TASK_TYPE_INBOUND,
        ),
    }

    result = _find_driveway_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

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
    Rs, agent = _make_loader_with_agent(state=completed_goal)
    J = {
        7: (
            frozenset({empty_driveway}),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            150, 1, TASK_TYPE_INBOUND,
        ),
    }

    result = _find_driveway_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

    assert result is None


def test_driveway_dc_returns_none_when_completed_goal_not_driveway(populated_graph):
    """The driveway DC hook must no-op if the just-completed dropoff isn't a
    driveway cell."""
    completed_goal = populated_graph.aisle_locations[0]
    Rs, agent = _make_loader_with_agent(state=completed_goal)
    populated_graph.driveway.add_sku_instance(1, populated_graph.station_locations[0])
    J = {
        7: (
            frozenset({populated_graph.station_locations[0]}),
            frozenset(populated_graph.warehouse.get_empty_locations()),
            150, 1, TASK_TYPE_INBOUND,
        ),
    }

    result = _find_driveway_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

    assert result is None


def test_driveway_dc_ignores_outbound_and_shuffle_tasks(populated_graph):
    completed_goal = populated_graph.station_locations[0]
    populated_graph.driveway.add_sku_instance(1, populated_graph.station_locations[1])
    Rs, agent = _make_loader_with_agent(state=completed_goal)
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

    result = _find_driveway_dual_cycle_chain(agent, completed_goal, J, Rs, populated_graph)

    assert result is None


# ---------------------------------------------------------------------------
# Integration: simulate() honors the on/off flags and chains the next task
# ---------------------------------------------------------------------------
def _setup_ib_dropoff_state(populated_graph, minimal_stats, *, sku=1):
    """Place an agent in the delivery phase of an IB task at a warehouse
    aisle cell. Returns ``(Rs, agent, ib_task_id, drop_loc, J)``.
    """
    drop_loc = next(
        a for a in populated_graph.aisle_locations
        if a in populated_graph.warehouse.get_empty_locations()
    )
    pickup_loc = populated_graph.station_locations[0]
    populated_graph.driveway.add_sku_instance(sku, pickup_loc)

    ib_task_id = 100
    deadline = 100
    J = {
        ib_task_id: (
            frozenset({pickup_loc}),
            frozenset({drop_loc}),
            deadline, sku, TASK_TYPE_INBOUND,
        ),
    }
    minimal_stats.add_task_release(ib_task_id, 0)
    minimal_stats.add_task_deadline(ib_task_id, deadline)
    minimal_stats.add_actual_distance(ib_task_id)
    minimal_stats.add_actual_pickup_distance(ib_task_id)
    minimal_stats.add_actual_duration(ib_task_id)
    minimal_stats.add_actual_pickup_duration(ib_task_id)

    agent = Agent(agent_id=0, state=drop_loc)
    agent.status = 2
    agent.task_sequence = [(ib_task_id, pickup_loc, drop_loc, deadline)]
    agent.path_sequence = []
    agent.set_sku_id_carrying(sku)
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(drop_loc, True)
    return Rs, agent, ib_task_id, drop_loc, J


def test_simulate_aisle_dual_cycle_chains_when_flag_on(populated_graph, minimal_stats, seeded_rng):
    """With ``aisle_dual_cycle=True``, completing an IB at an aisle cell that
    has a same-aisle OB candidate in J must leave the agent with that OB
    chained onto its task_sequence (status=1, ready to pick it up)."""
    Rs, agent, ib_task_id, drop_loc, J = _setup_ib_dropoff_state(populated_graph, minimal_stats)

    column = drop_loc[1]
    same_column_full = [
        loc for loc in populated_graph.warehouse.get_full_locations()
        if loc[1] == column
    ]
    if not same_column_full:
        pytest.skip("populated_graph has no full cells in the IB drop's aisle")
    ob_task_id = 200
    ob_deadline = 200
    J[ob_task_id] = (
        frozenset(same_column_full),
        frozenset(populated_graph.driveway.get_empty_locations()),
        ob_deadline, 2, TASK_TYPE_OUTBOUND,
    )
    minimal_stats.add_task_release(ob_task_id, 0)
    minimal_stats.add_task_deadline(ob_task_id, ob_deadline)

    Rs, J = simulate(
        minimal_stats, populated_graph, Rs, J, "small_test", t=0,
        aisle_dual_cycle=True, driveway_dual_cycle=False,
    )

    assert ib_task_id not in J, "IB task should be popped after dropoff"
    assert ob_task_id in J, "OB task remains in J until simulate completes it"
    assert agent.status == 1, "agent should be in pickup phase of the chained OB"
    assert len(agent.task_sequence) == 1, "exactly one chained task on the sequence"
    chained = agent.task_sequence[0]
    assert chained[0] == ob_task_id
    assert chained[1][1] == column, "chained pickup must be in the same aisle"


def test_simulate_aisle_dual_cycle_does_not_chain_when_flag_off(
    populated_graph, minimal_stats, seeded_rng
):
    """With ``aisle_dual_cycle=False``, the same setup must leave the agent
    idle (status=0) instead of chaining the OB."""
    Rs, agent, ib_task_id, drop_loc, J = _setup_ib_dropoff_state(populated_graph, minimal_stats)

    column = drop_loc[1]
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
    minimal_stats.add_task_release(ob_task_id, 0)
    minimal_stats.add_task_deadline(ob_task_id, 200)

    Rs, J = simulate(
        minimal_stats, populated_graph, Rs, J, "small_test", t=0,
        aisle_dual_cycle=False, driveway_dual_cycle=False,
    )

    assert ib_task_id not in J
    assert ob_task_id in J, "OB stays unallocated when the flag is off"
    assert agent.status == 0, "agent should be idle when dual cycling is disabled"
    assert agent.task_sequence == []


def _setup_ob_dropoff_state(populated_graph, minimal_stats, *, sku=1):
    """Place an agent in the delivery phase of an OB task at a driveway cell.
    Returns ``(Rs, agent, ob_task_id, drop_loc, J)``.
    """
    pickup_loc = next(
        a for a in populated_graph.aisle_locations
        if a in populated_graph.warehouse.get_full_locations()
    )
    pickup_sku = populated_graph.warehouse.get_sku_at_location(pickup_loc).sku_id
    drop_loc = populated_graph.station_locations[0]

    ob_task_id = 100
    deadline = 100
    J = {
        ob_task_id: (
            frozenset({pickup_loc}),
            frozenset({drop_loc}),
            deadline, pickup_sku, TASK_TYPE_OUTBOUND,
        ),
    }
    minimal_stats.add_task_release(ob_task_id, 0)
    minimal_stats.add_task_deadline(ob_task_id, deadline)
    minimal_stats.add_actual_distance(ob_task_id)
    minimal_stats.add_actual_pickup_distance(ob_task_id)
    minimal_stats.add_actual_duration(ob_task_id)
    minimal_stats.add_actual_pickup_duration(ob_task_id)

    agent = Agent(agent_id=0, state=drop_loc)
    agent.status = 2
    agent.task_sequence = [(ob_task_id, pickup_loc, drop_loc, deadline)]
    agent.path_sequence = []
    agent.set_sku_id_carrying(pickup_sku)
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(drop_loc, True)
    return Rs, agent, ob_task_id, drop_loc, J


def test_simulate_driveway_dual_cycle_chains_when_flag_on(
    populated_graph, minimal_stats, seeded_rng
):
    """OB delivery at a driveway cell with a pending IB whose pickup is at
    another driveway cell holding a SKU should chain that IB."""
    Rs, agent, ob_task_id, drop_loc, J = _setup_ob_dropoff_state(populated_graph, minimal_stats)

    ib_pickup = populated_graph.station_locations[1]
    populated_graph.driveway.add_sku_instance(3, ib_pickup)
    ib_task_id = 200
    ib_deadline = 200
    J[ib_task_id] = (
        frozenset({ib_pickup}),
        frozenset(populated_graph.warehouse.get_empty_locations()),
        ib_deadline, 3, TASK_TYPE_INBOUND,
    )
    minimal_stats.add_task_release(ib_task_id, 0)
    minimal_stats.add_task_deadline(ib_task_id, ib_deadline)

    Rs, J = simulate(
        minimal_stats, populated_graph, Rs, J, "small_test", t=0,
        aisle_dual_cycle=False, driveway_dual_cycle=True,
    )

    assert ob_task_id not in J
    assert ib_task_id in J
    assert agent.status == 1
    assert len(agent.task_sequence) == 1
    chained = agent.task_sequence[0]
    assert chained[0] == ib_task_id
    assert chained[1] == ib_pickup, "chained IB must pick up at the driveway with the pre-placed SKU"


def test_simulate_driveway_dual_cycle_does_not_chain_when_flag_off(
    populated_graph, minimal_stats, seeded_rng
):
    Rs, agent, ob_task_id, drop_loc, J = _setup_ob_dropoff_state(populated_graph, minimal_stats)

    ib_pickup = populated_graph.station_locations[1]
    populated_graph.driveway.add_sku_instance(3, ib_pickup)
    ib_task_id = 200
    J[ib_task_id] = (
        frozenset({ib_pickup}),
        frozenset(populated_graph.warehouse.get_empty_locations()),
        200, 3, TASK_TYPE_INBOUND,
    )
    minimal_stats.add_task_release(ib_task_id, 0)
    minimal_stats.add_task_deadline(ib_task_id, 200)

    Rs, J = simulate(
        minimal_stats, populated_graph, Rs, J, "small_test", t=0,
        aisle_dual_cycle=False, driveway_dual_cycle=False,
    )

    assert ob_task_id not in J
    assert ib_task_id in J
    assert agent.status == 0
    assert agent.task_sequence == []
