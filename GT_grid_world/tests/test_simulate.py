"""Unit tests for ``simulate.py`` -- focused on the type=2 (shuffle) skeleton
added in roadmap section 1.4.

The pre-existing IB/OB pickup and dropoff branches dispatch by *location*
(warehouse vs driveway), not by task type, so the legacy code already lifts a
SKU off a warehouse cell and drops it onto another warehouse cell when it
sees a shuffle task. The only new logic in this section is the
``_refresh_tasks_after_warehouse_change`` extension that keeps in-flight
tasks consistent with warehouse occupancy after a shuffle pickup or dropoff.
That is what these tests cover most carefully.

A smoke test at the bottom drives a full pickup+dropoff cycle through the
real ``simulate`` loop to confirm the location-dispatched code path actually
handles a shuffle end-to-end without raising.
"""

from __future__ import annotations

import pytest

from GT_grid_world.src.simulate import (
    TASK_TYPE_INBOUND,
    TASK_TYPE_OUTBOUND,
    TASK_TYPE_SHUFFLE,
    WAREHOUSE_DROPOFF_TASK_TYPES,
    WAREHOUSE_PICKUP_TASK_TYPES,
    _refresh_tasks_after_warehouse_change,
    simulate,
)


# ---------------------------------------------------------------------------
# Task type membership
# ---------------------------------------------------------------------------
def test_pickup_task_type_set_includes_outbound_and_shuffle():
    assert TASK_TYPE_OUTBOUND in WAREHOUSE_PICKUP_TASK_TYPES
    assert TASK_TYPE_SHUFFLE in WAREHOUSE_PICKUP_TASK_TYPES
    assert TASK_TYPE_INBOUND not in WAREHOUSE_PICKUP_TASK_TYPES


def test_dropoff_task_type_set_includes_inbound_and_shuffle():
    assert TASK_TYPE_INBOUND in WAREHOUSE_DROPOFF_TASK_TYPES
    assert TASK_TYPE_SHUFFLE in WAREHOUSE_DROPOFF_TASK_TYPES
    assert TASK_TYPE_OUTBOUND not in WAREHOUSE_DROPOFF_TASK_TYPES


# ---------------------------------------------------------------------------
# _refresh_tasks_after_warehouse_change -- shuffle (type=2) cases
# ---------------------------------------------------------------------------
def test_refresh_no_op_when_J_is_empty(populated_graph):
    """The function must not raise on the empty-J fast path."""
    J: dict = {}
    _refresh_tasks_after_warehouse_change(J, populated_graph, changed_task_id=99, sku_id=1)
    assert J == {}


def test_refresh_skips_the_changed_task(populated_graph):
    """The task whose pickup or dropoff triggered the refresh must be left
    untouched -- ``simulate`` is the one rewriting that one explicitly."""
    sentinel_starts = frozenset({(0, 0)})
    sentinel_goals = frozenset({(0, 1)})
    J = {
        7: (sentinel_starts, sentinel_goals, 100, 1, TASK_TYPE_SHUFFLE),
    }

    _refresh_tasks_after_warehouse_change(J, populated_graph, changed_task_id=7, sku_id=1)

    assert J[7] == (sentinel_starts, sentinel_goals, 100, 1, TASK_TYPE_SHUFFLE)


def test_refresh_updates_same_sku_shuffle_start_locs_after_pickup(populated_graph):
    """A shuffle pickup of SKU s removes one warehouse instance of s. Other
    in-flight shuffle tasks for the same SKU must see their start_locs shrink
    accordingly (re-derived from current SKU instances).
    """
    target_sku = next(
        sku for sku in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(sku) >= 2
    )
    instances_before = list(populated_graph.warehouse.get_sku_instances(target_sku))
    pickup_loc = instances_before[0]

    other_shuffle_id = 1
    stale_starts = frozenset(instances_before)
    stale_goals = frozenset(populated_graph.warehouse.get_empty_locations())
    J = {
        other_shuffle_id: (stale_starts, stale_goals, 100, target_sku, TASK_TYPE_SHUFFLE),
    }
    populated_graph.warehouse.remove_sku_instance(pickup_loc)

    _refresh_tasks_after_warehouse_change(J, populated_graph, changed_task_id=999, sku_id=target_sku)

    new_starts, new_goals, _, _, _ = J[other_shuffle_id]
    assert new_starts == frozenset(populated_graph.warehouse.get_sku_instances(target_sku))
    assert pickup_loc not in new_starts
    # The pickup location is now empty, so it should appear in the refreshed
    # goal_locs (any shuffle can drop a different SKU there).
    assert pickup_loc in new_goals


def test_refresh_updates_other_sku_shuffle_goal_locs(populated_graph):
    """A shuffle pickup or dropoff that changes warehouse occupancy must
    update goal_locs of *other-SKU* shuffle tasks -- their start_locs do not
    change, but their goal_locs (warehouse-empty cells) shift.
    """
    skus = list(populated_graph.warehouse.get_all_skus().keys())
    sku_a, sku_b = skus[0], skus[1]
    a_locs = list(populated_graph.warehouse.get_sku_instances(sku_a))
    b_locs = list(populated_graph.warehouse.get_sku_instances(sku_b))
    if not a_locs or not b_locs:
        pytest.skip("populated_graph fixture didn't yield both target SKUs")
    pickup_loc_a = a_locs[0]

    stale_b_starts = frozenset(b_locs)
    stale_empty = frozenset(populated_graph.warehouse.get_empty_locations())
    J = {
        2: (stale_b_starts, stale_empty, 100, sku_b, TASK_TYPE_SHUFFLE),
    }
    populated_graph.warehouse.remove_sku_instance(pickup_loc_a)

    _refresh_tasks_after_warehouse_change(J, populated_graph, changed_task_id=999, sku_id=sku_a)

    new_starts, new_goals, _, _, _ = J[2]
    assert new_starts == stale_b_starts, "other-SKU starts should not change after a different-SKU pickup"
    assert pickup_loc_a in new_goals, "the freshly-emptied cell should appear in refreshed goal_locs"


def test_refresh_outbound_task_goal_locs_track_driveway_empty(populated_graph):
    """Outbound (type=0) tasks dropoff at the driveway, so their goal_locs
    are driveway-empty cells. The refresh must leave them tracking driveway
    state, not warehouse state.
    """
    sku = next(iter(populated_graph.warehouse.get_all_skus()))
    instances = list(populated_graph.warehouse.get_sku_instances(sku))
    if not instances:
        pytest.skip("populated_graph has no SKU instances for this test")

    stale_starts = frozenset(instances)
    stale_goals = frozenset({(0, 0)})  # deliberately wrong, refresh should fix
    J = {
        5: (stale_starts, stale_goals, 100, sku, TASK_TYPE_OUTBOUND),
    }

    _refresh_tasks_after_warehouse_change(J, populated_graph, changed_task_id=999, sku_id=sku)

    _, new_goals, _, _, _ = J[5]
    expected_goals = frozenset(populated_graph.driveway.get_empty_locations())
    assert new_goals == expected_goals


def test_refresh_inbound_task_goal_locs_track_warehouse_empty(populated_graph):
    """Inbound (type=1) tasks dropoff into the warehouse, so their goal_locs
    must be re-derived from warehouse-empty cells after any change."""
    stale_starts = frozenset({(0, 0)})  # IB starts come from driveway; we don't refresh them here
    stale_goals = frozenset({(0, 1)})    # deliberately wrong
    J = {
        9: (stale_starts, stale_goals, 100, 1, TASK_TYPE_INBOUND),
    }

    _refresh_tasks_after_warehouse_change(J, populated_graph, changed_task_id=999, sku_id=1)

    new_starts, new_goals, _, _, _ = J[9]
    assert new_starts == stale_starts, "IB start_locs (driveway side) are not refreshed by warehouse changes"
    assert new_goals == frozenset(populated_graph.warehouse.get_empty_locations())


# ---------------------------------------------------------------------------
# End-to-end shuffle pickup + dropoff smoke test
# ---------------------------------------------------------------------------
def test_simulate_completes_shuffle_pickup_and_dropoff_end_to_end(
    populated_graph, minimal_stats, seeded_rng
):
    """Drive a single shuffle task through the real ``simulate`` loop end to
    end and assert the warehouse, agent, and J state look right at each step.

    This is a smoke test, not a fine-grained unit test -- the goal is to
    confirm that the pre-existing location-dispatched pickup and dropoff
    branches work for a shelf-to-shelf task without raising.
    """
    from GT_grid_world.src.agent import Agent, AgentLoader

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]
    assert pickup_loc != goal_loc

    task_id = 1
    deadline = 100
    J = {
        task_id: (
            frozenset({pickup_loc}),
            frozenset({goal_loc}),
            deadline,
            sku,
            TASK_TYPE_SHUFFLE,
        ),
    }
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    agent.path_sequence = []
    Rs = AgentLoader([agent])

    populated_graph.set_occupied(pickup_loc, True)

    Rs, J = simulate(minimal_stats, populated_graph, Rs, J, "small_test", t=0)

    assert agent.status == 2, "agent should transition to delivery phase after pickup"
    assert agent.get_sku_id_carrying() == sku, "agent should be carrying the picked SKU"
    assert pickup_loc in populated_graph.warehouse.get_empty_locations(), \
        "pickup location should be empty after pickup"
    assert task_id in J, "task should not be popped until dropoff completes"

    agent.state = goal_loc
    agent.path_sequence = []

    Rs, J = simulate(minimal_stats, populated_graph, Rs, J, "small_test", t=1)

    assert agent.status == 0, "agent should be idle after dropoff (no remaining tasks)"
    assert agent.get_sku_id_carrying() is None, "agent should not be carrying anything after dropoff"
    assert goal_loc in populated_graph.warehouse.get_full_locations(), \
        "goal location should now hold the dropped SKU"
    placed = populated_graph.warehouse.get_sku_at_location(goal_loc)
    assert placed is not None and placed.sku_id == sku, "goal cell should hold the original SKU"
    assert task_id not in J, "task should be removed from J after dropoff"
