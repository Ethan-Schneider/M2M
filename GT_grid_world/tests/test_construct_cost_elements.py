"""Unit tests for ``construct_cost_elements`` -- focused on the type=2
(shuffle) extension added in roadmap section 1.4.

The 1.4 skeleton routes shuffle tasks through the same ``outbound``-style
SKU-distribution cost path as type=0 outbound tasks (because both pick up
from a warehouse SKU instance) AND populates the inbound-style cost matrix
for shuffle tasks (because shuffle drops off in the warehouse). The proper
rearrangement-specific objective lives in roadmap section 1.6.

These tests verify:
* shapes are correct in the presence of mixed type 0/1/2 tasks
* shuffle tasks have *finite* outbound_sku_distribution_costs at their
  start_locs (so allocation does not silently exclude them)
* shuffle tasks have *finite* inbound_sku_distribution_costs at their
  goal_locs (the second axis of the dual-cost story)
* the existing IB / OB cost contracts still hold (no regression)
"""

from __future__ import annotations

import numpy as np
import pytest

from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
    TASK_TYPE_INBOUND,
    TASK_TYPE_OUTBOUND,
    TASK_TYPE_SHUFFLE,
    WAREHOUSE_DROPOFF_TASK_TYPES,
    WAREHOUSE_PICKUP_TASK_TYPES,
    calculate_deadline_cost,
    construct_cost_elements,
)


# ---------------------------------------------------------------------------
# Task type membership constants
# ---------------------------------------------------------------------------
def test_task_type_constants_match_inventory_dispatch():
    """The cost-tensor module's WAREHOUSE_*_TASK_TYPES sets must agree with
    simulate.py's so the allocator and simulator never disagree about which
    task type touches which inventory."""
    from GT_grid_world.src import simulate

    assert WAREHOUSE_PICKUP_TASK_TYPES == simulate.WAREHOUSE_PICKUP_TASK_TYPES
    assert WAREHOUSE_DROPOFF_TASK_TYPES == simulate.WAREHOUSE_DROPOFF_TASK_TYPES


# ---------------------------------------------------------------------------
# calculate_deadline_cost
# ---------------------------------------------------------------------------
def test_calculate_deadline_cost_zero_when_far_in_future():
    assert calculate_deadline_cost(deadline=200, current_time=0) == 0


def test_calculate_deadline_cost_increases_as_deadline_approaches():
    far = calculate_deadline_cost(deadline=50, current_time=0)
    near = calculate_deadline_cost(deadline=10, current_time=0)
    assert near > far >= 0


def test_calculate_deadline_cost_penalizes_overdue():
    overdue = calculate_deadline_cost(deadline=-5, current_time=0)
    on_time = calculate_deadline_cost(deadline=0, current_time=0)
    assert overdue > on_time


# ---------------------------------------------------------------------------
# construct_cost_elements -- mixed type 0/1/2 tasks
# ---------------------------------------------------------------------------
@pytest.fixture()
def mixed_type_tasks(populated_graph):
    """Build one OB (type=0), one IB (type=1), and one shuffle (type=2)
    task against the populated graph. Returns (J, sku_for_ob_and_shuffle).
    """
    G = populated_graph
    sku = next(
        s for s in G.warehouse.get_all_skus()
        if G.warehouse.get_sku_instance_count(s) >= 1
    )
    full = G.warehouse.get_sku_instances(sku)
    empty = G.warehouse.get_empty_locations()
    stations = G.get_station_locations()
    if not full or not empty or not stations:
        pytest.skip("populated_graph does not have enough cells for mixed-type test")

    J = {
        1: (frozenset(full), frozenset(stations), 100, sku, TASK_TYPE_OUTBOUND),
        2: (frozenset(stations), frozenset(empty), 100, sku, TASK_TYPE_INBOUND),
        3: (frozenset(full), frozenset(empty), 100, sku, TASK_TYPE_SHUFFLE),
    }
    return J, sku


def test_construct_cost_elements_returns_correct_shapes(
    mixed_type_tasks, populated_graph, sample_agents
):
    J, _ = mixed_type_tasks
    M = len(sample_agents.agents)
    N = len(J)

    (
        agent_start_cost_tensor,
        start_goal_dist,
        task_start_mask,
        task_goal_mask,
        start_locs,
        goal_locs,
        idx_to_task_id,
        task_id_to_idx,
        task_deadline_costs,
        inbound_sku_distribution_costs,
        outbound_sku_distribution_costs,
        agent_task_sequence_time,
    ) = construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    P = len(start_locs)
    Q = len(goal_locs)

    assert agent_start_cost_tensor.shape == (M, P)
    assert start_goal_dist.shape == (P, Q)
    assert task_start_mask.shape == (N, P)
    assert task_goal_mask.shape == (N, Q)
    assert task_deadline_costs.shape == (N,)
    assert inbound_sku_distribution_costs.shape == (N, Q)
    assert outbound_sku_distribution_costs.shape == (N, P)
    assert agent_task_sequence_time.shape == (M,)


def test_construct_cost_elements_shuffle_has_finite_outbound_cost(
    mixed_type_tasks, populated_graph, sample_agents
):
    """Shuffle tasks must have at least one finite entry in the outbound
    SKU-distribution cost matrix, otherwise fast_greedy would refuse to
    allocate them when ``sku_distribution_weight > 0``.
    """
    J, _ = mixed_type_tasks

    (_, _, _, _, _, _, _, task_id_to_idx, _, inbound_costs, outbound_costs, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    shuffle_idx = task_id_to_idx[3]
    shuffle_outbound = outbound_costs[shuffle_idx, :]
    assert np.isfinite(shuffle_outbound).any(), (
        "expected shuffle (type=2) task to have at least one finite entry in "
        "outbound_sku_distribution_costs; otherwise fast_greedy will skip it"
    )


def test_construct_cost_elements_shuffle_has_finite_inbound_cost(
    mixed_type_tasks, populated_graph, sample_agents
):
    """Shuffle tasks drop off in the warehouse so they should also have at
    least one finite entry in the inbound-style cost matrix. The 1.4
    skeleton populates this even though fast_greedy currently routes
    shuffle through the outbound branch -- 1.6 is where the dual-cost
    objective lights up.
    """
    J, _ = mixed_type_tasks

    (_, _, _, _, _, _, _, task_id_to_idx, _, inbound_costs, outbound_costs, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    shuffle_idx = task_id_to_idx[3]
    shuffle_inbound = inbound_costs[shuffle_idx, :]
    assert np.isfinite(shuffle_inbound).any(), (
        "expected shuffle (type=2) task to have at least one finite entry in "
        "inbound_sku_distribution_costs; needed for the 1.6 dual-cost "
        "rearrangement objective"
    )


def test_construct_cost_elements_outbound_cost_only_populates_warehouse_pickup_types(
    mixed_type_tasks, populated_graph, sample_agents
):
    """Inbound (type=1) tasks pick up from the driveway, not the warehouse,
    so their row in ``outbound_sku_distribution_costs`` must remain entirely
    np.inf (the legacy contract). Without this guarantee, fast_greedy would
    let an inbound task be allocated using a warehouse-SKU-distance cost,
    which is meaningless.
    """
    J, _ = mixed_type_tasks

    (_, _, _, _, _, _, _, task_id_to_idx, _, _, outbound_costs, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    inbound_idx = task_id_to_idx[2]
    assert np.isinf(outbound_costs[inbound_idx, :]).all(), \
        "inbound tasks must not appear in outbound_sku_distribution_costs"


def test_construct_cost_elements_inbound_cost_only_populates_warehouse_dropoff_types(
    mixed_type_tasks, populated_graph, sample_agents
):
    """Outbound (type=0) tasks drop off at the driveway, not the warehouse,
    so their row in ``inbound_sku_distribution_costs`` must remain entirely
    np.inf.
    """
    J, _ = mixed_type_tasks

    (_, _, _, _, _, _, _, task_id_to_idx, _, inbound_costs, _, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    outbound_idx = task_id_to_idx[1]
    assert np.isinf(inbound_costs[outbound_idx, :]).all(), \
        "outbound tasks must not appear in inbound_sku_distribution_costs"
