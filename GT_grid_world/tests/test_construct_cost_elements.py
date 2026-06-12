"""Unit tests for ``construct_cost_elements`` -- updated for roadmap section 1.6.

History: in 1.4 the skeleton routed shuffle (type=2) tasks through the
*outbound* SKU-distribution matrix (because both pick up from a warehouse
SKU instance) AND populated the *inbound* matrix for shuffle (because
shuffle drops off in the warehouse). 1.6 split that out: shuffle now has
its own ``rearrangement_sku_distribution_costs`` matrix (a 1.6
PLACEHOLDER mirroring the outbound-style formula until Ethan's
rearrangement objective lands) and the IB/OB matrices are populated
*only* for their exact owning task type.

These tests verify:
* the 13-tuple return shape (1.6 added one more matrix)
* shuffle has finite entries in ``rearrangement_sku_distribution_costs``
* shuffle does NOT contaminate the IB or OB matrices any more
* IB / OB cost contracts still hold
* the per-task-type SKU dispatch helper routes by exact type
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
    per_task_type_sku_distribution_term,
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
        rearrangement_sku_distribution_costs,
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
    assert rearrangement_sku_distribution_costs.shape == (N, P)
    assert agent_task_sequence_time.shape == (M,)


def test_construct_cost_elements_shuffle_has_finite_rearrangement_cost(
    mixed_type_tasks, populated_graph, sample_agents
):
    """1.6: shuffle (type=2) tasks must have at least one finite entry in the
    dedicated ``rearrangement_sku_distribution_costs`` matrix. The
    fast_greedy three-way dispatch routes type=2 through this matrix; if it
    were all-inf the allocator would silently refuse to allocate any shuffle.
    """
    J, _ = mixed_type_tasks

    (*_, task_id_to_idx, _, _, _, rearrangement_costs, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    shuffle_idx = task_id_to_idx[3]
    assert np.isfinite(rearrangement_costs[shuffle_idx, :]).any(), (
        "expected shuffle (type=2) task to have at least one finite entry in "
        "rearrangement_sku_distribution_costs"
    )


def test_construct_cost_elements_shuffle_does_not_contaminate_inbound_or_outbound(
    mixed_type_tasks, populated_graph, sample_agents
):
    """1.6 split: shuffle's row in IB and OB matrices must be entirely np.inf,
    because shuffle now has its own dedicated rearrangement matrix. (1.4 used
    to populate both IB and OB for shuffle; that's the cross-pollination 1.6
    cleaned up.)
    """
    J, _ = mixed_type_tasks

    (*_, task_id_to_idx, _, inbound_costs, outbound_costs, _, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    shuffle_idx = task_id_to_idx[3]
    assert np.isinf(inbound_costs[shuffle_idx, :]).all(), \
        "shuffle must not appear in inbound_sku_distribution_costs after 1.6 split"
    assert np.isinf(outbound_costs[shuffle_idx, :]).all(), \
        "shuffle must not appear in outbound_sku_distribution_costs after 1.6 split"


def test_construct_cost_elements_outbound_cost_only_populates_outbound(
    mixed_type_tasks, populated_graph, sample_agents
):
    """OB matrix is populated for type=0 only. IB and shuffle rows must stay
    entirely np.inf so the dispatch can never accidentally route a non-OB
    task through the OB cost path."""
    J, _ = mixed_type_tasks

    (*_, task_id_to_idx, _, _, outbound_costs, rearr_costs, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    inbound_idx = task_id_to_idx[2]
    shuffle_idx = task_id_to_idx[3]
    assert np.isinf(outbound_costs[inbound_idx, :]).all(), \
        "inbound tasks must not appear in outbound_sku_distribution_costs"
    assert np.isinf(outbound_costs[shuffle_idx, :]).all(), \
        "shuffle tasks must not appear in outbound_sku_distribution_costs"


def test_construct_cost_elements_inbound_cost_only_populates_inbound(
    mixed_type_tasks, populated_graph, sample_agents
):
    """IB matrix is populated for type=1 only. OB and shuffle rows must stay
    entirely np.inf."""
    J, _ = mixed_type_tasks

    (*_, task_id_to_idx, _, inbound_costs, _, _, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    outbound_idx = task_id_to_idx[1]
    shuffle_idx = task_id_to_idx[3]
    assert np.isinf(inbound_costs[outbound_idx, :]).all(), \
        "outbound tasks must not appear in inbound_sku_distribution_costs"
    assert np.isinf(inbound_costs[shuffle_idx, :]).all(), \
        "shuffle tasks must not appear in inbound_sku_distribution_costs"


def test_construct_cost_elements_rearrangement_cost_only_populates_shuffle(
    mixed_type_tasks, populated_graph, sample_agents
):
    """Symmetric: rearrangement matrix is populated for type=2 only. OB and
    IB rows must stay entirely np.inf."""
    J, _ = mixed_type_tasks

    (*_, task_id_to_idx, _, _, _, rearr_costs, _) = \
        construct_cost_elements(J, sample_agents, populated_graph, current_time=0)

    outbound_idx = task_id_to_idx[1]
    inbound_idx = task_id_to_idx[2]
    assert np.isinf(rearr_costs[outbound_idx, :]).all(), \
        "outbound tasks must not appear in rearrangement_sku_distribution_costs"
    assert np.isinf(rearr_costs[inbound_idx, :]).all(), \
        "inbound tasks must not appear in rearrangement_sku_distribution_costs"


# ---------------------------------------------------------------------------
# per_task_type_sku_distribution_term -- the 1.6 dispatch helper
# ---------------------------------------------------------------------------
def _make_dispatch_matrices(N=4, P=3, Q=5):
    """Three sentinel matrices distinguishable by their fill values so we can
    verify which one the dispatch helper actually pulled from."""
    inbound = np.full((N, Q), 100.0)
    outbound = np.full((N, P), 200.0)
    rearrangement = np.full((N, P), 300.0)
    return inbound, outbound, rearrangement


def test_dispatch_inbound_returns_inbound_matrix_along_goals_axis():
    inbound, outbound, rearrangement = _make_dispatch_matrices()
    valid_p = np.array([0, 1])
    valid_q = np.array([0, 2, 4])

    term, axis = per_task_type_sku_distribution_term(
        TASK_TYPE_INBOUND, n=2, valid_p=valid_p, valid_q=valid_q,
        inbound_sku_distribution_costs=inbound,
        outbound_sku_distribution_costs=outbound,
        rearrangement_sku_distribution_costs=rearrangement,
    )

    assert axis == "goals"
    assert term.shape == (3,)
    assert np.all(term == 100.0)


def test_dispatch_outbound_returns_outbound_matrix_along_starts_axis():
    inbound, outbound, rearrangement = _make_dispatch_matrices()
    valid_p = np.array([0, 1])
    valid_q = np.array([0, 2, 4])

    term, axis = per_task_type_sku_distribution_term(
        TASK_TYPE_OUTBOUND, n=2, valid_p=valid_p, valid_q=valid_q,
        inbound_sku_distribution_costs=inbound,
        outbound_sku_distribution_costs=outbound,
        rearrangement_sku_distribution_costs=rearrangement,
    )

    assert axis == "starts"
    assert term.shape == (2,)
    assert np.all(term == 200.0)


def test_dispatch_shuffle_returns_rearrangement_matrix_along_starts_axis():
    inbound, outbound, rearrangement = _make_dispatch_matrices()
    valid_p = np.array([0, 1])
    valid_q = np.array([0, 2, 4])

    term, axis = per_task_type_sku_distribution_term(
        TASK_TYPE_SHUFFLE, n=2, valid_p=valid_p, valid_q=valid_q,
        inbound_sku_distribution_costs=inbound,
        outbound_sku_distribution_costs=outbound,
        rearrangement_sku_distribution_costs=rearrangement,
    )

    assert axis == "starts"
    assert term.shape == (2,)
    assert np.all(term == 300.0), \
        "shuffle must pull from the rearrangement matrix (sentinel 300), " \
        "not the outbound matrix (200) -- the 1.4 cross-pollination is gone"


def test_dispatch_unknown_task_type_raises_value_error():
    inbound, outbound, rearrangement = _make_dispatch_matrices()
    valid_p = np.array([0, 1])
    valid_q = np.array([0, 2, 4])

    with pytest.raises(ValueError, match="Unknown task type"):
        per_task_type_sku_distribution_term(
            999, n=0, valid_p=valid_p, valid_q=valid_q,
            inbound_sku_distribution_costs=inbound,
            outbound_sku_distribution_costs=outbound,
            rearrangement_sku_distribution_costs=rearrangement,
        )


# ---------------------------------------------------------------------------
# calculate_deadline_cost piecewise shape (1.6 tightened)
# ---------------------------------------------------------------------------
def test_calculate_deadline_cost_linear_overdue_growth():
    """1.6 docstring fix: the overdue branch is LINEAR (60 + |t|), not
    quadratic. Two consecutive overdue timesteps differ by exactly 1.
    """
    just_overdue = calculate_deadline_cost(deadline=-1, current_time=0)
    more_overdue = calculate_deadline_cost(deadline=-2, current_time=0)
    way_overdue = calculate_deadline_cost(deadline=-10, current_time=0)
    assert more_overdue - just_overdue == 1
    # Linear, not quadratic: the gap from -1 to -10 should be exactly 9, not 99.
    assert way_overdue - just_overdue == 9


def test_calculate_deadline_cost_zero_strictly_above_60():
    """Boundary check: the no-urgency branch fires for ``> 60``, so 61 returns 0
    and 60 returns 0 (the linear-ramp branch is ``time_until_deadline > 0``
    AND ``<= 60``, mapping 60 to ``60 - 60 = 0``)."""
    assert calculate_deadline_cost(deadline=61, current_time=0) == 0
    assert calculate_deadline_cost(deadline=60, current_time=0) == 0
    assert calculate_deadline_cost(deadline=59, current_time=0) == 1
