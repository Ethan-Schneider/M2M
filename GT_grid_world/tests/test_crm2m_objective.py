"""Unit tests for the crM2M (concatenated rearrangement) objective and the
J_r consumer/lifecycle layer (roadmap section 4.2).

Two layers are covered:

1. The rearrangement utility itself -- ``rearrangement_cost_cube`` (cost = -U,
   gated) and ``compute_crm2m_terms`` (the static detour/benefit reference
   distances + same-aisle coupling mask). These are tested with hand-computed
   values so the three gates (U<=0, weighted-detour cutoff, cross-aisle) and the
   ``U = b - lambda*Delta`` arithmetic are pinned exactly.

2. The J_r consumer lifecycle -- ``merge_reallocation_tasks_into_J`` (4-tuple ->
   5-tuple, per-SKU dedup, high id range), ``drop_expired_rearrangements`` (a
   shuffle past its window must not be re-attempted), and
   ``prune_uncommitted_rearrangements`` (candidates no agent committed to are not
   left lingering in J). These use lightweight fakes so the lifecycle logic is
   tested in isolation from the full warehouse/graph stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from GT_grid_world.src.agent import Agent, AgentLoader
from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
    compute_crm2m_terms,
    rearrangement_cost_cube,
)
from GT_grid_world.src.task_allocation_algorithms.initial_solutions.fast_greedy import (
    fast_greedy_call,
)
from GT_grid_world.src.reallocation_tasks.jr_consumer import (
    add_reallocation_tasks_to_J_a,
    drop_expired_rearrangements,
    prune_uncommitted_rearrangements,
    REARRANGEMENT_TASK_ID_BASE,
    TASK_TYPE_SHUFFLE,
)


# ---------------------------------------------------------------------------
# rearrangement_cost_cube: arithmetic + gates
# ---------------------------------------------------------------------------
def test_cost_cube_returns_negative_utility_when_beneficial():
    """cost = -U = -(b - lambda*Delta). With an anchor that coincides with the
    home reference the detour can be zero/negative, so a closer-to-home goal
    yields a strictly positive U and hence a strictly negative cost."""
    # P=2 starts, Q=2 goals, M=1 agent.
    start_home = np.array([10.0, 5.0])      # dist(s_p, h0)
    goal_home = np.array([1.0, 8.0])        # dist(d_q, h0)
    agent_start = np.array([[0.0, 0.0]])    # dist(g^m_{i-1}, s_p)
    agent_home = np.array([10.0])           # dist(a_m, h0)
    coupling = np.ones((2, 2), dtype=bool)
    valid_p = np.array([0, 1])
    valid_q = np.array([0, 1])

    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        valid_p, valid_q, lambda_=1.0, detour_cutoff=100.0,
    )

    # Delta[0,0]=0+10-10=0, b[0,0]=10-1=9 -> U=9 -> cost=-9
    # Delta[0,1]=0+5-10=-5, b[1,1]=5-8=-3 -> U=-3-(-5)=2 -> cost=-2
    assert cost.shape == (1, 2, 2)
    assert cost[0, 0, 0] == pytest.approx(-9.0)
    assert cost[0, 0, 1] == pytest.approx(-2.0)  # b[0,1]=10-8=2, Delta=0 -> U=2
    assert cost[0, 1, 0] == pytest.approx(-9.0)  # b[1,0]=5-1=4, Delta=-5 -> U=9
    assert cost[0, 1, 1] == pytest.approx(-2.0)


def test_cost_cube_gates_non_positive_utility():
    """U <= 0 must be gated to +inf (no net benefit -> never chosen)."""
    start_home = np.array([5.0])
    goal_home = np.array([5.0])             # b = 0
    agent_start = np.array([[1.0]])
    agent_home = np.array([5.0])            # Delta = 1+5-5 = 1 > 0 -> U = 0 - 1 < 0
    coupling = np.ones((1, 1), dtype=bool)
    vp = np.array([0]); vq = np.array([0])

    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=1.0, detour_cutoff=100.0,
    )
    assert np.isinf(cost[0, 0, 0]) and cost[0, 0, 0] > 0


def test_cost_cube_gates_weighted_detour_cutoff():
    """A candidate with positive U but lambda*Delta >= cutoff is still gated."""
    start_home = np.array([10.0])
    goal_home = np.array([1.0])             # b = 9
    agent_start = np.array([[3.0]])
    agent_home = np.array([10.0])           # Delta = 3+10-10 = 3
    coupling = np.ones((1, 1), dtype=bool)
    vp = np.array([0]); vq = np.array([0])

    # lambda*Delta = 2*3 = 6 >= cutoff 5, even though U = 9 - 6 = 3 > 0.
    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=2.0, detour_cutoff=5.0,
    )
    assert np.isinf(cost[0, 0, 0])

    # Loosen the cutoff and the same candidate becomes choosable (cost = -3).
    cost2 = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=2.0, detour_cutoff=100.0,
    )
    assert cost2[0, 0, 0] == pytest.approx(-3.0)


def test_cost_cube_gates_cross_aisle_pairs():
    """coupling_mask False (start/goal not same aisle) must be gated to +inf."""
    start_home = np.array([10.0, 10.0])
    goal_home = np.array([1.0, 1.0])
    agent_start = np.array([[0.0, 0.0]])
    agent_home = np.array([10.0])
    # Only the diagonal pairs share an aisle.
    coupling = np.array([[True, False], [False, True]], dtype=bool)
    vp = np.array([0, 1]); vq = np.array([0, 1])

    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=1.0, detour_cutoff=100.0,
    )
    assert np.isfinite(cost[0, 0, 0]) and np.isfinite(cost[0, 1, 1])
    assert np.isinf(cost[0, 0, 1]) and np.isinf(cost[0, 1, 0])


# ---------------------------------------------------------------------------
# compute_crm2m_terms: reference distances + coupling + anchor
# ---------------------------------------------------------------------------
def test_compute_crm2m_terms_distances_coupling_and_anchor(sample_agents, tiny_graph):
    """start_home/goal_home are Manhattan distances to h_0 = agents[0].home;
    coupling is same-column; agent_home uses the agent's anchor (last task goal,
    else current state)."""
    from GT_grid_world.src.utils import manhattan_distance

    Rs = sample_agents
    h0 = Rs.agents[0].home

    aisles = tiny_graph.get_aisle_locations()
    # Pick two starts and two goals spanning at least two columns when possible.
    cols = sorted({c for _, c in aisles})
    start_locs = [next(a for a in aisles if a[1] == cols[0])]
    goal_locs = [next(a for a in aisles if a[1] == cols[0])]
    if len(cols) > 1:
        start_locs.append(next(a for a in aisles if a[1] == cols[1]))
        goal_locs.append(next(a for a in aisles if a[1] == cols[1]))

    # Give agent 1 an anchor via a task sequence (goal = some aisle cell).
    anchor_goal = aisles[-1]
    Rs.agents[1].task_sequence.append((42, aisles[0], anchor_goal, 100))

    home, start_home, goal_home, agent_home, coupling = compute_crm2m_terms(
        Rs, tiny_graph, start_locs, goal_locs, method="manhattan"
    )

    assert home == h0
    for i, s in enumerate(start_locs):
        assert start_home[i] == pytest.approx(manhattan_distance(s, h0))
    for j, g in enumerate(goal_locs):
        assert goal_home[j] == pytest.approx(manhattan_distance(g, h0))

    # Coupling reflects same-column membership.
    for i, s in enumerate(start_locs):
        for j, g in enumerate(goal_locs):
            assert bool(coupling[i, j]) == (s[1] == g[1])

    # Idle agent 0 anchors at its state; agent 1 anchors at its last task goal.
    assert agent_home[0] == pytest.approx(manhattan_distance(Rs.agents[0].state, h0))
    assert agent_home[1] == pytest.approx(manhattan_distance(anchor_goal, h0))


# ---------------------------------------------------------------------------
# Integration: a beneficial shuffle is actually allocated by fast_greedy_call
# ---------------------------------------------------------------------------
def _beneficial_shuffle_scenario(G):
    """Find (column, start, anchor, goal) for a clearly-beneficial same-aisle
    shuffle on graph ``G``: a SKU instance ``s``, an empty anchor cell one row
    toward home, and an empty goal >= 4 rows below ``s`` in the same column.
    Returns None if the warehouse layout doesn't admit one."""
    aisles = G.get_aisle_locations()
    full = set(G.warehouse.get_full_locations())
    empty = set(G.warehouse.get_empty_locations())

    by_col = {}
    for loc in aisles:
        by_col.setdefault(loc[1], []).append(loc)

    for c, locs in by_col.items():
        rows = sorted(r for r, _ in locs)
        for s_row in rows:
            s = (s_row, c)
            a = (s_row + 1, c)
            if s not in full or a not in empty:
                continue
            g_candidates = [(gr, c) for gr in rows if gr >= s_row + 4 and (gr, c) in empty]
            if not g_candidates:
                continue
            g = max(g_candidates, key=lambda cell: cell[0])
            return (c, s, a, g)
    return None



def test_fast_greedy_allocates_beneficial_shuffle(populated_graph, minimal_stats, seeded_rng):
    """End-to-end through the live allocator: a same-aisle shuffle that moves a
    SKU much closer to the home reference, with the agent anchored right next to
    the pickup (small detour), has U > 0 and is allocated with negative cost.

    Geometry (Manhattan, single aisle column ``c``, home at the driveway row 17):
        b     = dist(s, h0) - dist(g, h0) = (g_row - s_row)         (same column)
        Delta = dist(anchor, s) + dist(s, h0) - dist(anchor, h0) = 2 (anchor one
                row below s, i.e. one step toward home)
        U     = b - lambda*Delta = (g_row - s_row) - 3   (lambda=1.5)
    so g at least ~4 rows below s gives U > 0.
    """
    G = populated_graph
    scenario = _beneficial_shuffle_scenario(G)
    if scenario is None:
        pytest.skip("populated_graph has no aisle column with the needed full/empty layout")

    c, s, a, g = scenario
    sku = G.warehouse.get_sku_at_location(s).sku_id
    h0 = (17, c)

    # Idle agent anchored at ``a`` (its state, since it has no task sequence),
    # with home = h0 so dist(a, h0) = dist(s, h0) - 1 and the detour is just 2.
    agent = Agent(agent_id=0, state=a, home=h0)
    Rs = AgentLoader([agent])

    shuffle_id = REARRANGEMENT_TASK_ID_BASE
    J = {shuffle_id: (frozenset({s}), frozenset({g}), 100, sku, TASK_TYPE_SHUFFLE)}

    _, allocations, cost = fast_greedy_call(
        minimal_stats, G, Rs, J, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.0,
        sku_distribution_weight=0.0, crm2m_lambda=1.5, crm2m_detour_cutoff=100.0,
    )

    assert allocations, "a clearly beneficial same-aisle shuffle should be allocated"
    assert allocations[0][1] == shuffle_id
    assert cost < 0, "a beneficial shuffle has cost -U < 0"


def test_py_lns_handles_shuffle_via_greedy_repair(populated_graph, minimal_stats, seeded_rng):
    """The LNS improver (py_lns_call -> greedy_repair) must thread the crM2M
    type=2 path without raising and keep a clearly-beneficial shuffle assigned."""
    from GT_grid_world.src.task_allocation_algorithms.py_lns_V2 import py_lns_call

    G = populated_graph
    scenario = _beneficial_shuffle_scenario(G)
    if scenario is None:
        pytest.skip("populated_graph has no aisle column with the needed full/empty layout")
    c, s, a, g = scenario
    sku = G.warehouse.get_sku_at_location(s).sku_id

    agent = Agent(agent_id=0, state=a, home=(17, c))
    Rs = AgentLoader([agent])
    shuffle_id = REARRANGEMENT_TASK_ID_BASE
    J = {shuffle_id: (frozenset({s}), frozenset({g}), 100, sku, TASK_TYPE_SHUFFLE)}

    Rs_out, allocations, _ = py_lns_call(
        minimal_stats, G, Rs, J, initial_task_assignment_strategy="fast_greedy",
        time_limit=0.05, cost_calculation_method="manhattan", repair_operator="greedy",
        t=0, crm2m_lambda=1.5, crm2m_detour_cutoff=100.0,
    )

    assigned_ids = {task[0] for ag in Rs_out.agents for task in ag.task_sequence}
    assert shuffle_id in assigned_ids, "LNS should keep the beneficial shuffle assigned"


# ---------------------------------------------------------------------------
# simulate: completed shuffle is reported on the rearrangement track
# ---------------------------------------------------------------------------
def test_simulate_routes_completed_shuffle_to_rearrangement_stats(
    populated_graph, minimal_stats, seeded_rng
):
    """A completed type=2 task must be recorded via
    ``add_completed_rearrangement_task_id`` (separate throughput track) and NOT
    via the regular ``add_completed_task_id`` / service-time path."""
    from GT_grid_world.src.simulate import simulate, TASK_TYPE_SHUFFLE as SIM_SHUFFLE

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]

    task_id = REARRANGEMENT_TASK_ID_BASE
    deadline = 100
    J = {task_id: (frozenset({pickup_loc}), frozenset({goal_loc}), deadline, sku, SIM_SHUFFLE)}
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(pickup_loc, True)

    # Tick 0: pickup. Tick 1: dropoff (completes the shuffle).
    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, J, {}, "small_test", t=0)
    agent.state = goal_loc
    agent.path_sequence = []
    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, J, {}, "small_test", t=1)

    assert task_id in minimal_stats.get_completed_rearrangement_task_ids()
    assert task_id not in minimal_stats.get_completed_task_ids()
    assert minimal_stats.get_total_completed_rearrangement_tasks() == 1


def test_simulate_picks_up_shuffle_living_in_J_a(
    populated_graph, minimal_stats, seeded_rng
):
    """A shuffle stored in ``J_a`` (not ``J``) must be pickable.

    Regression: the status=1 pickup gate previously only admitted ``task_id in
    J``; shuffles live in ``J_a``, so every shuffle pickup was silently skipped
    and the move could never execute. Here the agent sits on the pickup cell with
    the task in ``J_a`` only -- after one tick it must be carrying the SKU
    (status 1 -> 2)."""
    from GT_grid_world.src.simulate import simulate, TASK_TYPE_SHUFFLE as SIM_SHUFFLE

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]

    task_id = REARRANGEMENT_TASK_ID_BASE
    deadline = 100
    J_a = {task_id: (frozenset({pickup_loc}), frozenset({goal_loc}), deadline, sku, SIM_SHUFFLE)}
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(pickup_loc, True)

    # Task is only in J_a; J is empty. The agent is already on the pickup cell.
    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, {}, J_a, "small_test", t=0)

    assert agent.status == 2, "agent should have picked up the J_a shuffle and moved to delivery"
    assert agent.get_sku_id_carrying() == sku


def test_simulate_aborts_stale_shuffle_with_wrong_sku_at_cell(
    populated_graph, minimal_stats, seeded_rng
):
    """A shuffle whose committed source cell no longer holds its SKU must be
    aborted, not executed.

    Regression: under warehouse churn an outbound can empty a shuffle's source
    cell and an inbound can refill it with a *different* SKU before the agent
    arrives. Picking up the wrong item used to crash at the delivery-side
    SKU-match check. The agent must instead drop the stale shuffle (pop from
    ``J_a``), carry nothing, and be freed for re-tasking -- no exception."""
    from GT_grid_world.src.simulate import simulate, TASK_TYPE_SHUFFLE as SIM_SHUFFLE

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]
    actual_sku = populated_graph.warehouse.get_sku_at_location(pickup_loc).sku_id

    # The task expects a SKU that is NOT the one physically at the cell.
    wrong_sku = actual_sku + 1
    task_id = REARRANGEMENT_TASK_ID_BASE
    deadline = 100
    J_a = {task_id: (frozenset({pickup_loc}), frozenset({goal_loc}), deadline, wrong_sku, SIM_SHUFFLE)}

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(pickup_loc, True)

    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, {}, J_a, "small_test", t=0)

    assert task_id not in J_a, "stale shuffle should have been dropped from J_a"
    assert agent.get_sku_id_carrying() is None, "agent must not pick up the wrong SKU"
    assert agent.status == 0 and agent.task_sequence == [], "agent should be freed"
    # The SKU that was at the cell must remain in the warehouse (not removed).
    assert pickup_loc in populated_graph.warehouse.get_full_locations()


# ---------------------------------------------------------------------------
# J_r consumer: fakes
# ---------------------------------------------------------------------------
class _FakeSku:
    def __init__(self, sku_id):
        self.sku_id = sku_id


class _FakeWarehouse:
    def __init__(self, loc_to_sku):
        self._loc_to_sku = loc_to_sku

    def get_sku_at_location(self, loc):
        sku = self._loc_to_sku.get(loc)
        return _FakeSku(sku) if sku is not None else None


class _FakeGraph:
    def __init__(self, loc_to_sku):
        self.warehouse = _FakeWarehouse(loc_to_sku)


class _FakeAgent:
    def __init__(self, task_sequence=None, status=0):
        self.task_sequence = list(task_sequence or [])
        self.status = status
        self.path_sequence = []


class _FakeRs:
    def __init__(self, agents):
        self.agents = agents


def test_merge_reallocation_tasks_dedups_by_sku_and_uses_high_ids():
    """Two candidates for the SAME sku produce a single J entry; ids start at the
    rearrangement base; the 4-tuple becomes a 5-tuple type=2 J entry."""
    s1, g1 = (5, 2), (3, 2)
    s2, g2 = (6, 4), (2, 4)
    G = _FakeGraph({s1: 7, s2: 7})  # both starts hold sku 7
    Ta = {
        0: ({(s1, g1)}, 0, 50, TASK_TYPE_SHUFFLE),
        1: ({(s2, g2)}, 0, 60, TASK_TYPE_SHUFFLE),  # same sku 7 -> deduped
    }
    J = {}
    next_id = add_reallocation_tasks_to_J_a(Ta, J, G, REARRANGEMENT_TASK_ID_BASE)

    assert len(J) == 1
    task_id, entry = next(iter(J.items()))
    assert task_id == REARRANGEMENT_TASK_ID_BASE
    assert next_id == REARRANGEMENT_TASK_ID_BASE + 1
    starts, goals, deadline, sku, ttype = entry
    assert ttype == TASK_TYPE_SHUFFLE and sku == 7
    assert starts == frozenset({s1}) and goals == frozenset({g1})
    assert deadline == 50


def test_merge_skips_sku_already_active_in_J():
    """A candidate whose sku already has a live rearrangement task in J is skipped."""
    s, g = (5, 2), (3, 2)
    G = _FakeGraph({s: 7})
    J = {REARRANGEMENT_TASK_ID_BASE: (frozenset({(9, 2)}), frozenset({(4, 2)}), 40, 7, TASK_TYPE_SHUFFLE)}
    Ta = {0: ({(s, g)}, 0, 50, TASK_TYPE_SHUFFLE)}

    next_id = add_reallocation_tasks_to_J_a(Ta, J, G, REARRANGEMENT_TASK_ID_BASE + 1)
    assert len(J) == 1  # nothing added
    assert next_id == REARRANGEMENT_TASK_ID_BASE + 1


def test_drop_expired_rearrangements_removes_past_window_and_resets_agent():
    """A committed (status=1, heading to pickup) shuffle whose deadline has passed
    is dropped from J and from the agent's sequence, and the agent is reset."""
    rid = REARRANGEMENT_TASK_ID_BASE
    shuffle_task = (rid, (5, 2), (3, 2), 40)
    agent = _FakeAgent(task_sequence=[shuffle_task], status=1)
    Rs = _FakeRs([agent])
    J = {rid: (frozenset({(5, 2)}), frozenset({(3, 2)}), 40, 7, TASK_TYPE_SHUFFLE)}

    drop_expired_rearrangements(J, Rs, t=40)  # t >= deadline -> expired

    assert rid not in J
    assert agent.task_sequence == []
    assert agent.status == 0


def test_drop_expired_keeps_carried_shuffle():
    """An agent already carrying the shuffled SKU (status=2) finishes the move
    even if the window passed -- it must not be stranded mid-carry."""
    rid = REARRANGEMENT_TASK_ID_BASE
    shuffle_task = (rid, (5, 2), (3, 2), 40)
    agent = _FakeAgent(task_sequence=[shuffle_task], status=2)
    Rs = _FakeRs([agent])
    J = {rid: (frozenset({(5, 2)}), frozenset({(3, 2)}), 40, 7, TASK_TYPE_SHUFFLE)}

    drop_expired_rearrangements(J, Rs, t=50)

    assert rid in J
    assert agent.task_sequence == [shuffle_task]


def test_prune_uncommitted_rearrangements_drops_only_unallocated():
    """Type=2 tasks not in any agent sequence are pruned; committed ones remain.
    Real (non-type-2) tasks are never touched."""
    committed_id = REARRANGEMENT_TASK_ID_BASE
    orphan_id = REARRANGEMENT_TASK_ID_BASE + 1
    agent = _FakeAgent(task_sequence=[(committed_id, (5, 2), (3, 2), 40)], status=1)
    Rs = _FakeRs([agent])
    J = {
        committed_id: (frozenset({(5, 2)}), frozenset({(3, 2)}), 40, 7, TASK_TYPE_SHUFFLE),
        orphan_id: (frozenset({(6, 4)}), frozenset({(2, 4)}), 50, 8, TASK_TYPE_SHUFFLE),
        1: (frozenset({(9, 0)}), frozenset({(0, 0)}), 100, 3, 0),  # real OB, untouched
    }

    prune_uncommitted_rearrangements(J, Rs)

    assert committed_id in J
    assert orphan_id not in J
    assert 1 in J
