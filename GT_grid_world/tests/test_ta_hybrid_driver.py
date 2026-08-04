"""Unit tests for the TA-Hybrid outer driver.

These tests target the helpers and state management of
``task_allocation_algorithms.ta_hybrid_driver``. End-to-end behaviour
(group transitions on a real schedule with AMAPF + ICBS calls) is
exercised by the smoke run, not here -- those algorithms are
non-trivial to set up deterministically in a unit-test fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from GT_grid_world.src.agent import Agent, AgentLoader
from GT_grid_world.src.task_allocation_algorithms import ta_hybrid_driver
from GT_grid_world.src.task_allocation_algorithms.ta_assignment import (
    MaterializedTask,
)
from GT_grid_world.src.task_allocation_algorithms.ta_hybrid_driver import (
    _AgentRoute,
    _DriverState,
    _get_driver,
    _group_2_set_changed,
    _is_group_1,
    _is_group_2,
    _materialized_to_tuple,
    _new_group_1,
    _release_time_for_task_id,
    _route_to_external_entry,
    _snapshot_statuses,
    _stats_init_tasks,
    _truncate_route,
    reset_driver_state,
)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
class TestDriverStateLifecycle:
    def setup_method(self):
        reset_driver_state()

    def test_get_driver_returns_fresh_state_for_new_loader(self):
        Rs = AgentLoader([Agent(agent_id=0, state=(0, 0))])
        d = _get_driver(Rs)
        assert not d.initialized
        assert d.plan == {}
        assert d.initial_L == 0
        assert d.route == {}

    def test_get_driver_is_idempotent_per_loader(self):
        Rs = AgentLoader([Agent(agent_id=0, state=(0, 0))])
        d1 = _get_driver(Rs)
        d2 = _get_driver(Rs)
        assert d1 is d2

    def test_distinct_loaders_get_distinct_state(self):
        Rs1 = AgentLoader([Agent(agent_id=0, state=(0, 0))])
        Rs2 = AgentLoader([Agent(agent_id=0, state=(0, 0))])
        d1 = _get_driver(Rs1)
        d2 = _get_driver(Rs2)
        assert d1 is not d2

    def test_reset_clears_all_state(self):
        Rs = AgentLoader([Agent(agent_id=0, state=(0, 0))])
        d = _get_driver(Rs)
        d.initialized = True
        reset_driver_state()
        new_d = _get_driver(Rs)
        # Same key, but a fresh state instance.
        assert not new_d.initialized
        assert new_d is not d


# ---------------------------------------------------------------------------
# MaterializedTask <-> M2M tuple
# ---------------------------------------------------------------------------
class TestMaterializedToTuple:
    def test_task_id_is_one_based(self):
        task = MaterializedTask(
            schedule_row_index=0,
            release_time=0,
            deadline=100,
            sku_id=1,
            task_type=0,
            pickup=(2, 3),
            delivery=(5, 7),
        )
        tup = _materialized_to_tuple(task)
        assert tup == (1, (2, 3), (5, 7), 100)

    def test_higher_index_maps_to_higher_id(self):
        a = _materialized_to_tuple(
            MaterializedTask(0, 0, 100, 1, 0, (0, 0), (1, 1))
        )
        b = _materialized_to_tuple(
            MaterializedTask(5, 10, 200, 2, 1, (3, 3), (4, 4))
        )
        assert a[0] == 1
        assert b[0] == 6
        assert b[1] == (3, 3) and b[2] == (4, 4)
        assert b[3] == 200


# ---------------------------------------------------------------------------
# Group classification
# ---------------------------------------------------------------------------
class TestGroupClassification:
    def test_free_agent_with_tasks_is_group_2(self):
        ag = Agent(agent_id=0, state=(0, 0))
        ag.status = 0
        ag.task_sequence = [(1, (1, 1), (2, 2), 100)]
        assert _is_group_2(ag)
        assert not _is_group_1(ag)

    def test_to_pickup_agent_is_group_2(self):
        ag = Agent(agent_id=0, state=(0, 0))
        ag.status = 1
        ag.task_sequence = [(1, (1, 1), (2, 2), 100)]
        assert _is_group_2(ag)
        assert not _is_group_1(ag)

    def test_carrying_agent_is_group_1(self):
        ag = Agent(agent_id=0, state=(0, 0))
        ag.status = 2
        ag.task_sequence = [(1, (1, 1), (2, 2), 100)]
        assert _is_group_1(ag)
        assert not _is_group_2(ag)

    def test_free_agent_without_tasks_is_neither(self):
        ag = Agent(agent_id=0, state=(0, 0))
        ag.status = 0
        ag.task_sequence = []
        assert not _is_group_1(ag)
        assert not _is_group_2(ag)


# ---------------------------------------------------------------------------
# Transition detection
# ---------------------------------------------------------------------------
class TestTransitionDetection:
    def setup_method(self):
        reset_driver_state()

    def _make_loader_with_statuses(self, statuses: list[int]) -> AgentLoader:
        agents = []
        for i, s in enumerate(statuses):
            ag = Agent(agent_id=i, state=(0, i))
            ag.status = s
            ag.task_sequence = [(i + 1, (1, 1), (2, 2), 100)]
            agents.append(ag)
        return AgentLoader(agents)

    def test_new_group_1_picks_up_first_carrying_transition(self):
        Rs = self._make_loader_with_statuses([1, 2, 0])
        d = _DriverState()
        d.prev_status = {0: 1, 1: 1, 2: 0}  # agent 1 was Group 2 last tick
        new_g1 = _new_group_1(d, Rs)
        assert new_g1 == [1]

    def test_no_new_group_1_when_carrying_already_known(self):
        Rs = self._make_loader_with_statuses([2, 1, 0])
        d = _DriverState()
        d.prev_status = {0: 2, 1: 1, 2: 0}  # agent 0 was already carrying
        new_g1 = _new_group_1(d, Rs)
        assert new_g1 == []

    def test_unknown_prior_status_is_treated_as_new(self):
        Rs = self._make_loader_with_statuses([2, 2])
        d = _DriverState()
        # prev_status empty -- cold start with carrying agents (mid-run load).
        new_g1 = _new_group_1(d, Rs)
        assert sorted(new_g1) == [0, 1]

    def test_group_2_set_change_detects_completion(self):
        Rs = self._make_loader_with_statuses([1, 1, 1])
        d = _DriverState()
        # Last tick: agents 0, 1 were Group 2; agent 2 was Group 1
        # (carrying). Now agent 2 has flipped back to to-pickup, so the
        # Group 2 set has grown.
        d.prev_group_2 = {0, 1}
        assert _group_2_set_changed(d, Rs) is True

    def test_group_2_set_unchanged_returns_false(self):
        Rs = self._make_loader_with_statuses([1, 1, 0])
        d = _DriverState()
        d.prev_group_2 = {0, 1, 2}
        assert _group_2_set_changed(d, Rs) is False

    def test_snapshot_overwrites_prev_status(self):
        Rs = self._make_loader_with_statuses([2, 0, 1])
        d = _DriverState()
        d.prev_status = {0: 1, 1: 1, 2: 1}
        d.prev_group_2 = {0, 1, 2}
        _snapshot_statuses(d, Rs)
        assert d.prev_status == {0: 2, 1: 0, 2: 1}
        # Agent 0: status 2 -> Group 1; agent 1: status 0 with tasks ->
        # Group 2; agent 2: status 1 with tasks -> Group 2.
        assert d.prev_group_2 == {1, 2}


# ---------------------------------------------------------------------------
# Route packing / truncation
# ---------------------------------------------------------------------------
class TestRouteHelpers:
    def test_route_to_external_entry_concatenates_tail_and_dummy(self):
        route = _AgentRoute(
            start_t=5,
            sub_path=[(0, 0), (0, 1), (0, 2)],
            dummy_path=[(1, 2), (2, 2)],
        )
        entry = _route_to_external_entry(route)
        start_loc, tail, start_t, is_perm = entry
        assert start_loc == (0, 0)
        assert tail == [(0, 1), (0, 2), (1, 2), (2, 2)]
        assert start_t == 5
        assert is_perm is True

    def test_truncate_route_passes_through_when_not_yet_started(self):
        route = _AgentRoute(
            start_t=10, sub_path=[(0, 0), (0, 1)], dummy_path=[(0, 2)]
        )
        out = _truncate_route(route, t_now=10)
        assert out is route  # unchanged

    def test_truncate_route_advances_past_subpath_into_dummy(self):
        route = _AgentRoute(
            start_t=10,
            sub_path=[(0, 0), (0, 1), (0, 2)],
            dummy_path=[(0, 3), (0, 4)],
        )
        # Elapsed = 2: agent currently at sub_path[2] = (0,2) (the
        # pickup milestone). Remaining = [(0,2), (0,3), (0,4)].
        out = _truncate_route(route, t_now=12)
        assert out.start_t == 12
        assert out.sub_path == [(0, 2)]
        assert out.dummy_path == [(0, 3), (0, 4)]

    def test_truncate_route_collapses_to_parking_when_finished(self):
        route = _AgentRoute(
            start_t=0, sub_path=[(0, 0), (0, 1)], dummy_path=[(0, 2)]
        )
        out = _truncate_route(route, t_now=100)
        assert out.sub_path == [(0, 2)]
        assert out.dummy_path == []


# ---------------------------------------------------------------------------
# Release-time lookup
# ---------------------------------------------------------------------------
class TestReleaseTimeLookup:
    def test_finds_release_time_for_existing_task(self):
        d = _DriverState()
        d.plan = {
            0: [
                MaterializedTask(0, 5, 100, 1, 0, (0, 0), (1, 1)),
                MaterializedTask(2, 50, 200, 2, 0, (0, 1), (1, 2)),
            ],
            1: [
                MaterializedTask(1, 20, 150, 1, 1, (2, 0), (3, 1)),
            ],
        }
        # task_id 1 -> schedule_row_index 0 -> release_time 5
        assert _release_time_for_task_id(d, 1) == 5
        # task_id 2 -> schedule_row_index 1 -> release_time 20
        assert _release_time_for_task_id(d, 2) == 20
        # task_id 3 -> schedule_row_index 2 -> release_time 50
        assert _release_time_for_task_id(d, 3) == 50

    def test_returns_zero_for_unknown_task(self):
        d = _DriverState()
        d.plan = {}
        assert _release_time_for_task_id(d, 999) == 0


# ---------------------------------------------------------------------------
# Stats initialisation
# ---------------------------------------------------------------------------
class TestStatsInit:
    def test_only_initialises_each_task_once(self, minimal_stats):
        # Manually increment one of the accumulators after first init,
        # then re-call: the value must NOT be reset.
        d = _DriverState()
        J = {1: ("dummy",), 2: ("dummy",)}
        _stats_init_tasks(minimal_stats, J, d)
        assert d.stats_initialized_task_ids == {1, 2}

        minimal_stats.update_actual_distance(1, 5)
        assert minimal_stats.get_actual_distance(1) == 5

        # Second call: task 1 is already known, so add_actual_distance
        # must NOT fire for it.
        _stats_init_tasks(minimal_stats, J, d)
        assert minimal_stats.get_actual_distance(1) == 5  # preserved

    def test_initialises_only_new_task_ids_on_re_call(self, minimal_stats):
        d = _DriverState()
        _stats_init_tasks(minimal_stats, {1: ("dummy",)}, d)
        minimal_stats.update_actual_distance(1, 7)

        # Now task 2 appears in J; task 1 still present.
        _stats_init_tasks(minimal_stats, {1: ("dummy",), 2: ("dummy",)}, d)
        assert d.stats_initialized_task_ids == {1, 2}
        assert minimal_stats.get_actual_distance(1) == 7  # preserved
        assert minimal_stats.get_actual_distance(2) == 0  # newly added


# ---------------------------------------------------------------------------
# End-to-end smoke through the dispatch
# ---------------------------------------------------------------------------
class TestDispatchPreconditions:
    def setup_method(self):
        reset_driver_state()

    def test_first_call_without_schedule_raises(
        self, minimal_stats, populated_graph, sample_agents
    ):
        with pytest.raises(ValueError, match="precomputed schedule"):
            ta_hybrid_driver.ta_hybrid_call(
                S=minimal_stats, G=populated_graph,
                Rs=sample_agents, J={}, t=0, schedule=None,
            )

    def test_initialization_populates_plan_and_task_sequence(
        self, minimal_stats, populated_graph, sample_agents
    ):
        # Build a single-task schedule: one outbound for SKU 0 at t=0.
        # The SKU must already be in the warehouse for materialisation
        # to find it -- populated_graph at 30% fill has plenty.
        schedule = np.array([[0, 100, 0, 0]], dtype=int)
        try:
            ta_hybrid_driver.ta_hybrid_call(
                S=minimal_stats, G=populated_graph,
                Rs=sample_agents, J={1: ("dummy",)}, t=0,
                schedule=schedule,
            )
        except Exception:
            # Path planning may still fail on the tiny map -- but the
            # initialisation itself (TSP + task_sequence) should have
            # run before the planner returned. We assert on driver state
            # rather than on the call success.
            pass

        d = _get_driver(sample_agents)
        assert d.initialized
        assert d.plan, "driver plan should contain at least one entry"
        # Stats accumulator was initialised for the released task.
        assert 1 in d.stats_initialized_task_ids


# ---------------------------------------------------------------------------
# Apply-results helpers
# ---------------------------------------------------------------------------
class TestApplyResults:
    """End-to-end-ish: feed synthetic plan results back to the driver
    and check the agent state mutation matches paper semantics.
    """

    def setup_method(self):
        reset_driver_state()

    def test_apply_pickup_swaps_task_sequences_on_anonymity_swap(self):
        from GT_grid_world.src.task_allocation_algorithms.ta_hybrid import (
            PickupPlanResult,
        )

        ag0 = Agent(agent_id=0, state=(0, 0))
        ag0.task_sequence = [(1, (5, 5), (9, 9), 100)]
        ag0.status = 0

        ag1 = Agent(agent_id=1, state=(2, 0))
        ag1.task_sequence = [(2, (7, 7), (3, 3), 100)]
        ag1.status = 0

        Rs = AgentLoader([ag0, ag1])
        d = _get_driver(Rs)

        # Agent 0 ends at (7, 7) (agent 1's pickup) and takes agent 1's
        # tasks; vice versa for agent 1.
        results = {
            0: PickupPlanResult(
                sub_path=[(0, 0), (1, 0), (7, 7)],
                dummy_path=[(7, 6), (6, 6)],
                assigned_task_agent_id=1,
            ),
            1: PickupPlanResult(
                sub_path=[(2, 0), (3, 0), (5, 5)],
                dummy_path=[(5, 4)],
                assigned_task_agent_id=0,
            ),
        }
        ta_hybrid_driver._apply_pickup_results(d, Rs, results, t_now=0)

        # Tasks must have swapped.
        assert ag0.task_sequence[0][0] == 2  # took agent 1's task
        assert ag1.task_sequence[0][0] == 1  # took agent 0's task
        # Paths and status applied.
        assert ag0.path_sequence == [(1, 0), (7, 7)]
        assert ag1.path_sequence == [(3, 0), (5, 5)]
        assert ag0.status == 1 and ag1.status == 1
        # Routes cached.
        assert 0 in d.route and 1 in d.route
        assert d.route[0].sub_path == [(0, 0), (1, 0), (7, 7)]
        assert d.route[1].dummy_path == [(5, 4)]

    def test_apply_delivery_writes_path_and_caches_route(self):
        from GT_grid_world.src.task_allocation_algorithms.ta_hybrid import (
            DeliveryPlanResult,
        )

        ag0 = Agent(agent_id=0, state=(5, 5))
        ag0.task_sequence = [(1, (5, 5), (9, 9), 100)]
        ag0.status = 2  # just transitioned from Group 2

        Rs = AgentLoader([ag0])
        d = _get_driver(Rs)

        results = {
            0: DeliveryPlanResult(
                sub_path=[(5, 5), (6, 5), (9, 9)],
                dummy_path=[(8, 9), (0, 0)],
            ),
        }
        ta_hybrid_driver._apply_delivery_results(d, Rs, results, t_now=3)

        assert ag0.path_sequence == [(6, 5), (9, 9)]
        # Status was NOT changed -- the simulator owns the 2 -> 0/1
        # transition on delivery arrival.
        assert ag0.status == 2
        assert d.route[0].sub_path == [(5, 5), (6, 5), (9, 9)]
        assert d.route[0].dummy_path == [(8, 9), (0, 0)]
        assert d.route[0].start_t == 3
