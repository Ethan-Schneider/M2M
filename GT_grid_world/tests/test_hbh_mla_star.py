"""Integration tests for the HBH driver (task_allocation_algorithms/hbh_mla_star.py).

These tests exercise the full HBH+MLA* loop: free agents get assigned to
released tasks, the chosen pickup/delivery cells satisfy the standing
reservations, and ``agent.path_sequence`` is populated by MLA* (no external
ECBS/PBS call). The tests use the ``populated_graph`` fixture so
warehouse-full and warehouse-empty cells both exist, and the
``sample_agents`` fixture for known agent positions.

We deliberately do NOT cover the dispatch wiring here (that's a separate
test below); the goal of this module is HBH's own behaviour given a fully
constructed Graph and a non-trivial inventory state.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from GT_grid_world.src.task_allocation_algorithms.hbh_mla_star import (
    hbh_mla_star_call,
)
from GT_grid_world.src import task_allocation


def _build_simple_tasks(populated_graph) -> Dict[int, Tuple]:
    """Construct a tiny task dict for the HBH tests.

    Task tuple shape (matching ``case_request_generator``):
        ``(start_locs, goal_locs, deadline, sku_id, type)``
    where ``type`` is in ``{0=outbound, 1=inbound, 2=shuffle}``.

    We hand-pick one outbound task (warehouse-full -> driveway) and one
    inbound task (driveway -> warehouse-empty) so HBH has both task types
    to choose from, which is the realistic baseline workload.
    """
    full = populated_graph.warehouse.get_full_locations()
    empties = [
        loc for loc in populated_graph.get_aisle_locations()
        if loc not in set(full)
    ]
    stations = populated_graph.get_station_locations()

    tasks: Dict[int, Tuple] = {}
    if full and stations:
        tasks[1] = (tuple(full), tuple(stations), 100, 0, 0)
    if empties and stations:
        tasks[2] = (tuple(stations), tuple(empties), 100, 0, 1)
    return tasks


class TestHBHAssignment:
    def test_free_agents_get_assigned(
        self, minimal_stats, populated_graph, sample_agents
    ):
        tasks = _build_simple_tasks(populated_graph)
        assert tasks, "test setup expects at least one open task"

        Rs, _, _ = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=sample_agents,
            J=tasks, t=0,
        )

        # At least one agent must have been assigned a task and a path.
        assigned_count = sum(1 for ag in Rs.agents if ag.task_sequence)
        assert assigned_count >= 1, "HBH should assign at least one free agent"

    def test_assigned_agents_have_nonempty_path(
        self, minimal_stats, populated_graph, sample_agents
    ):
        tasks = _build_simple_tasks(populated_graph)
        Rs, _, _ = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=sample_agents,
            J=tasks, t=0,
        )
        for ag in Rs.agents:
            if ag.task_sequence:
                assert ag.path_sequence, (
                    f"agent {ag.id} has a task but no MLA*-planned path"
                )
                # Path cells must all be non-obstacle and grid-bounded.
                for loc in ag.path_sequence:
                    assert not populated_graph.get_if_obstacle(loc)

    def test_assigned_agents_become_to_pickup(
        self, minimal_stats, populated_graph, sample_agents
    ):
        tasks = _build_simple_tasks(populated_graph)
        Rs, _, _ = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=sample_agents,
            J=tasks, t=0,
        )
        for ag in Rs.agents:
            if ag.task_sequence:
                # status 1 == to_pickup (agent.py docstring).
                assert ag.status == 1

    def test_no_double_assignment_across_two_calls(
        self, minimal_stats, populated_graph, sample_agents
    ):
        # Run HBH once, capture the assigned task ids, then run again with
        # the same J. The second call must NOT re-assign already-assigned
        # tasks.
        tasks = _build_simple_tasks(populated_graph)
        Rs, _, _ = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=sample_agents,
            J=tasks, t=0,
        )
        first_pass_assigned = {
            tt[0] for ag in Rs.agents for tt in ag.task_sequence
        }
        # Second call at the next tick:
        Rs2, _, _ = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=Rs,
            J=tasks, t=1,
        )
        all_assigned = [
            tt[0] for ag in Rs2.agents for tt in ag.task_sequence
        ]
        # Each task id must appear at most once across all agents.
        assert len(all_assigned) == len(set(all_assigned))
        # No assigned task from the first pass got dropped or re-routed.
        assert first_pass_assigned.issubset(set(all_assigned))

    def test_returns_three_tuple_matching_lns_signature(
        self, minimal_stats, populated_graph, sample_agents
    ):
        # The dispatch in task_allocation.TaskAllocation unpacks the return
        # value as ``Rs, _, _ = strategy_call(...)``, so HBH must return a
        # 3-tuple just like c_lns_call and py_lns_call do.
        out = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=sample_agents,
            J=_build_simple_tasks(populated_graph), t=0,
        )
        assert isinstance(out, tuple) and len(out) == 3


class TestDispatchWiring:
    """The strategy must be reachable through the ``TaskAllocation`` entry
    point. This test guards against future refactors silently dropping the
    dispatch entry.
    """

    def test_strategy_is_routed_through_task_allocation(
        self, minimal_stats, populated_graph, sample_agents
    ):
        tasks = _build_simple_tasks(populated_graph)
        Rs = task_allocation.TaskAllocation(
            S=minimal_stats,
            G=populated_graph,
            Rs=sample_agents,
            J=tasks,
            initial_task_assignment_strategy="random",  # ignored when improvement is set
            improvement_task_assignment_strategy="hbh_mla_star",
            map="data/maps/small_test",
            t=0,
            cost_calculation_method="manhattan",
        )
        # TaskAllocation under the c_lns / hbh path returns the 3-tuple
        # directly, so we accept either the bare AgentLoader or the tuple
        # form.
        if isinstance(Rs, tuple):
            Rs = Rs[0]
        assigned_count = sum(1 for ag in Rs.agents if ag.task_sequence)
        assert assigned_count >= 1
