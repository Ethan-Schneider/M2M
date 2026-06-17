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


class TestCollisionFreeness:
    """Regression coverage for the multi-agent reservation invariants.

    The HBH driver must produce a set of agent paths that, when stepped
    through tick-by-tick, never place two agents at the same cell at the
    same time. The bug we hit in the 30%/30-bot/600-tick HBH baseline was
    that idle free agents were only reserved at the *current* tick, so
    later-planned agents' MLA* searches happily routed through their
    cells in any future tick. The first concrete vertex collision in that
    run was at simulator tick 4 (agent 19 idle at (19,22), agent 20
    walked into the cell on its first move from (19,23)).

    These tests verify the invariant at the scale ``populated_graph``
    affords; the integrated baseline is the larger soak test.
    """

    def _no_two_agents_share_a_cell_simultaneously(self, Rs, t0: int):
        """Step every agent's planned path forward and assert no collision.

        For each tick ``t in [t0, t0 + max_path_len]`` compute every
        agent's predicted state; assert no two agents share that state.
        """
        max_len = max(len(ag.path_sequence) for ag in Rs.agents) if Rs.agents else 0
        # tick 0 is "current state", subsequent ticks consume one path
        # entry each. For agents whose path is shorter than max_len, hold
        # them at their last cell (matches the simulator's behaviour when
        # ``path_sequence`` runs out).
        for k in range(max_len + 1):
            seen = {}
            for ag in Rs.agents:
                if k == 0:
                    loc = ag.state
                else:
                    if k - 1 < len(ag.path_sequence):
                        loc = ag.path_sequence[k - 1]
                    elif ag.path_sequence:
                        loc = ag.path_sequence[-1]
                    else:
                        loc = ag.state
                assert loc not in seen, (
                    f"vertex collision at relative tick {k}: agents "
                    f"{seen[loc]} and {ag.id} both at {loc}"
                )
                seen[loc] = ag.id

    def test_paths_from_single_hbh_call_are_collision_free(
        self, minimal_stats, populated_graph, sample_agents
    ):
        tasks = _build_simple_tasks(populated_graph)
        Rs, _, _ = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=sample_agents,
            J=tasks, t=0,
        )
        self._no_two_agents_share_a_cell_simultaneously(Rs, t0=0)

    def test_idle_agent_cell_blocks_other_agents_through_planning(
        self, minimal_stats, populated_graph, sample_agents
    ):
        """If agent 0 sits idle at a cell with no task, any other agent
        planned in the same HBH call must not include that cell in its
        path at any future tick. This is exactly the invariant that the
        ``is_permanent_terminal=True`` fix in the rebuild loop preserves.
        """
        # Force agent 0 to have no task at all (idle, status=0, no path).
        # The other two agents are free and will be planned by HBH below.
        idle_cell = sample_agents.agents[0].state
        tasks = _build_simple_tasks(populated_graph)
        Rs, _, _ = hbh_mla_star_call(
            S=minimal_stats, G=populated_graph, Rs=sample_agents,
            J=tasks, t=0,
        )
        # Find agent 0 (the idle one). It may have been pushed off by the
        # endpoint-clearing pass if it happens to sit on a pickup cell;
        # if so, this scenario doesn't exercise the bug and we skip.
        idle_ag = Rs.get_agent(0)
        if idle_ag.path_sequence:
            pytest.skip(
                "agent 0 was pushed off its cell by endpoint clearing; "
                "this run doesn't exercise the idle-agent invariant"
            )
        # Nobody else's planned path may visit ``idle_cell`` at any tick.
        for ag in Rs.agents:
            if ag.id == 0:
                continue
            for step_idx, loc in enumerate(ag.path_sequence):
                assert loc != idle_cell, (
                    f"agent {ag.id} routed through idle agent 0's cell "
                    f"{idle_cell} at relative tick {step_idx + 1}"
                )


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
