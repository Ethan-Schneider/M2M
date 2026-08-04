"""Tests for the TA-Hybrid task-assignment stage (paper Section 3).

The unit tests use a ``StubGraph`` with deterministic Manhattan distances so
edge-weight formulas, partition logic, and execution-time computation can be
asserted exactly. The end-to-end smoke test uses the real ``tiny_graph``
fixture from ``conftest.py`` to exercise the full pipeline with elkai's
LKH-3 solver.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from GT_grid_world.src.agent import Agent, AgentLoader
from GT_grid_world.src.task_allocation_algorithms.ta_assignment import (
    MaterializedTask,
    TASK_TYPE_INBOUND,
    TASK_TYPE_OUTBOUND,
    build_assignment_matrix,
    compute_execution_times,
    compute_per_task_L_bounds,
    materialize_schedule,
    partition_tour,
    solve_special_tsp,
    ta_assignment_plan,
)


# ---------------------------------------------------------------------------
# Stubs: deterministic Manhattan distances + minimal warehouse/driveway state
# ---------------------------------------------------------------------------
class _StubInventory:
    def __init__(
        self,
        sku_instances: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        empty_locations: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self._sku_instances = sku_instances or {}
        self._empty = list(empty_locations or [])

    def get_sku_instances(self, sku_id: int) -> Iterable[Tuple[int, int]]:
        return list(self._sku_instances.get(sku_id, []))

    def get_empty_locations(self) -> Iterable[Tuple[int, int]]:
        return list(self._empty)


class _StubGraph:
    """Graph stub returning Manhattan distance between any two cells."""

    def __init__(
        self,
        aisles: Sequence[Tuple[int, int]],
        stations: Sequence[Tuple[int, int]],
        sku_instances: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        warehouse_empty: Optional[List[Tuple[int, int]]] = None,
        driveway_empty: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self._aisles = list(aisles)
        self._stations = list(stations)
        self.warehouse = _StubInventory(
            sku_instances=sku_instances,
            empty_locations=warehouse_empty or [],
        )
        self.driveway = _StubInventory(
            sku_instances={},
            empty_locations=driveway_empty if driveway_empty is not None else list(stations),
        )

    def get_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def get_aisle_locations(self) -> List[Tuple[int, int]]:
        return list(self._aisles)

    def get_station_locations(self) -> List[Tuple[int, int]]:
        return list(self._stations)


def _make_agent(agent_id: int, home: Tuple[int, int]) -> Agent:
    """``Agent(agent_id, state=home)`` -- ``Agent`` derives ``home`` from state."""
    return Agent(agent_id=agent_id, state=home)


def _materialized(
    schedule_row_index: int,
    pickup: Tuple[int, int],
    delivery: Tuple[int, int],
    release_time: int = 0,
    deadline: int = 100,
    sku_id: int = 1,
    task_type: int = TASK_TYPE_OUTBOUND,
) -> MaterializedTask:
    return MaterializedTask(
        schedule_row_index=schedule_row_index,
        release_time=release_time,
        deadline=deadline,
        sku_id=sku_id,
        task_type=task_type,
        pickup=pickup,
        delivery=delivery,
    )


# ---------------------------------------------------------------------------
# materialize_schedule
# ---------------------------------------------------------------------------
class TestMaterializeSchedule:
    def test_empty_schedule_returns_empty_list(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        assert materialize_schedule(np.zeros((0, 4)), G) == []

    def test_none_schedule_returns_empty_list(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        assert materialize_schedule(None, G) == []  # type: ignore[arg-type]

    def test_outbound_picks_warehouse_cell_holding_sku(self):
        # sku 2 lives at (1, 1) and (3, 3). _first_sorted should pick (1, 1).
        G = _StubGraph(
            aisles=[(1, 1), (3, 3)],
            stations=[(10, 10), (10, 11)],
            sku_instances={2: [(3, 3), (1, 1)]},
            driveway_empty=[(10, 11), (10, 10)],
        )
        # schedule has sku_id stored as raw (sku_id - sku_id_offset). offset=1 -> stored 1
        schedule = np.array([[0, 50, 1, TASK_TYPE_OUTBOUND]], dtype=np.int64)
        out = materialize_schedule(schedule, G, sku_id_offset=1)
        assert len(out) == 1
        assert out[0].pickup == (1, 1)
        assert out[0].delivery == (10, 10)
        assert out[0].sku_id == 2
        assert out[0].task_type == TASK_TYPE_OUTBOUND

    def test_inbound_picks_driveway_to_empty_warehouse(self):
        G = _StubGraph(
            aisles=[(2, 2), (5, 5)],
            stations=[(20, 20), (20, 21)],
            warehouse_empty=[(5, 5), (2, 2)],
            driveway_empty=[(20, 21), (20, 20)],
        )
        schedule = np.array([[0, 60, 0, TASK_TYPE_INBOUND]], dtype=np.int64)
        out = materialize_schedule(schedule, G, sku_id_offset=1)
        assert out[0].pickup == (20, 20)
        assert out[0].delivery == (2, 2)
        assert out[0].task_type == TASK_TYPE_INBOUND

    def test_outbound_falls_back_when_sku_absent_from_warehouse(self):
        # SKU 3 is requested but the warehouse holds no SKU 3 at t=0.
        # We expect a fallback to the warehouse-region centroid, not a crash.
        G = _StubGraph(
            aisles=[(1, 1), (3, 3), (5, 5)],
            stations=[(10, 10)],
            sku_instances={2: [(1, 1)]},  # only sku 2 is in stock
            driveway_empty=[(10, 10)],
        )
        schedule = np.array([[0, 100, 2, TASK_TYPE_OUTBOUND]], dtype=np.int64)
        out = materialize_schedule(schedule, G, sku_id_offset=1)
        assert out[0].pickup in {(1, 1), (3, 3), (5, 5)}  # fallback centroid snaps to aisle

    def test_unknown_task_type_raises(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        schedule = np.array([[0, 10, 1, 7]], dtype=np.int64)
        with pytest.raises(ValueError, match="Unsupported task_type"):
            materialize_schedule(schedule, G, sku_id_offset=1)

    def test_orders_rows_by_release_time_stable(self):
        G = _StubGraph(
            aisles=[(0, 0)],
            stations=[(5, 5)],
            sku_instances={2: [(0, 0)], 3: [(0, 0)]},
            driveway_empty=[(5, 5)],
        )
        # Rows intentionally out of order: r=10, r=5, r=10.
        # After stable sort by release_time the order is rows [1, 0, 2].
        schedule = np.array(
            [
                [10, 50, 1, TASK_TYPE_OUTBOUND],  # sku 2
                [5,  50, 2, TASK_TYPE_OUTBOUND],  # sku 3
                [10, 50, 1, TASK_TYPE_OUTBOUND],  # sku 2 again
            ],
            dtype=np.int64,
        )
        out = materialize_schedule(schedule, G, sku_id_offset=1)
        assert [m.schedule_row_index for m in out] == [1, 0, 2]
        # First entry corresponds to the r=5 row -> task_id == 1 at runtime
        assert out[0].release_time == 5


# ---------------------------------------------------------------------------
# build_assignment_matrix
# ---------------------------------------------------------------------------
class TestBuildAssignmentMatrix:
    def test_shape_is_m_plus_n_square(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0)), _make_agent(1, (1, 1))]
        tasks = [_materialized(0, (2, 0), (5, 0))]
        cost = build_assignment_matrix(agents, tasks, G)
        assert cost.shape == (3, 3)

    def test_diagonal_is_zero(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0))]
        tasks = [_materialized(0, (3, 0), (3, 5))]
        cost = build_assignment_matrix(agents, tasks, G)
        assert cost[0, 0] == 0
        assert cost[1, 1] == 0

    def test_alpha_to_alpha_block_is_zero(self):
        # M=3 agents, N=2 tasks -> top-left 3x3 block must be all zeros.
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(i, (i, 0)) for i in range(3)]
        tasks = [
            _materialized(0, (4, 4), (6, 6)),
            _materialized(1, (8, 8), (10, 10)),
        ]
        cost = build_assignment_matrix(agents, tasks, G)
        assert (cost[:3, :3] == 0).all()

    def test_alpha_to_tau_uses_travel_when_travel_exceeds_release(self):
        # Travel = dist((0,0), (0,5)) = 5; release_time = 2; weight = max(5, 2) = 5
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0))]
        tasks = [_materialized(0, (0, 5), (0, 10), release_time=2)]
        cost = build_assignment_matrix(agents, tasks, G)
        assert cost[0, 1] == 5  # alpha_0 -> tau_0

    def test_alpha_to_tau_uses_release_when_release_exceeds_travel(self):
        # Travel = 2; release = 9 -> weight = 9 (release binding)
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0))]
        tasks = [_materialized(0, (0, 2), (0, 10), release_time=9)]
        cost = build_assignment_matrix(agents, tasks, G)
        assert cost[0, 1] == 9

    def test_tau_to_tau_is_traverse_plus_transition(self):
        # task_0: (0,0) -> (0,5), traverse=5
        # task_1: (0,8) -> (0,12)
        # transition from delivery of 0 to pickup of 1: dist((0,5),(0,8)) = 3
        # expected w(tau_0, tau_1) = 5 + 3 = 8
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (50, 50))]  # far away -- irrelevant for this block
        tasks = [
            _materialized(0, (0, 0), (0, 5)),
            _materialized(1, (0, 8), (0, 12)),
        ]
        cost = build_assignment_matrix(agents, tasks, G)
        assert cost[1, 2] == 8  # alpha=0, tau_0=1, tau_1=2

    def test_tau_to_alpha_is_just_task_traverse(self):
        # task_0: (0,0) -> (0,5), traverse=5
        # alpha_0: (100,100). The edge w(tau_0, alpha_0) does NOT depend on alpha's
        # location per the paper -- it is the per-task traversal distance.
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (100, 100)), _make_agent(1, (200, 200))]
        tasks = [_materialized(0, (0, 0), (0, 5))]
        cost = build_assignment_matrix(agents, tasks, G)
        # tau_0 is index M+0=2; agents at 0 and 1.
        assert cost[2, 0] == 5
        assert cost[2, 1] == 5


# ---------------------------------------------------------------------------
# solve_special_tsp
# ---------------------------------------------------------------------------
class TestSolveSpecialTSP:
    def test_empty_matrix_returns_empty_tour(self):
        assert solve_special_tsp(np.zeros((0, 0), dtype=np.int64)) == []

    def test_single_node_matrix_returns_singleton(self):
        assert solve_special_tsp(np.zeros((1, 1), dtype=np.int64)) == [0]

    def test_returns_permutation_of_all_nodes(self):
        # Asymmetric cost matrix
        cost = np.array(
            [
                [0, 4, 9, 7],
                [5, 0, 6, 8],
                [10, 7, 0, 3],
                [6, 9, 2, 0],
            ],
            dtype=np.int64,
        )
        tour = solve_special_tsp(cost)
        assert sorted(tour) == [0, 1, 2, 3]
        assert tour[0] == 0  # elkai rotates to start at vertex 0

    def test_finds_optimal_tour_on_small_symmetric_instance(self):
        # Symmetric square; optimal tour is 0->1->2->3->0 (or reverse) cost = 4.
        cost = np.array(
            [
                [0, 1, 100, 1],
                [1, 0, 1, 100],
                [100, 1, 0, 1],
                [1, 100, 1, 0],
            ],
            dtype=np.int64,
        )
        tour = solve_special_tsp(cost)
        assert sorted(tour) == [0, 1, 2, 3]
        # Recompute the tour cost
        total = sum(cost[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
        assert total == 4


# ---------------------------------------------------------------------------
# partition_tour
# ---------------------------------------------------------------------------
class TestPartitionTour:
    def test_single_agent_gets_all_tasks_in_tour_order(self):
        # M=1, N=3. Tour: [0, 1, 2, 3]. Tasks are 1,2,3 (offset by M=1).
        sequences = partition_tour([0, 1, 2, 3], num_agents=1)
        assert sequences == {0: [0, 1, 2]}

    def test_two_agents_split_at_agent_vertices(self):
        # M=2 (agents 0,1), N=4 (tasks 2,3,4,5).
        # Tour [0, 2, 3, 1, 4, 5] -> agent 0 gets tasks {2,3} = [0,1],
        # agent 1 gets tasks {4,5} = [2,3].
        sequences = partition_tour([0, 2, 3, 1, 4, 5], num_agents=2)
        assert sequences == {0: [0, 1], 1: [2, 3]}

    def test_rotates_tour_when_first_vertex_is_a_task(self):
        # M=2, N=3. Tour [3, 1, 4, 0, 2]
        # rotates to [0, 2, 3, 1, 4] -> agent 0: [0, 1], agent 1: [2]
        # (task 2 is the suffix that belongs to agent 1 because the cycle closes back to agent 0)
        sequences = partition_tour([3, 1, 4, 0, 2], num_agents=2)
        assert sequences == {0: [0, 1], 1: [2]}

    def test_agents_with_no_tasks_get_empty_list(self):
        # M=3, N=2. Tour [0, 2, 3, 1, 2]... actually need a clean partition.
        # Tour [0, 2, 1, 3, 2]: M=3, tasks 3,4 -> M+0=3, M+1=4
        # Use: M=3, N=2, tour [0, 3, 4, 1, 2]
        # -> agent 0: [0,1] (tasks 3,4); agent 1: []; agent 2: []
        sequences = partition_tour([0, 3, 4, 1, 2], num_agents=3)
        assert sequences == {0: [0, 1], 1: [], 2: []}

    def test_empty_tour_returns_all_empty_lists(self):
        sequences = partition_tour([], num_agents=2)
        assert sequences == {0: [], 1: []}

    def test_raises_when_tour_has_no_agent_vertices(self):
        with pytest.raises(ValueError, match="no agent vertices"):
            partition_tour([3, 4, 5], num_agents=2)  # all task vertices


# ---------------------------------------------------------------------------
# compute_execution_times
# ---------------------------------------------------------------------------
class TestComputeExecutionTimes:
    def test_empty_sequence_yields_zero(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0))]
        et = compute_execution_times({0: []}, agents, [], G)
        assert et == {0: 0}

    def test_single_task_is_travel_plus_traverse(self):
        # Agent at (0,0); task pickup (0,3) delivery (0,7); release=0.
        # start = max(travel=3, release=0) = 3
        # M_i = start + traverse = 3 + 4 = 7
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0))]
        tasks = [_materialized(0, (0, 3), (0, 7), release_time=0)]
        et = compute_execution_times({0: [0]}, agents, tasks, G)
        assert et == {0: 7}

    def test_release_time_delays_first_task_start(self):
        # Travel=3 but release=10 -> start=10; M_i = 10 + 4 = 14.
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0))]
        tasks = [_materialized(0, (0, 3), (0, 7), release_time=10)]
        et = compute_execution_times({0: [0]}, agents, tasks, G)
        assert et == {0: 14}

    def test_chain_release_time_binding_between_tasks(self):
        # Agent (0,0); t0 pickup (0,1) delivery (0,2) r=0;
        # t1 pickup (0,4) delivery (0,5) r=100.
        # start(t0) = max(1, 0) = 1. start(t1) = max(1 + 1 + 2, 100) = 100.
        # M_i = 100 + 1 = 101.
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        agents = [_make_agent(0, (0, 0))]
        tasks = [
            _materialized(0, (0, 1), (0, 2), release_time=0),
            _materialized(1, (0, 4), (0, 5), release_time=100),
        ]
        et = compute_execution_times({0: [0, 1]}, agents, tasks, G)
        assert et == {0: 101}


# ---------------------------------------------------------------------------
# compute_per_task_L_bounds (paper Section 3.2)
# ---------------------------------------------------------------------------
class TestComputePerTaskLBounds:
    def test_empty_plan_yields_empty_bounds(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        bounds = compute_per_task_L_bounds({}, {}, G)
        assert bounds == {}

    def test_single_task_uses_global_L_minus_traverse(self):
        # Global L = 10, single task pickup (0, 0) -> delivery (0, 4).
        # tail = traverse = 4 -> L_{0, 0} = 10 - 4 = 6.
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        tasks = [_materialized(0, (0, 0), (0, 4), release_time=0)]
        plan = {7: tasks}
        bounds = compute_per_task_L_bounds(plan, {7: 10}, G)
        assert bounds == {(7, 0): 6}

    def test_chain_backward_dp(self):
        # Two tasks for a single agent.
        # tasks[0]: pickup (0, 0) delivery (0, 2)  -- traverse 2
        # tasks[1]: pickup (0, 5) delivery (0, 7)  -- traverse 2
        # transition delivery_0 (0,2) -> pickup_1 (0,5) = 3
        # Global L = 100.
        # tail(K=1) = traverse(t1) = 2
        # L_{i, t1} = 100 - 2 = 98
        # tail(0)   = traverse(t0) + transition + tail(1)
        #           = 2 + 3 + 2 = 7
        # L_{i, t0} = 100 - 7 = 93
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        tasks = [
            _materialized(0, (0, 0), (0, 2), release_time=0),
            _materialized(1, (0, 5), (0, 7), release_time=0),
        ]
        plan = {3: tasks}
        bounds = compute_per_task_L_bounds(plan, {3: 100}, G)
        assert bounds == {(3, 0): 93, (3, 1): 98}

    def test_global_L_is_max_across_agents(self):
        # Two agents with different M_i; global L = max.
        # Agent 0: 1 task with traverse 5  -> M_0 = ?
        # Agent 1: 1 task with traverse 1  -> M_1 = ?
        # We hand exec_times directly so we don't need to compute them
        # here -- the function just takes max(exec_times).
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        plan = {
            0: [_materialized(0, (0, 0), (0, 5), release_time=0)],
            1: [_materialized(1, (1, 0), (1, 1), release_time=0)],
        }
        # Suppose agent 0 has M_0 = 50, agent 1 has M_1 = 5. Global L = 50.
        # L_{0, 0} = 50 - traverse(0) = 50 - 5 = 45.
        # L_{1, 1} = 50 - traverse(1) = 50 - 1 = 49.  (looser, slack agent)
        bounds = compute_per_task_L_bounds(plan, {0: 50, 1: 5}, G)
        assert bounds == {(0, 0): 45, (1, 1): 49}

    def test_per_task_L_is_monotone_nondecreasing_along_sequence(self):
        # A core invariant: walking forward through an agent's task
        # sequence, L_{i,k} only increases (later tasks have less
        # remaining work, so a looser deadline). This is what AMAPF
        # relies on -- earlier tasks get tighter meta windows.
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        tasks = [
            _materialized(0, (0, 0), (0, 1), release_time=0),
            _materialized(1, (0, 3), (0, 4), release_time=0),
            _materialized(2, (0, 6), (0, 7), release_time=0),
        ]
        plan = {0: tasks}
        bounds = compute_per_task_L_bounds(plan, {0: 100}, G)
        L0 = bounds[(0, 0)]
        L1 = bounds[(0, 1)]
        L2 = bounds[(0, 2)]
        assert L0 <= L1 <= L2

    def test_skips_agents_with_no_tasks(self):
        # An agent with empty plan -> no entries in bounds for it.
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        plan = {
            0: [_materialized(0, (0, 0), (0, 1), release_time=0)],
            1: [],  # idle agent
        }
        bounds = compute_per_task_L_bounds(plan, {0: 10, 1: 0}, G)
        assert (0, 0) in bounds
        assert all(agent_id != 1 for agent_id, _ in bounds.keys())


# ---------------------------------------------------------------------------
# ta_assignment_plan (end-to-end)
# ---------------------------------------------------------------------------
class TestTaAssignmentPlan:
    def test_empty_schedule_returns_empty_plans(self):
        G = _StubGraph(aisles=[(0, 0)], stations=[(5, 5)])
        Rs = AgentLoader([_make_agent(0, (0, 0)), _make_agent(1, (1, 1))])
        plan, et = ta_assignment_plan(G, Rs, np.zeros((0, 4)))
        assert plan == {0: [], 1: []}
        assert et == {0: 0, 1: 0}

    def test_smoke_pipeline_runs_on_small_schedule(self):
        # Two agents, two outbound tasks, 5x5 workspace.
        G = _StubGraph(
            aisles=[(0, 0), (1, 0), (2, 0)],
            stations=[(0, 4), (1, 4), (2, 4)],
            sku_instances={2: [(0, 0)], 3: [(2, 0)]},
            driveway_empty=[(0, 4), (1, 4), (2, 4)],
        )
        Rs = AgentLoader(
            [_make_agent(0, (3, 0)), _make_agent(1, (3, 2))]
        )
        schedule = np.array(
            [
                [0, 50, 1, TASK_TYPE_OUTBOUND],   # sku 2 -> pickup (0,0)
                [0, 50, 2, TASK_TYPE_OUTBOUND],   # sku 3 -> pickup (2,0)
            ],
            dtype=np.int64,
        )
        plan, et = ta_assignment_plan(G, Rs, schedule, sku_id_offset=1)
        assert set(plan.keys()) == {0, 1}
        # All tasks accounted for exactly once
        all_assigned = [t.schedule_row_index for tasks in plan.values() for t in tasks]
        assert sorted(all_assigned) == [0, 1]
        # Execution times are non-negative integers
        assert all(isinstance(v, int) and v >= 0 for v in et.values())

    def test_single_agent_owns_every_task(self):
        G = _StubGraph(
            aisles=[(0, 0), (1, 0), (2, 0)],
            stations=[(0, 5)],
            sku_instances={2: [(0, 0)]},
            driveway_empty=[(0, 5)],
        )
        Rs = AgentLoader([_make_agent(0, (5, 0))])
        schedule = np.array(
            [
                [0,  60, 1, TASK_TYPE_OUTBOUND],
                [5,  60, 1, TASK_TYPE_OUTBOUND],
                [10, 60, 1, TASK_TYPE_OUTBOUND],
            ],
            dtype=np.int64,
        )
        plan, _ = ta_assignment_plan(G, Rs, schedule, sku_id_offset=1)
        assert len(plan[0]) == 3
        assert sorted(t.schedule_row_index for t in plan[0]) == [0, 1, 2]

    def test_plan_is_deterministic_across_runs(self):
        G = _StubGraph(
            aisles=[(0, 0), (1, 0), (2, 0)],
            stations=[(0, 4), (1, 4)],
            sku_instances={2: [(0, 0)], 3: [(2, 0)]},
            driveway_empty=[(0, 4), (1, 4)],
        )
        Rs = AgentLoader([_make_agent(0, (3, 0)), _make_agent(1, (3, 2))])
        schedule = np.array(
            [
                [0, 50, 1, TASK_TYPE_OUTBOUND],
                [0, 50, 2, TASK_TYPE_OUTBOUND],
            ],
            dtype=np.int64,
        )
        plan_a, _ = ta_assignment_plan(G, Rs, schedule, sku_id_offset=1)
        plan_b, _ = ta_assignment_plan(G, Rs, schedule, sku_id_offset=1)
        # Two runs should produce the same task assignments.
        keys_a = {aid: [t.schedule_row_index for t in tasks] for aid, tasks in plan_a.items()}
        keys_b = {aid: [t.schedule_row_index for t in tasks] for aid, tasks in plan_b.items()}
        assert keys_a == keys_b


# ---------------------------------------------------------------------------
# Integration with the real M2M Graph fixture (smoke only)
# ---------------------------------------------------------------------------
class TestRealGraphSmoke:
    def test_runs_on_populated_graph(self, populated_graph, sample_agents, seeded_rng):
        # ``populated_graph`` is the small_test map with the warehouse 30%
        # pre-filled, so the warehouse already has SKU instances we can
        # target with an outbound schedule row.
        warehouse = populated_graph.warehouse
        sku_counts = {
            sku: len(warehouse.get_sku_instances(sku)) for sku in range(1, 6)
        }
        live_sku = max(sku_counts, key=lambda s: sku_counts[s])
        if sku_counts[live_sku] == 0:
            pytest.skip("no SKUs landed in warehouse for this fixture state")

        # sku_id in the schedule is stored raw (offset is added later by
        # materialize_schedule via the default ``sku_id_offset=1``).
        schedule = np.array(
            [
                [0, 100, live_sku - 1, TASK_TYPE_OUTBOUND],
                [5, 100, live_sku - 1, TASK_TYPE_OUTBOUND],
            ],
            dtype=np.int64,
        )
        plan, et = ta_assignment_plan(populated_graph, sample_agents, schedule)
        # Every task should be assigned to exactly one agent.
        all_assigned = [t.schedule_row_index for tasks in plan.values() for t in tasks]
        assert sorted(all_assigned) == [0, 1]
        # Execution times are well-formed.
        assert all(v >= 0 for v in et.values())
