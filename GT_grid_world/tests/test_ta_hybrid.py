"""Tests for TA-Hybrid's PlanPathsToDelivery (paper Algorithm 2).

We use a minimal in-memory ``_MockGraph`` plus a fresh
``ReservationTable`` per test so every assertion is fully controllable.
The wrapper module (:mod:`path_finding_algorithms.icbs_planner`) is
already covered by its own tests; here we focus on the new
two-phase orchestration: (sub-path ICBS) -> (dummy-path A*) -> (joint
reservation + path stitching).

Conventions exercised by these tests:

-   ``sub_path[0] == current_loc``, ``sub_path[-1] == delivery_loc``.
-   ``dummy_path[0]`` is the first move AFTER the delivery cell, so
    ``dummy_path[-1] == parking_loc``. The delivery cell is **not** the
    first element of ``dummy_path``.
-   Time is absolute: a reservation passed in at ``t = t_now + k`` is
    seen as being ``k`` ticks into the future. The function converts
    internally to CBS-relative time for ICBS but preserves absolute time
    for the dummy-path search.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest

from GT_grid_world.src.path_finding_algorithms.mla_star import ReservationTable
from GT_grid_world.src.task_allocation_algorithms.ta_hybrid import (
    DeliveryPlanRequest,
    DeliveryPlanResult,
    PickupPlanRequest,
    PickupPlanResult,
    _detect_edge_collision,
    _partition_group2,
    _resolve_edge_collisions,
    _swap_walkers_at,
    plan_paths_to_delivery,
    plan_paths_to_pickup,
)
from GT_grid_world.src.task_allocation_algorithms.ta_hybrid_amapf import AmapfWalker


Loc = Tuple[int, int]


# ---------------------------------------------------------------------------
# Mock Graph (same minimal interface the wrapper + mla_star_single_goal need)
# ---------------------------------------------------------------------------
class _MockGraph:
    """Tiny stand-in for ``Graph`` covering the methods this module touches."""

    def __init__(self, my_map: List[List[bool]]) -> None:
        self._map = my_map
        self.height = len(my_map)
        self.width = len(my_map[0]) if my_map else 0

    def get_if_obstacle(self, node: Tuple[int, int]) -> bool:
        r, c = node
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True
        return self._map[r][c]

    def get_graph_size(self) -> Tuple[int, int]:
        # mla_star_single_goal uses this for in-bounds checks.
        return (self.height, self.width)


def _build_open_grid(width: int, height: int, obstacles: Sequence[Loc] = ()) -> _MockGraph:
    grid = [[False] * width for _ in range(height)]
    for r, c in obstacles:
        grid[r][c] = True
    return _MockGraph(grid)


def _path_collides_with(
    path: List[Loc], rt: ReservationTable, start_t: int, agent_id: int
) -> bool:
    """Check if ``path`` (absolute occupancy starting at ``start_t``) hits
    any reservation in ``rt`` that doesn't belong to ``agent_id``."""
    for i, cell in enumerate(path):
        t = start_t + i
        if rt.is_vertex_reserved(t, cell, ignore_agent=agent_id):
            return True
    for i in range(1, len(path)):
        t = start_t + i
        if rt.is_edge_reserved(t, path[i - 1], path[i], ignore_agent=agent_id):
            return True
    return False


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------
class TestEmptyInputs:
    def test_no_requests_returns_empty_dict(self):
        G = _build_open_grid(width=4, height=4)
        result = plan_paths_to_delivery(
            G, requests=[], other_agent_reservations=ReservationTable(), t_now=0
        )
        assert result == {}

    def test_single_agent_no_other_reservations(self):
        # 5x5 open grid, agent at (1, 0), delivery (1, 3), parking (1, 1).
        # We expect a 3-move sub_path and a short dummy path back.
        G = _build_open_grid(width=5, height=5)
        req = DeliveryPlanRequest(
            agent_id=42,
            current_loc=(1, 0),
            delivery_loc=(1, 3),
            parking_loc=(1, 1),
        )
        result = plan_paths_to_delivery(
            G, requests=[req], other_agent_reservations=ReservationTable(), t_now=0
        )
        assert result is not None
        assert 42 in result
        r = result[42]
        assert r.sub_path[0] == (1, 0)
        assert r.sub_path[-1] == (1, 3)
        assert r.dummy_path[-1] == (1, 1)
        # sub_path includes start cell -> 4 cells for a 3-move walk
        assert len(r.sub_path) == 4


# ---------------------------------------------------------------------------
# Path-stitching invariants (paper Section 5.2 semantics)
# ---------------------------------------------------------------------------
class TestPathStitching:
    def test_sub_path_anchors_correctly(self):
        G = _build_open_grid(width=6, height=3)
        req = DeliveryPlanRequest(0, (1, 0), (1, 4), (1, 1))
        result = plan_paths_to_delivery(G, [req], ReservationTable(), t_now=10)
        assert result is not None
        r = result[0]
        assert r.sub_path[0] == req.current_loc
        assert r.sub_path[-1] == req.delivery_loc

    def test_dummy_path_starts_after_delivery_and_ends_at_parking(self):
        G = _build_open_grid(width=6, height=3)
        req = DeliveryPlanRequest(0, (1, 0), (1, 4), (1, 1))
        result = plan_paths_to_delivery(G, [req], ReservationTable(), t_now=0)
        assert result is not None
        r = result[0]
        # dummy_path's first cell is a NEIGHBOUR of delivery (or delivery
        # itself if a wait was needed); it must not be a no-op duplicate
        # of the agent's current location.
        assert r.dummy_path[-1] == req.parking_loc
        # The concatenation that the outer driver will use as the agent's
        # future positions: sub_path[1:] + dummy_path -- must not have any
        # internal teleport (each consecutive pair is adjacent or equal).
        future = r.sub_path[1:] + r.dummy_path
        prev = req.current_loc
        for cell in future:
            dr = abs(cell[0] - prev[0])
            dc = abs(cell[1] - prev[1])
            assert dr + dc <= 1, f"teleport from {prev} to {cell}"
            prev = cell

    def test_zero_length_sub_path_when_current_equals_delivery(self):
        # If the agent's current cell already equals the delivery cell,
        # ICBS returns a single-cell path. The dummy-path planning then
        # has to handle delivery == start.
        G = _build_open_grid(width=4, height=3)
        req = DeliveryPlanRequest(0, (1, 1), (1, 1), (1, 2))
        result = plan_paths_to_delivery(G, [req], ReservationTable(), t_now=0)
        assert result is not None
        r = result[0]
        assert r.sub_path == [(1, 1)]
        # Dummy path should walk from (1,1) -> (1,2)
        assert r.dummy_path[-1] == (1, 2)


# ---------------------------------------------------------------------------
# External reservations are honoured
# ---------------------------------------------------------------------------
class TestExternalReservations:
    def test_corridor_with_blocked_cell_forces_detour(self):
        # 3-row tall open grid. Reserve (1,2) at t=2 (the cell the
        # agent would normally pass through). Agent must detour.
        G = _build_open_grid(width=6, height=3)
        rt = ReservationTable()
        # Use reserve_path to set a single-cell occupancy at t=2.
        rt.reserve_path(
            agent_id=99,
            start_loc=(1, 2),
            path=[],  # no successors -> just occupies (1,2) at t=2
            start_t=2,
            is_permanent_terminal=False,
        )
        req = DeliveryPlanRequest(0, (1, 0), (1, 4), (0, 0))
        result = plan_paths_to_delivery(G, [req], rt, t_now=0)
        assert result is not None
        r = result[0]
        # Verify the agent isn't at (1,2) at t=2 (== sub_path[2]).
        assert r.sub_path[2] != (1, 2)

    def test_dummy_paths_do_not_collide_with_each_other(self):
        # Two agents deliver to corners and park at opposite corners,
        # so the dummy paths route through disjoint regions of the grid
        # and don't require intermediate-cell waits to coexist. The
        # paper's goal-test gate handles the harder "two dummies must
        # share a corridor" case via ICBS retry; our two-phase
        # approximation does not (see module docstring), so we exercise
        # the well-conditioned case here.
        G = _build_open_grid(width=7, height=7)
        req_a = DeliveryPlanRequest(
            agent_id=1,
            current_loc=(0, 0),
            delivery_loc=(0, 6),
            parking_loc=(0, 5),
        )
        req_b = DeliveryPlanRequest(
            agent_id=2,
            current_loc=(6, 0),
            delivery_loc=(6, 6),
            parking_loc=(6, 5),
        )
        result = plan_paths_to_delivery(G, [req_a, req_b], ReservationTable(), t_now=0)
        assert result is not None
        r_a = result[1]
        r_b = result[2]

        # Each agent's full reservation footprint (sub_path then dummy
        # starting at delivery arrival) must not collide with the other's.
        # We check by re-building a reservation table from one agent and
        # asserting the other's path is collision-free with it.
        rt_a = ReservationTable()
        rt_a.reserve_path(
            agent_id=1,
            start_loc=r_a.sub_path[0],
            path=list(r_a.sub_path[1:]),
            start_t=0,
            is_permanent_terminal=False,
        )
        delivery_arrival_a = len(r_a.sub_path) - 1
        rt_a.reserve_path(
            agent_id=1,
            start_loc=req_a.delivery_loc,
            path=list(r_a.dummy_path),
            start_t=delivery_arrival_a,
            is_permanent_terminal=True,
        )

        # Build agent B's full executed cells (in absolute time)
        b_cells = [r_b.sub_path[0]] + list(r_b.sub_path[1:])
        # Then dummy path starting at delivery_b at t = len(sub_path_b) - 1
        # We pass agent_id=2 to ignore B's own reservations (none here).
        assert not _path_collides_with(b_cells, rt_a, start_t=0, agent_id=2)
        delivery_arrival_b = len(r_b.sub_path) - 1
        b_dummy_absolute = [req_b.delivery_loc] + list(r_b.dummy_path)
        assert not _path_collides_with(
            b_dummy_absolute, rt_a, start_t=delivery_arrival_b, agent_id=2
        )

    def test_horizon_truncation_does_not_drop_near_term_reservations(self):
        # An external reservation at t=1 (well inside the horizon) must be
        # respected even when horizon is small (so the planner can't
        # accidentally drop it).
        G = _build_open_grid(width=4, height=3)
        rt = ReservationTable()
        rt.reserve_path(
            agent_id=99,
            start_loc=(1, 1),
            path=[],
            start_t=1,
            is_permanent_terminal=False,
        )
        req = DeliveryPlanRequest(0, (1, 0), (1, 2), (1, 0))
        # Tiny horizon (just enough for a 2-move sub path + ~2-move dummy).
        result = plan_paths_to_delivery(G, [req], rt, t_now=0, horizon=10)
        assert result is not None
        # Agent must NOT be at (1,1) at t=1.
        assert result[0].sub_path[1] != (1, 1)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
class TestFailureModes:
    def test_unreachable_parking_returns_none(self):
        # 3x3 grid with parking cell walled off after the agent reaches
        # delivery: dummy-path planning should fail and the function
        # should return None (not raise).
        G = _MockGraph(
            [
                [False, True, False],   # (0,1) is a wall isolating (0,2)
                [False, False, False],
                [False, True, False],   # (2,1) is also a wall isolating (0,2)/(2,2)
            ]
        )
        # parking is (0, 2), but the only paths into it from the left
        # half go through (0, 1) or (2, 1), both walls. (0, 2) is also
        # bounded by (1, 2) which is open, but (1, 2) connects to (0, 2)
        # via (0, 2) <-> (1, 2). Let me re-pick.
        #
        # To make parking actually unreachable, wall the corner:
        G_iso = _MockGraph(
            [
                [False, True, True],
                [False, False, True],   # (1, 2) is a wall too
                [False, False, False],
            ]
        )
        # Now (0, 2) is completely walled off from the rest of the graph.
        req = DeliveryPlanRequest(
            agent_id=0,
            current_loc=(0, 0),
            delivery_loc=(2, 2),
            parking_loc=(0, 2),
        )
        result = plan_paths_to_delivery(
            G_iso, [req], ReservationTable(), t_now=0
        )
        assert result is None


# ---------------------------------------------------------------------------
# Reservation-table conversion (helper-level)
# ---------------------------------------------------------------------------
class TestReservationConversion:
    def test_permanent_terminal_extends_reservations_through_horizon(self):
        # Build a table with one permanent terminal and run a quick plan;
        # the agent should never step onto the permanently-occupied cell.
        G = _build_open_grid(width=5, height=3)
        rt = ReservationTable()
        rt.reserve_path(
            agent_id=99,
            start_loc=(1, 2),
            path=[],
            start_t=0,
            is_permanent_terminal=True,
        )
        req = DeliveryPlanRequest(0, (1, 0), (1, 4), (0, 0))
        # Use a generous horizon so the entire plan window is constrained.
        result = plan_paths_to_delivery(G, [req], rt, t_now=0, horizon=50)
        assert result is not None
        r = result[0]
        # (1, 2) is permanently reserved -- the agent must never step on it.
        full_path = [r.sub_path[0]] + list(r.sub_path[1:]) + list(r.dummy_path)
        assert (1, 2) not in full_path


# ===========================================================================
# Phase C.2: PlanPathsToPickup (paper Algorithm 3 / Section 5.3)
# ===========================================================================
class TestPartitioning:
    def test_distinct_pickups_one_subgroup(self):
        reqs = [
            PickupPlanRequest(0, (0, 0), (2, 2), (0, 1), 0, 10),
            PickupPlanRequest(1, (1, 0), (2, 3), (0, 2), 0, 10),
            PickupPlanRequest(2, (1, 1), (2, 4), (0, 3), 0, 10),
        ]
        sgs = _partition_group2(reqs)
        assert len(sgs) == 1
        assert sorted(r.agent_id for r in sgs[0]) == [0, 1, 2]

    def test_shared_pickup_splits_across_subgroups(self):
        # Two agents share pickup (2, 2); one agent has pickup (2, 3).
        # Result: subgroup_1 = {0, 2}, subgroup_2 = {1}.
        reqs = [
            PickupPlanRequest(0, (0, 0), (2, 2), (0, 1), 0, 10),
            PickupPlanRequest(1, (1, 0), (2, 2), (0, 2), 0, 10),
            PickupPlanRequest(2, (1, 1), (2, 3), (0, 3), 0, 10),
        ]
        sgs = _partition_group2(reqs)
        assert len(sgs) == 2
        # Every subgroup must have pairwise-distinct pickups.
        for sg in sgs:
            pickups = [r.pickup_loc for r in sg]
            assert len(pickups) == len(set(pickups))
        # Total agents preserved across all subgroups.
        all_ids = sorted(r.agent_id for sg in sgs for r in sg)
        assert all_ids == [0, 1, 2]

    def test_three_way_shared_pickup(self):
        reqs = [
            PickupPlanRequest(0, (0, 0), (2, 2), (0, 1), 0, 10),
            PickupPlanRequest(1, (1, 0), (2, 2), (0, 2), 0, 10),
            PickupPlanRequest(2, (2, 0), (2, 2), (0, 3), 0, 10),
        ]
        sgs = _partition_group2(reqs)
        # Each agent must go into a separate subgroup since they all share pickup.
        assert len(sgs) == 3
        for sg in sgs:
            assert len(sg) == 1

    def test_empty_input(self):
        assert _partition_group2([]) == []


class TestEdgeCollisionResolution:
    def test_detect_simple_swap(self):
        w_a = AmapfWalker(0, 0, [(0, 0), (0, 1)])
        w_b = AmapfWalker(1, 1, [(0, 1), (0, 0)])
        k = _detect_edge_collision(w_a, w_b)
        assert k == 0

    def test_no_collision_returns_none(self):
        w_a = AmapfWalker(0, 0, [(0, 0), (0, 1), (0, 2)])
        w_b = AmapfWalker(1, 1, [(2, 0), (2, 1), (2, 2)])
        assert _detect_edge_collision(w_a, w_b) is None

    def test_swap_walkers_inverts_endpoints_and_task_ids(self):
        # Two walkers with a swap at step 0.
        w_a = AmapfWalker(0, 0, [(0, 0), (0, 1), (0, 2)])
        w_b = AmapfWalker(1, 1, [(0, 1), (0, 0), (0, -1)])
        new_a, new_b = _swap_walkers_at(w_a, w_b, k=0)
        # Wait step inserted at k+1.
        assert new_a.subpath[0] == (0, 0) and new_a.subpath[1] == (0, 0)
        assert new_b.subpath[0] == (0, 1) and new_b.subpath[1] == (0, 1)
        # After the wait, each walker follows the OTHER's tail.
        assert new_a.subpath[2:] == [(0, 0), (0, -1)]
        assert new_b.subpath[2:] == [(0, 1), (0, 2)]
        # Task IDs are swapped.
        assert new_a.assigned_task_agent_id == 1
        assert new_b.assigned_task_agent_id == 0
        # Original walker IDs preserved.
        assert new_a.start_agent_id == 0
        assert new_b.start_agent_id == 1

    def test_resolve_returns_collision_free_set(self):
        w_a = AmapfWalker(0, 0, [(0, 0), (0, 1)])
        w_b = AmapfWalker(1, 1, [(0, 1), (0, 0)])
        resolved = _resolve_edge_collisions([w_a, w_b])
        assert resolved is not None
        # No remaining edge collisions.
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                assert _detect_edge_collision(resolved[i], resolved[j]) is None


class TestPlanPathsToPickup:
    def test_empty_input(self):
        G = _build_open_grid(width=4, height=4)
        out = plan_paths_to_pickup(
            G, [], {}, t_now=0, initial_L=10
        )
        assert out == {}

    def test_single_agent(self):
        G = _build_open_grid(width=5, height=3)
        req = PickupPlanRequest(
            agent_id=0,
            current_loc=(1, 0),
            pickup_loc=(1, 4),
            parking_loc=(0, 0),
            release_time=0,
            initial_deadline=10,
        )
        out = plan_paths_to_pickup(G, [req], {}, t_now=0, initial_L=10)
        assert out is not None
        assert 0 in out
        r = out[0]
        assert r.sub_path[0] == (1, 0)
        assert r.sub_path[-1] == (1, 4)
        assert r.dummy_path[-1] == (0, 0)
        # No swap with only one agent.
        assert r.assigned_task_agent_id == 0

    def test_two_agents_distinct_pickups_one_subgroup(self):
        # Two agents heading to distinct pickups; parking locations are
        # NOT colinear with either pickup so dummy paths are non-empty
        # regardless of any anonymity swap.
        G = _build_open_grid(width=7, height=5)
        reqs = [
            PickupPlanRequest(
                agent_id=0,
                current_loc=(0, 0), pickup_loc=(2, 6),
                parking_loc=(4, 0), release_time=0, initial_deadline=20,
            ),
            PickupPlanRequest(
                agent_id=1,
                current_loc=(4, 6), pickup_loc=(2, 0),
                parking_loc=(0, 6), release_time=0, initial_deadline=20,
            ),
        ]
        out = plan_paths_to_pickup(G, reqs, {}, t_now=0, initial_L=20)
        assert out is not None
        assert set(out.keys()) == {0, 1}
        all_pickups = {x.pickup_loc for x in reqs}
        for ag_id in (0, 1):
            r = out[ag_id]
            req = next(x for x in reqs if x.agent_id == ag_id)
            assert r.sub_path[0] == req.current_loc
            # The agent ends its sub_path at SOME subgroup pickup (possibly
            # swapped via anonymity).
            assert r.sub_path[-1] in all_pickups
            # Dummy path always ends at the agent's OWN parking (parking
            # is never swapped by the anonymity logic).
            final = r.dummy_path[-1] if r.dummy_path else r.sub_path[-1]
            assert final == req.parking_loc

    def test_shared_pickup_partitions_into_subgroups(self):
        # Three agents all assigned to the SAME pickup -> 3 subgroups,
        # each of size 1. All three should still get planned.
        # (In practice this almost never happens, but the partitioner
        # has to handle it.)
        G = _build_open_grid(width=5, height=5)
        reqs = [
            PickupPlanRequest(0, (0, 0), (2, 2), (4, 0), 0, 15),
            PickupPlanRequest(1, (0, 4), (2, 2), (4, 4), 0, 15),
            PickupPlanRequest(2, (4, 2), (2, 2), (0, 2), 0, 15),
        ]
        out = plan_paths_to_pickup(G, reqs, {}, t_now=0, initial_L=15)
        assert out is not None
        assert set(out.keys()) == {0, 1, 2}
        for ag_id in (0, 1, 2):
            assert out[ag_id].sub_path[-1] == (2, 2)
            req = next(x for x in reqs if x.agent_id == ag_id)
            assert out[ag_id].dummy_path[-1] == req.parking_loc

    def test_external_path_avoidance(self):
        # An external agent is permanently parked at (1, 2). Our agent
        # must walk from (1, 0) to (1, 4) and must not step on (1, 2).
        G = _build_open_grid(width=5, height=3)
        external = {99: ((1, 2), [], 0, True)}
        req = PickupPlanRequest(0, (1, 0), (1, 4), (0, 0), 0, 15)
        out = plan_paths_to_pickup(
            G, [req], external, t_now=0, initial_L=15
        )
        assert out is not None
        full = [out[0].sub_path[0]] + list(out[0].sub_path[1:]) + list(out[0].dummy_path)
        assert (1, 2) not in full

    def test_L_increment_eventually_succeeds(self):
        # Manhattan = 4, but initial_L is too small. The loop should
        # bump L until it works.
        G = _build_open_grid(width=5, height=3)
        req = PickupPlanRequest(0, (1, 0), (1, 4), (0, 0), 0, 4)
        out = plan_paths_to_pickup(G, [req], {}, t_now=0, initial_L=4, max_L=30)
        assert out is not None
        assert out[0].sub_path[-1] == (1, 4)

    def test_path_stitching_invariants(self):
        # sub_path[-1] == dummy_path source (== walker's assigned pickup)
        # but dummy_path[0] is NOT the pickup cell (mla_star_single_goal
        # strips its start). So the joint future = sub_path[1:] + dummy_path
        # must be cell-adjacent throughout.
        G = _build_open_grid(width=6, height=4)
        req = PickupPlanRequest(0, (0, 0), (3, 5), (0, 5), 0, 20)
        out = plan_paths_to_pickup(G, [req], {}, t_now=0, initial_L=20)
        assert out is not None
        r = out[0]
        future = list(r.sub_path[1:]) + list(r.dummy_path)
        prev = req.current_loc
        for cell in future:
            assert abs(cell[0] - prev[0]) + abs(cell[1] - prev[1]) <= 1
            prev = cell
        assert future[-1] == req.parking_loc

    def test_assigned_task_id_records_swap(self):
        # Two agents on a single-row corridor, each assigned the OTHER's
        # pickup. The MCMF anonymity swap gives total cost 0 by leaving
        # each agent at its current cell -- which is the other's pickup.
        # The resulting result MUST record assigned_task_agent_id != agent_id
        # to signal the swap to the outer driver.
        G = _build_open_grid(width=5, height=1)
        reqs = [
            PickupPlanRequest(0, (0, 0), (0, 4), (0, 0), 0, 10),
            PickupPlanRequest(1, (0, 4), (0, 0), (0, 4), 0, 10),
        ]
        out = plan_paths_to_pickup(G, reqs, {}, t_now=0, initial_L=10)
        assert out is not None
        # Each agent's sub_path[-1] should be its OWN current_loc (they
        # didn't move), and each assigned_task_agent_id should be the
        # OTHER agent's id.
        for ag_id, req in zip([0, 1], reqs):
            r = out[ag_id]
            assert r.sub_path[-1] == req.current_loc
            assert r.assigned_task_agent_id != ag_id

    def test_infeasibility_returns_none(self):
        # Pickup is walled off; no path can reach it.
        G = _MockGraph([
            [False, True,  False],
            [True,  False, True],
            [False, True,  False],
        ])
        # (0, 0) is open but completely surrounded by walls.
        req = PickupPlanRequest(0, (0, 0), (2, 2), (0, 0), 0, 10)
        out = plan_paths_to_pickup(G, [req], {}, t_now=0, initial_L=10, max_L=10)
        assert out is None
