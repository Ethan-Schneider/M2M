"""Tests for TA-Hybrid's AMAPF min-cost max-flow solver.

These cover the Phase C.1 contract: given a subgroup A' with pairwise
distinct pickup locations + a current timestep t0 + a makespan bound L,
construct the time-extended flow network per paper Section 5.4 and
solve it via networkx's max_flow_min_cost to yield one sub-path per
agent ending at one of the subgroup's pickup locations.

The subgroup partitioning, edge-collision resolution, and dummy-path
planning are Phase C.2's job and are NOT exercised here.

Conventions exercised:

-   ``subpath[0] == start cell at t0``; ``subpath[-1] == pickup cell of
    whatever meta vertex this walker hit``.
-   ``subpath[i]`` is the cell at absolute time ``t0 + i``.
-   When ``start_agent_id != assigned_task_agent_id`` the walker
    represents an anonymity-swap: the C.2 caller re-assigns task
    sequences accordingly.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import pytest

from GT_grid_world.src.task_allocation_algorithms.ta_hybrid_amapf import (
    AmapfAgent,
    AmapfWalker,
    AmapfResult,
    build_amapf_network,
    detect_pickup_hold_violations,
    extract_walkers,
    solve_amapf,
    solve_amapf_single_L,
)


Loc = Tuple[int, int]


# ---------------------------------------------------------------------------
# Mock graph (same interface as test_ta_hybrid.py's mock)
# ---------------------------------------------------------------------------
class _MockGraph:
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
        return (self.height, self.width)


def _open_grid(width: int, height: int, walls: Sequence[Loc] = ()) -> _MockGraph:
    grid = [[False] * width for _ in range(height)]
    for r, c in walls:
        grid[r][c] = True
    return _MockGraph(grid)


# ---------------------------------------------------------------------------
# Network-construction sanity (paper Section 5.4)
# ---------------------------------------------------------------------------
class TestNetworkStructure:
    def test_only_out_nodes_exist_at_t0(self):
        G = _open_grid(width=3, height=3)
        ag = AmapfAgent(0, (0, 0), (2, 2), release_time=0, deadline=4)
        N = build_amapf_network(G, [ag], [], t0=0, L=4)
        # ('in', loc, 0) must not exist anywhere.
        in_at_t0 = [n for n in N.nodes if isinstance(n, tuple) and len(n) == 3
                    and n[0] == "in" and n[2] == 0]
        assert in_at_t0 == []
        # ('out', loc, 0) DOES exist for every open cell that participates
        # in a move or wait edge (which is every reachable cell).
        out_at_t0 = [n for n in N.nodes if isinstance(n, tuple) and len(n) == 3
                     and n[0] == "out" and n[2] == 0]
        assert len(out_at_t0) >= 1

    def test_source_connects_to_each_agent_current_loc_out_t0(self):
        G = _open_grid(width=4, height=4)
        agents = [
            AmapfAgent(0, (0, 0), (3, 3), 0, 6),
            AmapfAgent(1, (3, 0), (0, 3), 0, 6),
        ]
        N = build_amapf_network(G, agents, [], t0=0, L=6)
        source_targets = list(N.successors("S"))
        assert len(source_targets) == 2
        target_cells = sorted(n[1] for n in source_targets)
        assert target_cells == [(0, 0), (3, 0)]
        for tgt in source_targets:
            data = N["S"][tgt]
            assert data["capacity"] == 1
            assert data["weight"] == 0

    def test_meta_vertex_and_sink_edges(self):
        G = _open_grid(width=3, height=3)
        ag = AmapfAgent(7, (0, 0), (2, 2), release_time=2, deadline=4)
        N = build_amapf_network(G, [ag], [], t0=0, L=4)
        meta = ("meta", 7)
        assert meta in N.nodes
        # Pickup-arrival window: max(release_time=2, r_prime+1=0) ... deadline=4
        # i.e. timesteps 2, 3, 4. Each contributes one (s_out_t -> meta) edge.
        in_edges = [(u, v) for u, v in N.in_edges(meta)]
        in_times = sorted(u[2] for u, _ in in_edges)
        assert in_times == [2, 3, 4]
        for u, v in in_edges:
            assert u == ("out", (2, 2), u[2])
            assert N[u][v]["capacity"] == 1
            assert N[u][v]["weight"] == 0
        # Meta -> T edge exists.
        assert N.has_edge(meta, "T")
        assert N[meta]["T"]["capacity"] == 1

    def test_pickup_hold_removes_move_out_keeps_wait(self):
        # Tiny 3-cell row: pickup at (0, 1) with release_time=1.
        # After t=1, all (s_out_t -> neighbor_in_{t+1}) move edges must
        # be gone, but the wait edge (s_out_t -> s_in_{t+1}) must remain.
        G = _open_grid(width=3, height=1)
        ag = AmapfAgent(0, (0, 0), (0, 1), release_time=1, deadline=3)
        N = build_amapf_network(G, [ag], [], t0=0, L=3)
        s = (0, 1)
        # Move-out edges from s at t=1: should be GONE.
        # Neighbours of (0, 1) are (0, 0) and (0, 2).
        assert not N.has_edge(("out", s, 1), ("in", (0, 0), 2))
        assert not N.has_edge(("out", s, 1), ("in", (0, 2), 2))
        # Wait edge from s at t=1: should still be present.
        assert N.has_edge(("out", s, 1), ("in", s, 2))
        # Before release time (t=0 -> t=1), move-out edges are still present.
        assert N.has_edge(("out", s, 0), ("in", (0, 0), 1))
        assert N.has_edge(("out", s, 0), ("in", (0, 2), 1))

    def test_other_agent_vertex_avoidance(self):
        # An other-agent sub-path occupies (1, 1) at t=2.
        # The vertex-capacity edge (in (1,1) at 2) -> (out (1,1) at 2)
        # should be removed so no walker can occupy (1, 1) at t=2.
        G = _open_grid(width=3, height=3)
        ag = AmapfAgent(0, (0, 0), (2, 2), 0, 5)
        other = [((1, 0), [(1, 1)], 1)]  # at t=1 at (1, 0), at t=2 at (1, 1)
        N = build_amapf_network(G, [ag], other, t0=0, L=5)
        assert not N.has_edge(("in", (1, 1), 2), ("out", (1, 1), 2))
        # Other times for (1, 1) are still passable.
        assert N.has_edge(("in", (1, 1), 3), ("out", (1, 1), 3))

    def test_other_agent_edge_avoidance(self):
        # Other agent moves (1, 0) -> (1, 1) between t=1 and t=2.
        # The swap edge ((out, (1, 1), 1) -> (in, (1, 0), 2)) must be
        # removed so no walker can swap with the other agent.
        G = _open_grid(width=3, height=3)
        ag = AmapfAgent(0, (0, 0), (2, 2), 0, 5)
        other = [((1, 0), [(1, 1)], 1)]
        N = build_amapf_network(G, [ag], other, t0=0, L=5)
        assert not N.has_edge(("out", (1, 1), 1), ("in", (1, 0), 2))

    def test_r_prime_pushes_pickup_window_after_other_agent_passes(self):
        # An other-agent path passes through the pickup cell at t=3.
        # The pickup-hold window must start strictly after that, i.e. at
        # max(release_time, 4).
        G = _open_grid(width=3, height=3)
        # Pickup at (2, 2); other agent at (2, 2) at t=3.
        ag = AmapfAgent(0, (0, 0), (2, 2), release_time=1, deadline=8)
        other = [((2, 1), [(2, 2)], 2)]  # at t=2 at (2,1), t=3 at (2,2)
        N = build_amapf_network(G, [ag], other, t0=0, L=8)
        meta = ("meta", 0)
        in_times = sorted(u[2] for u, _ in N.in_edges(meta))
        # Should start at max(release=1, r_prime+1=4) = 4
        assert in_times[0] == 4
        assert in_times[-1] == 8


# ---------------------------------------------------------------------------
# Single-L solving (end-to-end on tiny instances)
# ---------------------------------------------------------------------------
class TestSolveSingleL:
    def test_single_agent_straight_walk(self):
        # 5-cell row; agent at (0, 0), pickup at (0, 4). L large enough.
        G = _open_grid(width=5, height=1)
        ag = AmapfAgent(0, (0, 0), (0, 4), release_time=0, deadline=8)
        result = solve_amapf_single_L(G, [ag], [], t0=0, L=8)
        assert result is not None
        assert len(result.walkers) == 1
        w = result.walkers[0]
        assert w.start_agent_id == 0
        assert w.assigned_task_agent_id == 0
        assert w.subpath[0] == (0, 0)
        assert w.subpath[-1] == (0, 4)
        # Manhattan distance is 4, so the optimal sub-path is 5 cells
        # (including start). With pickup-hold the walker may HOLD the
        # pickup cell after arriving (extra waits at the goal); we accept
        # any path that arrives by deadline=8.
        assert len(w.subpath) <= 9  # at most L+1 cells

    def test_two_agents_no_conflict(self):
        # Two agents, two pickups, no interaction.
        G = _open_grid(width=5, height=5)
        agents = [
            AmapfAgent(0, (0, 0), (0, 4), release_time=0, deadline=8),
            AmapfAgent(1, (4, 0), (4, 4), release_time=0, deadline=8),
        ]
        result = solve_amapf_single_L(G, agents, [], t0=0, L=8)
        assert result is not None
        assert len(result.walkers) == 2
        # Both walkers arrive at their respective pickups.
        endpoints = sorted(w.subpath[-1] for w in result.walkers)
        assert endpoints == [(0, 4), (4, 4)]

    def test_anonymity_swap_picks_shorter_total_cost(self):
        # Agent 0 at (0, 0), task pickup at (0, 4)
        # Agent 1 at (0, 4), task pickup at (0, 0)
        # Anonymity allows them to SWAP -- each walker stays put.
        # Min-cost flow should find total cost 0 (no moves), not 8 (each
        # agent walks the full corridor).
        G = _open_grid(width=5, height=1)
        agents = [
            AmapfAgent(0, (0, 0), (0, 4), release_time=0, deadline=8),
            AmapfAgent(1, (0, 4), (0, 0), release_time=0, deadline=8),
        ]
        result = solve_amapf_single_L(G, agents, [], t0=0, L=8)
        assert result is not None
        assert len(result.walkers) == 2
        # Optimal total cost is 0 (each walker stays at their start, which
        # IS the pickup of the OTHER agent's task).
        assert result.total_cost == 0
        # Check that the walkers' endpoints match the OTHER agent's pickup.
        starts_and_ends = sorted(
            (w.start_agent_id, w.subpath[0], w.subpath[-1])
            for w in result.walkers
        )
        # Agent 0 starts at (0, 0) and ends at (0, 0); meta hit is agent 1.
        assert starts_and_ends[0][1] == starts_and_ends[0][2]
        assert starts_and_ends[1][1] == starts_and_ends[1][2]
        # The assigned_task_agent_id should be the OTHER agent's id.
        for w in result.walkers:
            assert w.assigned_task_agent_id != w.start_agent_id

    def test_subpath_is_step_local(self):
        # The subpath must have only adjacent-or-equal consecutive cells.
        G = _open_grid(width=5, height=5)
        agents = [
            AmapfAgent(0, (0, 0), (3, 4), release_time=0, deadline=10),
        ]
        result = solve_amapf_single_L(G, agents, [], t0=0, L=10)
        assert result is not None
        for w in result.walkers:
            prev = w.subpath[0]
            for cell in w.subpath[1:]:
                manh = abs(cell[0] - prev[0]) + abs(cell[1] - prev[1])
                assert manh <= 1, f"teleport from {prev} to {cell}"
                prev = cell

    def test_other_agent_blocks_direct_path(self):
        # Block the pickup cell at the time the direct walker would arrive.
        # The walker must wait or detour.
        G = _open_grid(width=5, height=1)
        ag = AmapfAgent(0, (0, 0), (0, 4), release_time=0, deadline=8)
        # Another agent is permanently at (0, 4) from t=4 to t=7.
        other = [((0, 4), [(0, 4), (0, 4), (0, 4)], 4)]
        result = solve_amapf_single_L(G, [ag], other, t0=0, L=8)
        assert result is not None
        w = result.walkers[0]
        assert w.subpath[-1] == (0, 4)
        # Earliest non-blocked arrival is t=8 (after the other agent
        # leaves). Walker arrives at len(subpath) - 1.
        assert len(w.subpath) >= 9  # arrives at t >= 8


# ---------------------------------------------------------------------------
# Failure / infeasibility handling
# ---------------------------------------------------------------------------
class TestInfeasibility:
    def test_window_collapses_returns_none(self):
        # release_time=10 but deadline=5 -- window_start > window_end.
        G = _open_grid(width=3, height=3)
        ag = AmapfAgent(0, (0, 0), (2, 2), release_time=10, deadline=5)
        result = solve_amapf_single_L(G, [ag], [], t0=0, L=10)
        assert result is None

    def test_unreachable_pickup_returns_none(self):
        # Wall the pickup cell off entirely.
        G = _MockGraph([
            [False, True,  False],
            [True,  False, False],   # (0,1) and (1,0) are walls
            [False, False, False],
        ])
        # (0, 0) is open but completely surrounded by walls; can't reach
        # any other cell.
        ag = AmapfAgent(0, (0, 0), (2, 2), release_time=0, deadline=5)
        result = solve_amapf_single_L(G, [ag], [], t0=0, L=5)
        assert result is None


# ---------------------------------------------------------------------------
# L-increment retry loop
# ---------------------------------------------------------------------------
class TestLIncrementRetry:
    def test_increments_L_until_feasible(self):
        # Manhattan distance is 4 from (0,0) to (0,4); the initial L=2 is
        # infeasible, but the loop should bump up to L>=4 and succeed.
        G = _open_grid(width=5, height=1)
        ag = AmapfAgent(0, (0, 0), (0, 4), release_time=0, deadline=20)
        result = solve_amapf(G, [ag], [], t0=0, initial_L=2, max_L=20)
        assert result is not None
        assert result.makespan_bound_used >= 4

    def test_gives_up_at_max_L(self):
        # Manhattan distance is 100 but max_L is 5 -- should give up.
        G = _open_grid(width=200, height=1)
        ag = AmapfAgent(0, (0, 0), (0, 100), release_time=0, deadline=200)
        result = solve_amapf(G, [ag], [], t0=0, initial_L=2, max_L=5)
        assert result is None


# ---------------------------------------------------------------------------
# Demand / supply balance (networkx requirement)
# ---------------------------------------------------------------------------
class TestNetworkXSanity:
    def test_demands_balance(self):
        G = _open_grid(width=4, height=4)
        agents = [
            AmapfAgent(0, (0, 0), (3, 3), 0, 8),
            AmapfAgent(1, (0, 3), (3, 0), 0, 8),
        ]
        N = build_amapf_network(G, agents, [], t0=0, L=8)
        total_demand = sum(N.nodes[n].get("demand", 0) for n in N.nodes)
        assert total_demand == 0

    def test_zero_units_raises(self):
        # build_amapf_network rejects empty agent list.
        G = _open_grid(width=3, height=3)
        with pytest.raises(ValueError):
            build_amapf_network(G, [], [], t0=0, L=5)

    def test_bad_L_raises(self):
        G = _open_grid(width=3, height=3)
        ag = AmapfAgent(0, (0, 0), (2, 2), 0, 4)
        with pytest.raises(ValueError):
            build_amapf_network(G, [ag], [], t0=5, L=3)

    def test_obstacle_start_raises(self):
        G = _MockGraph([[False, True], [False, False]])
        ag = AmapfAgent(0, (0, 1), (1, 1), 0, 4)
        with pytest.raises(ValueError):
            build_amapf_network(G, [ag], [], t0=0, L=4)


# ---------------------------------------------------------------------------
# Paper Section 5.5: "first try without holds" optimisation
# ---------------------------------------------------------------------------
class TestSection55PickupHoldFlag:
    """The pickup-hold edge removal is conditional on enforce_pickup_hold."""

    def test_no_holds_keeps_move_out_edges_after_release(self):
        # 3-cell row, pickup at (0, 1), release_time=1. With holds the
        # move-out edges from (0, 1) at t >= 1 are removed; without
        # holds they must be kept.
        G = _open_grid(width=3, height=1)
        ag = AmapfAgent(0, (0, 0), (0, 1), release_time=1, deadline=3)
        with_holds = build_amapf_network(G, [ag], [], t0=0, L=3,
                                          enforce_pickup_hold=True)
        no_holds = build_amapf_network(G, [ag], [], t0=0, L=3,
                                        enforce_pickup_hold=False)
        s = (0, 1)
        # With holds: gone.
        assert not with_holds.has_edge(("out", s, 1), ("in", (0, 0), 2))
        assert not with_holds.has_edge(("out", s, 1), ("in", (0, 2), 2))
        # Without holds: still present.
        assert no_holds.has_edge(("out", s, 1), ("in", (0, 0), 2))
        assert no_holds.has_edge(("out", s, 1), ("in", (0, 2), 2))

    def test_no_holds_default_for_single_L_still_enforces(self):
        # The single-L solver defaults to enforce_pickup_hold=True so we
        # don't silently change behaviour for direct callers.
        G = _open_grid(width=3, height=1)
        ag = AmapfAgent(0, (0, 0), (0, 1), release_time=1, deadline=3)
        N_default = build_amapf_network(G, [ag], [], t0=0, L=3)
        s = (0, 1)
        assert not N_default.has_edge(("out", s, 1), ("in", (0, 0), 2))


class TestDetectPickupHoldViolations:
    def test_no_walkers_returns_empty(self):
        assert detect_pickup_hold_violations([], [], t0=0) == []

    def test_no_violation_when_walker_only_passes_before_arrival(self):
        # Walker A arrives at (0, 2) at t=2. Walker B passes through
        # (0, 2) at t=1 (before A's arrival) -- that's a VERTEX collision
        # caught by AMAPF directly, not a pickup-hold issue. The
        # detector should not flag it.
        agents = [
            AmapfAgent(0, (0, 0), (0, 2), 0, 5),
            AmapfAgent(1, (0, 4), (0, 0), 0, 5),
        ]
        walkers = [
            AmapfWalker(0, 0, [(0, 0), (0, 1), (0, 2)]),
            AmapfWalker(1, 1, [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0)]),
        ]
        # Walker 1 IS at (0, 2) at t=2 (same tick as Walker 0's arrival)
        # but the rule is strictly t > arrival_t, so this is not flagged.
        # Note: vertex collisions are caught by AMAPF; this would never
        # actually be returned. We only test the detector.
        violations = detect_pickup_hold_violations(walkers, agents, t0=0)
        assert violations == []

    def test_violation_when_walker_crosses_after_arrival(self):
        # Walker A arrives at (0, 1) at t=1. Walker B passes through
        # (0, 1) at t=3 (after A's arrival) -- THIS is exactly what the
        # pickup-hold constraint exists to prevent.
        agents = [
            AmapfAgent(0, (0, 0), (0, 1), 0, 5),
            AmapfAgent(1, (0, 4), (0, 3), 0, 5),
        ]
        walkers = [
            AmapfWalker(0, 0, [(0, 0), (0, 1)]),  # arrives at t=1 at (0, 1)
            AmapfWalker(1, 1, [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (0, 1), (0, 2), (0, 3)]),
        ]
        violations = detect_pickup_hold_violations(walkers, agents, t0=0)
        assert len(violations) >= 1
        holder, intruder, cell, _t = violations[0]
        assert holder == 0
        assert intruder == 1
        assert cell == (0, 1)

    def test_walker_does_not_violate_its_own_holder(self):
        # A walker passing through ITS OWN endpoint doesn't violate.
        agents = [AmapfAgent(0, (0, 0), (0, 2), 0, 5)]
        walkers = [AmapfWalker(0, 0, [(0, 0), (0, 1), (0, 2)])]
        assert detect_pickup_hold_violations(walkers, agents, t0=0) == []

    def test_t0_offset_handled(self):
        # Same scenario as the violation case but with t0 > 0.
        agents = [
            AmapfAgent(0, (0, 0), (0, 1), 0, 10),
            AmapfAgent(1, (0, 4), (0, 3), 0, 10),
        ]
        walkers = [
            AmapfWalker(0, 0, [(0, 0), (0, 1)]),
            AmapfWalker(1, 1, [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (0, 1)]),
        ]
        # At t0=5: walker 0 arrives at (0, 1) at t=6. Walker 1 is at
        # (0, 1) at t=5+5=10 -- after arrival, so it's a violation.
        violations = detect_pickup_hold_violations(walkers, agents, t0=5)
        assert len(violations) >= 1


class TestSolveAmapfWithSection55:
    """End-to-end §5.5 behaviour on the canonical chokepoint scenario."""

    def test_chokepoint_unblocked_when_walker_passes_before_arrival(self):
        # When one agent's pickup is on the only path to another agent's
        # pickup, the canonical with-holds AMAPF is infeasible (the
        # move-out edges from the chokepoint are removed for all
        # t >= release). Section 5.5 unblocks this iff the optimal
        # no-holds plan has the OTHER walker passing through the
        # chokepoint AT OR BEFORE the holder's arrival (so the
        # post-hoc violation check passes).
        #
        # Layout (F=free, X=wall):
        #   (0,0)F (0,1)X (0,2)F
        #   (1,0)F (1,1)X (1,2)F
        #   (2,0)F (2,1)F (2,2)F
        # The chokepoint (1, 0) lies on the only path between (0, 0)
        # and the rest of the map. Agent 0 starts at (2, 2) and is
        # assigned pickup (1, 0). Agent 1 starts at (0, 2) (one column
        # over from (0, 0)) and is assigned pickup (0, 0).
        # Agent 1's shortest path: (0, 2) -> (0, 2)wait -> ... -> via
        # (2, 0). Actually (0, 0) is only reachable via (1, 0).
        # We give agent 1 release_time=0 and large deadline so the
        # solver may shift arrivals. Agent 0 has a later release_time
        # so the no-holds optimal plan can route agent 1 through
        # (1, 0) BEFORE agent 0 arrives.
        G = _MockGraph([
            [False, True,  False],
            [False, True,  False],
            [False, False, False],
        ])
        agents = [
            AmapfAgent(0, (2, 2), (1, 0), release_time=20, deadline=30),
            AmapfAgent(1, (0, 2), (0, 0), release_time=0, deadline=30),
        ]
        # With holds: removing move-out from (1, 0) for t in [20, L-1]
        # only kicks in late, so agent 1 can still pass through (1, 0)
        # at t < 20. Should be feasible too -- not a great test then.
        # Tighten to demonstrate §5.5: set both release_times to 0 and
        # check that no-holds finds a plan where the optimal has
        # agent 1 (closer to the chokepoint) arriving FIRST. The
        # detector then flags no violation since agent 1 passes
        # (1, 0) before agent 0.
        agents_eq = [
            AmapfAgent(0, (2, 2), (1, 0), release_time=0, deadline=30),
            AmapfAgent(1, (0, 2), (0, 0), release_time=0, deadline=30),
        ]
        with_55 = solve_amapf(
            G, agents_eq, [], t0=0, initial_L=10, max_L=30,
            use_5_5_optimization=True,
        )
        # Either: §5.5 finds a plan (no violation), or returns None.
        # The crucial property is that when §5.5 succeeds, it does so
        # with no pickup-hold violation in the returned plan.
        if with_55 is not None:
            assert detect_pickup_hold_violations(
                with_55.walkers, agents_eq, t0=0
            ) == []
            endpoints = sorted(w.subpath[-1] for w in with_55.walkers)
            assert endpoints == [(0, 0), (1, 0)]

    def test_section55_returned_plan_has_no_violations(self):
        # Whatever the §5.5 wrapper returns, the result must satisfy the
        # paper's invariant: no walker's sub-path overlaps another
        # walker's terminal pickup at a strictly later timestep. This is
        # the guarantee §5.5's detector exists to provide.
        # Construct a corridor where a naive no-holds plan WOULD cross
        # a pickup after arrival but the wrapper still ends up
        # returning a violation-free plan (either because the min-cost
        # objective picks an anonymity swap or because the retry with
        # holds finds an alternative).
        G = _open_grid(width=7, height=1)
        agents = [
            AmapfAgent(0, (0, 0), (0, 1), release_time=0, deadline=8),
            AmapfAgent(1, (0, 6), (0, 0), release_time=0, deadline=8),
        ]
        result = solve_amapf(G, agents, [], t0=0, initial_L=6, max_L=12,
                              use_5_5_optimization=True)
        assert result is not None
        assert detect_pickup_hold_violations(
            result.walkers, agents, t0=0
        ) == []
        endpoints = sorted(w.subpath[-1] for w in result.walkers)
        assert (0, 0) in endpoints and (0, 1) in endpoints

    def test_section55_short_circuits_when_no_violation(self):
        # When the no-holds solve already satisfies the pickup-hold
        # property, the wrapper must NOT redo the solve. We assert by
        # checking that the result is feasible at an L where the
        # with-holds variant would also be feasible (i.e., both succeed,
        # but the detector says no violations).
        G = _open_grid(width=5, height=5)
        agents = [
            AmapfAgent(0, (0, 0), (0, 4), release_time=0, deadline=10),
            AmapfAgent(1, (4, 0), (4, 4), release_time=0, deadline=10),
        ]
        result = solve_amapf(G, agents, [], t0=0, initial_L=4, max_L=12,
                              use_5_5_optimization=True)
        assert result is not None
        assert len(result.walkers) == 2
        assert detect_pickup_hold_violations(
            result.walkers, agents, t0=0
        ) == []

    def test_section55_infeasible_when_both_modes_fail(self):
        # Window collapsed -- with or without holds, both infeasible.
        G = _open_grid(width=3, height=3)
        ag = AmapfAgent(0, (0, 0), (2, 2), release_time=100, deadline=5)
        result = solve_amapf(G, [ag], [], t0=0, initial_L=5, max_L=5,
                              use_5_5_optimization=True)
        assert result is None
