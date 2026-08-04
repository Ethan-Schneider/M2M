"""Tests for the ICBS wrapper (``path_finding_algorithms/icbs_planner.py``).

We use a minimal ``_MockGraph`` for unit tests so each test can declare its
exact obstacle layout. The wrapper depends on the ``Graph`` interface only
via ``height``, ``width``, and ``get_if_obstacle((r, c)) -> bool``, so a
two-method stub is sufficient.

The ``TestSubmoduleSmoke`` class invokes the real submodule end-to-end on
small instances to verify the gloriyo/MAPF-ICBS submodule is functional
in our environment. These tests are the first line of defence if the
submodule moves to a new upstream commit.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pytest

from GT_grid_world.src.path_finding_algorithms.icbs_planner import (
    graph_to_my_map,
    path_to_edge_reservations,
    path_to_vertex_reservations,
    plan_icbs_paths,
    reservations_to_constraints,
)


Loc = Tuple[int, int]


# ---------------------------------------------------------------------------
# Mock Graph
# ---------------------------------------------------------------------------
class _MockGraph:
    """Tiny stand-in for the parts of ``Graph`` the wrapper actually uses."""

    def __init__(self, my_map: List[List[bool]]) -> None:
        self._map = my_map
        self.height = len(my_map)
        self.width = len(my_map[0]) if my_map else 0

    def get_if_obstacle(self, node: Tuple[int, int]) -> bool:
        r, c = node
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True
        return self._map[r][c]


def _build_grid(width: int, height: int, obstacles: Sequence[Loc] = ()) -> _MockGraph:
    """Build a ``width`` x ``height`` open grid with the listed obstacle cells.

    Note: signature is (width, height) to match the conventional ``(W, H)``
    grid-paper ordering even though our coordinate tuples are (row, col).
    Cells outside ``obstacles`` are open (``False``).
    """
    grid = [[False] * width for _ in range(height)]
    for r, c in obstacles:
        grid[r][c] = True
    return _MockGraph(grid)


def _collisions_in(paths: List[List[Loc]]) -> List[Tuple[int, int, int]]:
    """Return a list of ``(agent_i, agent_j, t)`` collisions across all paths.

    Includes both vertex collisions (same cell at same time) and edge
    collisions (swap). Paths shorter than the longest path are treated as
    "agent waiting at last cell" for trailing timesteps.
    """
    if not paths:
        return []
    T = max(len(p) for p in paths)

    def loc_at(path: List[Loc], t: int) -> Loc:
        return path[t] if t < len(path) else path[-1]

    found = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            for t in range(T):
                if loc_at(paths[i], t) == loc_at(paths[j], t):
                    found.append((i, j, t))
                    continue
                if t > 0 and (
                    loc_at(paths[i], t - 1) == loc_at(paths[j], t)
                    and loc_at(paths[j], t - 1) == loc_at(paths[i], t)
                ):
                    found.append((i, j, t))
    return found


# ---------------------------------------------------------------------------
# graph_to_my_map
# ---------------------------------------------------------------------------
class TestGraphToMyMap:
    def test_dimensions_match_graph(self):
        G = _build_grid(width=5, height=3)
        m = graph_to_my_map(G)
        assert len(m) == 3
        assert len(m[0]) == 5

    def test_open_cells_are_false(self):
        G = _build_grid(width=4, height=2)
        m = graph_to_my_map(G)
        assert all(cell is False for row in m for cell in row)

    def test_obstacle_cells_are_true(self):
        G = _build_grid(width=4, height=3, obstacles=[(0, 0), (1, 2), (2, 3)])
        m = graph_to_my_map(G)
        assert m[0][0] is True
        assert m[1][2] is True
        assert m[2][3] is True
        # Sanity: an open cell is still False
        assert m[0][1] is False


# ---------------------------------------------------------------------------
# path_to_vertex_reservations / path_to_edge_reservations
# ---------------------------------------------------------------------------
class TestPathToVertexReservations:
    def test_empty_path_returns_empty(self):
        assert path_to_vertex_reservations([]) == []

    def test_each_cell_gets_a_reservation_with_increasing_time(self):
        out = path_to_vertex_reservations([(0, 0), (0, 1), (0, 2)], start_time=5)
        assert out == [((0, 0), 5), ((0, 1), 6), ((0, 2), 7)]

    def test_extend_to_reserves_terminal_cell_until_given_time(self):
        out = path_to_vertex_reservations(
            [(0, 0), (0, 1)], start_time=0, extend_to=4
        )
        # path occupies (0,0)@0 and (0,1)@1; tail extends (0,1)@2,3,4
        assert out == [
            ((0, 0), 0),
            ((0, 1), 1),
            ((0, 1), 2),
            ((0, 1), 3),
            ((0, 1), 4),
        ]

    def test_extend_to_no_op_when_at_or_before_path_end(self):
        out = path_to_vertex_reservations(
            [(0, 0), (0, 1)], start_time=0, extend_to=1
        )
        assert out == [((0, 0), 0), ((0, 1), 1)]


class TestPathToEdgeReservations:
    def test_empty_path_returns_empty(self):
        assert path_to_edge_reservations([]) == []

    def test_single_cell_path_returns_empty(self):
        assert path_to_edge_reservations([(0, 0)]) == []

    def test_consecutive_moves_become_edge_reservations(self):
        out = path_to_edge_reservations(
            [(0, 0), (0, 1), (1, 1)], start_time=10
        )
        # arrivals at t=11 (into (0,1)) and t=12 (into (1,1))
        assert out == [((0, 0), (0, 1), 11), ((0, 1), (1, 1), 12)]

    def test_wait_steps_are_skipped(self):
        # An agent waiting at (0,0) for one tick should not generate a self-edge reservation.
        out = path_to_edge_reservations([(0, 0), (0, 0), (0, 1)], start_time=0)
        assert out == [((0, 0), (0, 1), 2)]


# ---------------------------------------------------------------------------
# reservations_to_constraints
# ---------------------------------------------------------------------------
class TestReservationsToConstraints:
    def test_empty_returns_empty(self):
        assert reservations_to_constraints([], [], num_agents=4) == []

    def test_vertex_reservation_is_one_per_agent(self):
        cs = reservations_to_constraints([((3, 4), 5)], [], num_agents=3)
        assert len(cs) == 3
        assert {c["agent"] for c in cs} == {0, 1, 2}
        for c in cs:
            assert c["loc"] == [(3, 4)]
            assert c["timestep"] == 5
            assert c["positive"] is False
            assert c["meta_agent"] == {c["agent"]}

    def test_edge_reservation_format(self):
        cs = reservations_to_constraints([], [((1, 1), (1, 2), 3)], num_agents=2)
        assert len(cs) == 2
        for c in cs:
            assert c["loc"] == [(1, 1), (1, 2)]
            assert c["timestep"] == 3
            assert c["positive"] is False
            assert c["meta_agent"] == {c["agent"]}

    def test_mixed_vertex_and_edge_are_both_present(self):
        cs = reservations_to_constraints(
            [((0, 0), 0)],
            [((0, 0), (0, 1), 1)],
            num_agents=1,
        )
        assert len(cs) == 2
        kinds = [len(c["loc"]) for c in cs]
        assert sorted(kinds) == [1, 2]

    def test_zero_agents_produces_no_constraints(self):
        cs = reservations_to_constraints([((0, 0), 0)], [], num_agents=0)
        assert cs == []

    def test_locations_are_tuple_coerced(self):
        # Caller may pass lists; we want tuples in the output for hashability.
        cs = reservations_to_constraints([([3, 4], 1)], [], num_agents=1)
        assert cs[0]["loc"] == [(3, 4)]
        assert isinstance(cs[0]["loc"][0], tuple)


# ---------------------------------------------------------------------------
# plan_icbs_paths -- argument validation
# ---------------------------------------------------------------------------
class TestPlanIcbsPathsValidation:
    def test_mismatched_starts_and_goals_raises(self):
        G = _build_grid(width=3, height=3)
        with pytest.raises(ValueError, match="len"):
            plan_icbs_paths(G, starts=[(0, 0)], goals=[(0, 1), (0, 2)])


# ---------------------------------------------------------------------------
# plan_icbs_paths -- single-agent sanity
# ---------------------------------------------------------------------------
class TestPlanIcbsPathsSingleAgent:
    def test_returns_straight_line_in_open_corridor(self):
        # Bordered 5x3 corridor: row 0 and row 2 are walls (gloriyo's A* needs
        # in-bounds neighbours to terminate cleanly when goal == start row).
        G = _MockGraph(
            [
                [True] * 5,
                [False] * 5,
                [True] * 5,
            ]
        )
        paths = plan_icbs_paths(G, starts=[(1, 0)], goals=[(1, 4)])
        assert paths is not None
        assert paths[0][0] == (1, 0)
        assert paths[0][-1] == (1, 4)
        # 4 moves -> 5 cells
        assert len(paths[0]) == 5

    def test_returns_path_format_is_list_of_tuple_coords(self):
        G = _MockGraph(
            [
                [True, True, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, True, True],
            ]
        )
        paths = plan_icbs_paths(G, starts=[(1, 1)], goals=[(3, 1)])
        assert paths is not None
        for step in paths[0]:
            assert isinstance(step, tuple) and len(step) == 2
            assert all(isinstance(c, int) for c in step)


# ---------------------------------------------------------------------------
# plan_icbs_paths -- multi-agent collision avoidance
# ---------------------------------------------------------------------------
class TestPlanIcbsPathsMultiAgent:
    def test_two_agents_in_corridor_are_collision_free(self):
        # 7x3 bordered corridor. Two agents swap (1,1) <-> (1,5).
        # Note: a one-row corridor without side detours would be unsolvable;
        # we add a 1-cell pocket at row 0 column 3 so one agent can step aside.
        G = _MockGraph(
            [
                [True, True, True, False, True, True, True],
                [True, False, False, False, False, False, True],
                [True, True, True, True, True, True, True],
            ]
        )
        paths = plan_icbs_paths(
            G,
            starts=[(1, 1), (1, 5)],
            goals=[(1, 5), (1, 1)],
        )
        assert paths is not None
        assert paths[0][0] == (1, 1) and paths[0][-1] == (1, 5)
        assert paths[1][0] == (1, 5) and paths[1][-1] == (1, 1)
        assert _collisions_in(paths) == []

    def test_three_agents_in_open_room_are_collision_free(self):
        # 5x5 open room. Three agents pick disjoint goals.
        grid = [[False] * 5 for _ in range(5)]
        # Wall off the boundary so A*'s 5-neighbour move set stays in-bounds.
        for r in range(5):
            grid[r][0] = grid[r][4] = True
        for c in range(5):
            grid[0][c] = grid[4][c] = True
        G = _MockGraph(grid)
        paths = plan_icbs_paths(
            G,
            starts=[(1, 1), (1, 3), (3, 1)],
            goals=[(3, 3), (3, 1), (1, 3)],
        )
        assert paths is not None
        assert len(paths) == 3
        assert _collisions_in(paths) == []


# ---------------------------------------------------------------------------
# plan_icbs_paths -- external reservations
# ---------------------------------------------------------------------------
class TestPlanIcbsPathsWithReservations:
    def test_vertex_reservation_forces_a_detour(self):
        # 3x3 open ring around the centre cell. Agent goes (1,0) -> (1,2).
        # Reserve (1,1) for every timestep that would let the agent pass
        # straight through; the agent must detour through (0,1) or (2,1).
        grid = [[False] * 3 for _ in range(3)]
        G = _MockGraph(grid)
        # The direct path takes 2 moves: (1,0) -> (1,1) at t=1 -> (1,2) at t=2.
        # Forbid (1,1) at t=1 -> the agent must detour, costing more.
        paths_baseline = plan_icbs_paths(G, starts=[(1, 0)], goals=[(1, 2)])
        paths_detour = plan_icbs_paths(
            G,
            starts=[(1, 0)],
            goals=[(1, 2)],
            vertex_reservations=[((1, 1), 1)],
        )
        assert paths_baseline is not None and paths_detour is not None
        # Detour has a strictly longer path (the agent can't go straight through (1,1) at t=1).
        assert len(paths_detour[0]) > len(paths_baseline[0])
        # And the detour respects the reservation.
        assert paths_detour[0][1] != (1, 1)

    def test_vertex_reservation_is_respected_across_multiple_agents(self):
        # Two agents that both want to pass through (1,1). Reserve (1,1) at t=1.
        # Both agents have to detour.
        grid = [[False] * 4 for _ in range(3)]
        G = _MockGraph(grid)
        paths = plan_icbs_paths(
            G,
            starts=[(1, 0), (1, 3)],
            goals=[(1, 3), (1, 0)],
            vertex_reservations=[((1, 1), 1)],
        )
        assert paths is not None
        # Neither agent is at (1,1) at t=1.
        for path in paths:
            if len(path) > 1:
                assert path[1] != (1, 1)
        assert _collisions_in(paths) == []


# ---------------------------------------------------------------------------
# stdout silencing
# ---------------------------------------------------------------------------
class TestStdoutSilencing:
    def test_silence_stdout_swallows_submodule_chatter(self, capsys):
        grid = [[False] * 3 for _ in range(3)]
        G = _MockGraph(grid)
        paths = plan_icbs_paths(
            G, starts=[(1, 0)], goals=[(1, 2)], silence_stdout=True
        )
        captured = capsys.readouterr()
        assert paths is not None
        # gloriyo's solver prints heavily on every node; we should see none.
        assert captured.out == ""

    def test_no_silence_lets_submodule_print(self, capsys):
        grid = [[False] * 3 for _ in range(3)]
        G = _MockGraph(grid)
        plan_icbs_paths(
            G, starts=[(1, 0)], goals=[(1, 2)], silence_stdout=False
        )
        captured = capsys.readouterr()
        # The submodule prints node generation messages by default.
        assert captured.out != ""


# ---------------------------------------------------------------------------
# Real M2M Graph smoke
# ---------------------------------------------------------------------------
class TestRealGraphSmoke:
    def test_runs_on_tiny_graph(self, tiny_graph):
        # The conftest tiny_graph is 21x14 (height x width). Two short walks.
        paths = plan_icbs_paths(
            tiny_graph,
            starts=[(17, 0), (17, 6)],
            goals=[(17, 6), (17, 0)],
            silence_stdout=True,
        )
        # Either we find a solution or the open layout has no side-step room.
        # The point of this test is just that we don't crash on a real Graph.
        if paths is not None:
            assert _collisions_in(paths) == []
            assert paths[0][0] == (17, 0)
            assert paths[1][0] == (17, 6)
