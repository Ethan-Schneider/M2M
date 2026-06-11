"""Shared pytest fixtures for the M2M test suite.

Fixtures here are conservative and match the *current* code on the
``symbotic_2026`` branch. As we land features from `local_task_reallocation`
(split inbound/outbound driveways, schedule-driven generation, rearrangement
task types), we'll grow these fixtures in lockstep — adding new ones rather
than mutating existing ones, so older tests keep passing while we iterate.

Usage
-----
    def test_something(tiny_graph, empty_inventory, sample_agents):
        ...

Fixtures
--------
- ``repo_root``         absolute path to the M2M repo root
- ``small_test_map``    path to ``data/maps/small_test`` (tiny 21x14 map)
- ``seeded_rng``        seeds numpy + Python ``random``; returns the seed used
- ``tiny_graph``        ``Graph`` built from ``small_test_map`` with 3 robots
- ``empty_inventory``   ``Inventory`` over the tiny graph aisles, 0% fill
- ``filled_inventory``  ``Inventory`` over the tiny graph aisles, 50% fill
- ``minimal_stats``     ``Stats`` with only required ctor args; output goes to ``tmp_path``
- ``sample_agents``     ``AgentLoader`` with 3 agents on the tiny graph
- ``sample_tasks``      ``Dict[int, Tuple]`` containing one IB and one OB task
"""

from __future__ import annotations

import os
import random as _stdrandom
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
# The pytest config in pyproject.toml already adds the repo root to PYTHONPATH
# but importing from inside conftest happens before that fixture sees the path,
# so we make absolutely sure here.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GT_grid_world.src.agent import Agent, AgentLoader  # noqa: E402
from GT_grid_world.src.analysis.statistics import Stats  # noqa: E402
from GT_grid_world.src.graph import Graph  # noqa: E402
from GT_grid_world.src.inventory_manager.inventory import Inventory  # noqa: E402
from GT_grid_world.src.inventory_manager.item import WeightInitialization  # noqa: E402


# ---------------------------------------------------------------------------
# Path / data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the M2M repository root."""
    return _REPO_ROOT


@pytest.fixture(scope="session")
def small_test_map(repo_root: Path) -> Path:
    """Path to the tiny ``small_test`` map (21x14, 3 robots, single driveway).

    We pin to ``small_test`` deliberately because it is the smallest map
    that exercises the full aisle/driveway/robot layout and lets unit tests
    run in well under a second. Larger ``study_*`` maps trigger expensive
    distance-matrix computation on first load.
    """
    path = repo_root / "data" / "maps" / "small_test"
    if not path.exists():
        pytest.skip(f"map {path} not present in repo")
    return path


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
@pytest.fixture()
def seeded_rng() -> int:
    """Seed numpy and the stdlib ``random`` module, return the seed.

    We don't return a ``Generator`` instance because the M2M code pulls from
    the global ``np.random`` state (e.g. ``np.random.choice`` in CRG). Returning
    the seed lets a test note the seed in failure messages or repro recipes.
    """
    seed = 42
    np.random.seed(seed)
    _stdrandom.seed(seed)
    return seed


# ---------------------------------------------------------------------------
# Graph / inventory
# ---------------------------------------------------------------------------
@pytest.fixture()
def tiny_graph(small_test_map: Path, seeded_rng: int) -> Graph:
    """A ``Graph`` instance built from the small_test map.

    Function-scoped so each test gets a fresh graph (mutation by pickup/dropoff
    in one test would otherwise leak into the next). The first call computes a
    distance matrix and caches it as a sibling ``*_distances.npy`` file.

    ``initial_warehouse_capacity=0.0`` so existing tests that assume an empty
    warehouse keep their behaviour. Tests that need a populated warehouse
    should use ``populated_graph`` instead.
    """
    return Graph(
        num_robots=3,
        file_name=str(small_test_map),
        initial_warehouse_capacity=0.0,
        num_skus=5,
        weight_init_method="uniform",
    )


@pytest.fixture()
def populated_graph(small_test_map: Path, seeded_rng: int) -> Graph:
    """A ``Graph`` whose warehouse is 30% pre-filled with SKUs.

    Use this fixture for tests that need both warehouse-full and
    warehouse-empty cells (e.g. shuffle / shelf-to-shelf, cost-tensor
    construction). 30% leaves enough room on both sides of the partition
    that allocator-style assertions about non-empty start_locs and
    non-empty goal_locs hold reliably.
    """
    return Graph(
        num_robots=3,
        file_name=str(small_test_map),
        initial_warehouse_capacity=30.0,
        num_skus=5,
        weight_init_method="uniform",
    )


@pytest.fixture()
def empty_inventory(tiny_graph: Graph) -> Inventory:
    """Empty ``Inventory`` over the tiny graph's aisle locations."""
    return Inventory(
        num_skus=5,
        warehouse_locations=tiny_graph.aisle_locations,
        fill_percentage=0.0,
        weight_init=WeightInitialization.UNIFORM,
    )


@pytest.fixture()
def filled_inventory(tiny_graph: Graph, seeded_rng: int) -> Inventory:
    """50%-filled ``Inventory`` over the tiny graph's aisle locations."""
    return Inventory(
        num_skus=5,
        warehouse_locations=tiny_graph.aisle_locations,
        fill_percentage=50.0,
        weight_init=WeightInitialization.UNIFORM,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@pytest.fixture()
def minimal_stats(tmp_path: Path) -> Stats:
    """``Stats`` with only the five required constructor args.

    ``Stats`` accepts ~30 parameters, all but five default to ``None``. We
    pass just the required ones plus a ``tmp_path`` output file so each test
    is hermetic. Writers in ``Stats`` that need an optional config key will
    surface as ``None``-related ``AttributeError`` / ``TypeError`` and that is
    intentional: it tells us the test needs to set that key explicitly.
    """
    return Stats(
        num_robots=3,
        simulation_time=100,
        output_file=str(tmp_path / "stats.json"),
        map_name="small_test",
        cost_calculation_method="manhattan",
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_agents(tiny_graph: Graph) -> AgentLoader:
    """An ``AgentLoader`` with 3 agents at known positions on the tiny graph.

    Agents are placed at the first three robot start positions detected by
    ``Graph.__load_graph``. We don't read those positions back from ``Graph``
    (they're private) so we instead pick three open ``.`` cells that the
    ``small_test`` map keeps available below the aisles. If you change
    ``small_test_map`` you may need to update these coordinates.
    """
    # small_test layout: rows 17-19 are open '.' and 'r' rows; safe spots:
    starts = [(17, 0), (17, 6), (17, 13)]
    agents = [Agent(agent_id=i, state=loc) for i, loc in enumerate(starts)]
    return AgentLoader(agents)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_tasks(tiny_graph: Graph, filled_inventory: Inventory) -> dict:
    """A small ``Dict[int, Tuple]`` of task tuples.

    Task tuple shape (current branch): ``(start_locs, goal_locs, deadline,
    sku_id, type)`` with ``type`` in ``{0=outbound, 1=inbound}``.
    """
    aisles = tiny_graph.get_aisle_locations()
    stations = tiny_graph.get_station_locations()
    full = filled_inventory.get_full_locations()

    tasks: dict = {}
    # Outbound: pick up from any full aisle, drop at any station
    if full:
        tasks[1] = (
            frozenset(full),
            frozenset(stations),
            100,    # deadline
            0,      # sku_id
            0,      # type=outbound
        )
    # Inbound: pick up at any station, drop at any empty aisle
    empties = [loc for loc in aisles if loc not in full]
    if empties and stations:
        tasks[2] = (
            frozenset(stations),
            frozenset(empties),
            100,
            0,
            1,      # type=inbound
        )
    return tasks
