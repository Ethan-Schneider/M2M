"""ICBS planner wrapper around the gloriyo/MAPF-ICBS submodule.

Adapts M2M's ``Graph`` / coordinate conventions to the
:mod:`icbs_complete` solver from the public ``gloriyo/MAPF-ICBS``
repository (vendored as a git submodule under
``GT_grid_world/src/path_finding_algorithms/external_algorithms/MAPF-ICBS``).

This wrapper exists because TA-Hybrid (Liu, Ma, Li, Koenig, AAMAS 2019)
uses ICBS (Boyarski et al., IJCAI 2015) as its Group-1 path planner
(paper Section 5.2). The original TA-Hybrid code was not released and the
original ICBS code is C# only; ``gloriyo`` is the closest paper-faithful
public Python implementation we located. See
``external_algorithms/MAPF-ICBS/THIRD_PARTY.md`` for licensing provenance.

Key features
------------

- **External reservation support.** TA-Hybrid's "reserving dummy paths"
  mechanism (paper Section 5.2) is enforced by injecting external
  reservations as ICBS constraints on the root node of the conflict tree.
  We avoid modifying the submodule's source by monkey-patching the
  ``A_Star`` low-level class for the duration of the solve so that every
  internal replan automatically sees our extra constraints in addition to
  whatever the CBS layer generated.

- **stdout silencing.** The submodule prints diagnostic information on
  *every* CBS node generation / expansion, which would flood the M2M log
  on realistic problem sizes. We redirect stdout to an in-memory buffer
  during the solve. ``silence_stdout=False`` is provided for debugging.

- **No upstream modifications.** This wrapper does not edit a single byte
  of the ``gloriyo`` source. Adapters live entirely on the M2M side.

Time / coordinate conventions
-----------------------------

- ``my_map[row][col]`` is ``True`` if (row, col) is an obstacle, ``False``
  otherwise. Same convention as gloriyo and as M2M's ``Graph``.
- All times in ``vertex_reservations`` / ``edge_reservations`` are
  **CBS-relative**: a path returned by :func:`plan_icbs_paths` has
  ``paths[a][t]`` at timestep ``t`` measured from the start of the solve
  (``t = 0`` corresponds to ``starts[a]``). The caller is responsible
  for shifting between absolute simulation time and CBS-relative time.
- An *edge reservation* ``(loc_from, loc_to, t)`` means "no agent may
  traverse from ``loc_from`` (at time ``t - 1``) to ``loc_to`` (at time
  ``t``)". This matches the gloriyo constraint convention (the constraint
  is keyed by the **arrival** timestep).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..graph import Graph
from ..logging_config import get_logger


_LOG = get_logger("path.icbs")

Loc = Tuple[int, int]

_SUBMODULE_CODE_DIR = (
    Path(__file__).parent / "external_algorithms" / "MAPF-ICBS" / "code"
)


# ---------------------------------------------------------------------------
# Submodule loading
# ---------------------------------------------------------------------------
def _import_icbs():
    """Lazily load the ``icbs_complete`` module from the submodule.

    The submodule's files use bare ``from a_star_class import A_Star``-style
    imports (no package qualifiers), so we have to add its ``code/``
    directory to ``sys.path``. We do this exactly once on first call. The
    module object is then cached by Python's normal import machinery.
    """
    code_dir = str(_SUBMODULE_CODE_DIR)
    if code_dir not in sys.path:
        sys.path.append(code_dir)
    import icbs_complete  # type: ignore[import-not-found]
    return icbs_complete


# ---------------------------------------------------------------------------
# M2M -> gloriyo adaptation
# ---------------------------------------------------------------------------
def graph_to_my_map(G: Graph) -> List[List[bool]]:
    """Convert an M2M ``Graph`` to the 2D boolean obstacle map gloriyo expects.

    ``gloriyo`` expects ``my_map[row][col] == True`` iff ``(row, col)`` is an
    obstacle. M2M's :meth:`Graph.get_if_obstacle` returns exactly that for
    any in-bounds cell.
    """
    return [
        [G.get_if_obstacle((r, c)) for c in range(G.width)]
        for r in range(G.height)
    ]


def path_to_vertex_reservations(
    path: Sequence[Loc],
    start_time: int = 0,
    extend_to: Optional[int] = None,
) -> List[Tuple[Loc, int]]:
    """Convert a planned path into ``(loc, time)`` vertex reservations.

    Used by TA-Hybrid's Algorithm 2 to translate dummy paths (planned via
    :func:`mla_star_single_goal` or any other single-agent planner) into
    reservations that :func:`plan_icbs_paths` will forbid for *other*
    Group-1 agents.

    Args:
        path: The path to reserve; e.g. ``[(r0, c0), (r1, c1), ...]``.
        start_time: CBS-relative time at which ``path[0]`` is occupied.
            Defaults to ``0``.
        extend_to: If given, the *last* cell of ``path`` is additionally
            reserved at every timestep in ``[start_time + len(path),
            extend_to]`` (inclusive) -- this models "agent waits forever
            at its parking location" semantics for dummy-path tails.

    Returns:
        List of ``(loc, time)`` tuples suitable as
        ``vertex_reservations`` to :func:`plan_icbs_paths`.
    """
    if not path:
        return []
    out: List[Tuple[Loc, int]] = []
    for i, cell in enumerate(path):
        out.append((tuple(cell), start_time + i))
    if extend_to is not None:
        last_t = start_time + len(path) - 1
        last_cell: Loc = tuple(path[-1])  # type: ignore[assignment]
        for t in range(last_t + 1, int(extend_to) + 1):
            out.append((last_cell, t))
    return out


def path_to_edge_reservations(
    path: Sequence[Loc],
    start_time: int = 0,
) -> List[Tuple[Loc, Loc, int]]:
    """Convert a planned path's consecutive transitions into edge reservations.

    For each non-wait transition ``path[i] -> path[i+1]`` we emit a reservation
    at arrival time ``start_time + i + 1`` (the gloriyo edge-constraint
    convention is keyed by the *arrival* timestep). Wait steps -- where
    ``path[i] == path[i + 1]`` -- are skipped because edge reservations
    only make sense for genuine movement.

    Args:
        path: The path to reserve.
        start_time: CBS-relative time at which ``path[0]`` is occupied.

    Returns:
        List of ``(loc_from, loc_to, arrival_time)`` triples suitable as
        ``edge_reservations`` to :func:`plan_icbs_paths`.
    """
    out: List[Tuple[Loc, Loc, int]] = []
    for i in range(len(path) - 1):
        a = tuple(path[i])
        b = tuple(path[i + 1])
        if a != b:
            out.append((a, b, start_time + i + 1))
    return out


def reservations_to_constraints(
    vertex_reservations: Sequence[Tuple[Loc, int]],
    edge_reservations: Sequence[Tuple[Loc, Loc, int]],
    num_agents: int,
) -> List[dict]:
    """Translate external reservations into gloriyo per-agent constraint dicts.

    Each external reservation is replicated as a negative constraint for
    every agent in the planning batch (``range(num_agents)``). This mirrors
    the paper's "reserving dummy paths" semantics: a reserved cell at time
    ``t`` is forbidden for *every* replanning agent, regardless of which
    agent owns the reservation.

    Constraint dict shape (per gloriyo's
    :mod:`a_star_class` / :mod:`single_agent_planner`):

    -  ``agent``       -- target agent index (one constraint per agent).
    -  ``meta_agent``  -- singleton set ``{agent}`` so it survives the
       meta-agent merge-and-restart pass in :mod:`icbs_complete`.
    -  ``loc``         -- ``[(row, col)]`` for a vertex constraint;
       ``[(r1, c1), (r2, c2)]`` for an edge constraint.
    -  ``timestep``    -- **arrival** time for edge constraints; the time
       the agent must avoid the cell for vertex constraints.
    -  ``positive``    -- ``False`` (negative / forbidding) for external
       reservations.

    Args:
        vertex_reservations: Iterable of ``(loc, timestep)`` tuples to
            forbid; e.g. dummy-path cells from already-planned agents.
        edge_reservations: Iterable of ``(loc_from, loc_to, timestep)``
            tuples; ``timestep`` is when the agent would *arrive* at
            ``loc_to``.
        num_agents: Number of agents in the current planning batch (i.e.
            ``len(starts)``). Every reservation is duplicated for every
            agent in ``range(num_agents)``.
    """
    constraints: List[dict] = []
    for loc, t in vertex_reservations:
        for agent_id in range(num_agents):
            constraints.append({
                "agent": agent_id,
                "meta_agent": {agent_id},
                "loc": [tuple(loc)],
                "timestep": int(t),
                "positive": False,
            })
    for loc_from, loc_to, t in edge_reservations:
        for agent_id in range(num_agents):
            constraints.append({
                "agent": agent_id,
                "meta_agent": {agent_id},
                "loc": [tuple(loc_from), tuple(loc_to)],
                "timestep": int(t),
                "positive": False,
            })
    return constraints


# ---------------------------------------------------------------------------
# Monkey-patch context manager
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _augmented_astar(icbs_complete, extra_constraints: List[dict]):
    """Monkey-patch ``icbs_complete.A_Star`` to prepend ``extra_constraints``.

    The gloriyo conflict-tree builds the root with ``constraints = []``
    (hard-coded inline) and then re-instantiates :class:`A_Star` at every
    expansion with whatever constraints the CT node carries. By replacing
    ``icbs_complete.A_Star`` with a subclass that augments the constraint
    list at ``__init__`` time, we ensure every replan -- including the
    initial root paths -- respects our external reservations, without
    forking the upstream source.

    This patch is per-process and not thread-safe. M2M's solver pipeline
    is single-threaded so this is fine; if that ever changes we will need
    to gate this behind a lock.

    Args:
        icbs_complete: The imported ``icbs_complete`` module object.
        extra_constraints: Per-agent constraint dicts to inject. May be
            empty (in which case this is a no-op patch).
    """
    if not extra_constraints:
        yield
        return

    original = icbs_complete.A_Star

    class _AugmentedAStar(original):  # type: ignore[misc, valid-type]
        def __init__(self, my_map, starts, goals, heuristics, agents, contraints):
            # Note: ``contraints`` (sic) is gloriyo's parameter name; we
            # preserve the spelling for minimal surface area.
            merged = list(contraints) + list(extra_constraints)
            super().__init__(my_map, starts, goals, heuristics, agents, merged)

    icbs_complete.A_Star = _AugmentedAStar
    try:
        yield
    finally:
        icbs_complete.A_Star = original


# ---------------------------------------------------------------------------
# Public planner entry point
# ---------------------------------------------------------------------------
def plan_icbs_paths(
    G: Graph,
    starts: Sequence[Loc],
    goals: Sequence[Loc],
    vertex_reservations: Sequence[Tuple[Loc, int]] = (),
    edge_reservations: Sequence[Tuple[Loc, Loc, int]] = (),
    disjoint_splitting: bool = False,
    silence_stdout: bool = True,
) -> Optional[List[List[Loc]]]:
    """Plan collision-free paths for a batch of agents via ICBS.

    Used as the Group-1 path planner inside TA-Hybrid's Algorithm 2
    (PlanPathsToDelivery). The caller is responsible for translating its
    own simulation-time reservations into CBS-relative timesteps (where
    ``t = 0`` corresponds to ``starts[*]``).

    Args:
        G: M2M ``Graph`` -- used only via :meth:`get_if_obstacle` and
            ``height`` / ``width`` attributes to render the obstacle map.
        starts: One ``(row, col)`` start cell per agent.
        goals:  One ``(row, col)`` goal cell per agent, in the same order
            as ``starts``.
        vertex_reservations: Per-cell time-extended reservations to honour.
            Each element ``(loc, t)`` forbids any planning agent from being
            at ``loc`` at CBS-relative time ``t``. Typically supplied by
            the caller from "dummy paths" of other agents already planned
            in earlier batches.
        edge_reservations: Per-edge reservations of the same form as
            vertex reservations, but for two-cell transitions.
        disjoint_splitting: If ``True``, use disjoint splitting (Li et al.,
            ICAPS 2021) instead of textbook ICBS standard splitting. The
            paper TA-Hybrid is based on uses standard splitting; default
            ``False`` matches that for paper fidelity.
        silence_stdout: If ``True`` (default), the submodule's verbose
            ``print()`` calls are swallowed. Set ``False`` only for
            interactive debugging.

    Returns:
        List of paths in gloriyo format (``paths[a][t]`` = ``(row, col)``
        of agent ``a`` at timestep ``t``, with ``paths[a][0] == starts[a]``
        and ``paths[a][-1] == goals[a]``), or ``None`` if the solver
        could not find a solution (no path, infeasible constraints, or
        the upstream node-generation cap of 50,000 was reached).

    Raises:
        ValueError: if ``len(starts) != len(goals)``.
    """
    if len(starts) != len(goals):
        raise ValueError(
            f"len(starts)={len(starts)} != len(goals)={len(goals)}; "
            "each agent needs exactly one start and one goal"
        )

    icbs_complete = _import_icbs()
    my_map = graph_to_my_map(G)
    constraints = reservations_to_constraints(
        vertex_reservations, edge_reservations, num_agents=len(starts)
    )

    out_buf = io.StringIO() if silence_stdout else None
    stdout_ctx = (
        contextlib.redirect_stdout(out_buf)
        if silence_stdout
        else contextlib.nullcontext()
    )

    with _augmented_astar(icbs_complete, constraints), stdout_ctx:
        try:
            solver = icbs_complete.ICBS_Solver(my_map, list(starts), list(goals))
            result = solver.find_solution(disjoint=disjoint_splitting)
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except BaseException as exc:  # pragma: no cover - depends on upstream behaviour
            # gloriyo raises BaseException('No solutions') from the low-level
            # A* failure path; we swallow it as "no plan available".
            _LOG.warning("ICBS_Solver raised %s: %s", type(exc).__name__, exc)
            return None

    if result is None:
        _LOG.warning("ICBS_Solver returned None (node-generation cap hit)")
        return None

    # gloriyo returns (paths, num_generated, num_expanded) on success.
    paths, n_generated, n_expanded = result
    _LOG.info(
        "ICBS solved %d-agent instance: generated=%d expanded=%d",
        len(starts), n_generated, n_expanded,
    )
    return paths
