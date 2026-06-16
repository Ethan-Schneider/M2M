"""Multi-Label A* (MLA*) for two-goal MAPD path finding.

Faithful re-implementation of Algorithm 1 from
Grenouilleau, van Hoeve, Hooker, "A Multi-Label A* Algorithm for Multi-Agent
Pathfinding," ICAPS 2019, pp. 181-185
(https://ojs.aaai.org/index.php/ICAPS/article/download/3474/3342).

The single-agent search expands a path through *both* a pickup goal ``pi1``
and a delivery goal ``pi2`` in one A* sweep, keyed on a label
``ell in {1, 2}``: ``ell == 1`` means the path is still seeking the pickup,
``ell == 2`` means it has already reached the pickup and is now seeking the
delivery. The label flip happens *without a physical move* the moment the
search reaches ``pi1`` -- which is what avoids the over-constraint at the
pickup that the paper's Cases 1 and 2 describe (the agent passing through
``pi1`` even while another agent is scheduled to terminate there or pass
through it later).

Design notes (see ``M2M_project_roadmap.md`` section 3.3):
- We follow RHCR's ``StateTimeAStar`` design pattern from
  https://github.com/Jiaoyang-Li/RHCR (specifically the
  ``(state, goal_id)`` node-equality rule and the hash-on-state pattern with
  the equality predicate doing the disambiguation), but write the search
  itself from scratch in Python rather than porting C++.
- We use Python's ``heapq`` rather than RHCR's BOOST ``fibonacci_heap`` --
  M2M's per-agent path lengths are short enough that the asymptotic gap is
  irrelevant.
- Tie-breaking on equal f-values prefers the larger g-value (i.e. the node
  closer to the goal), which is the standard "deep-first on f-ties" trick.
  RHCR breaks ties randomly; the deterministic variant gives reproducible
  search trees, which is useful for tests.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


Loc = Tuple[int, int]
ReservationKey = Tuple[int, Loc]
EdgeKey = Tuple[int, Loc, Loc]


_DEFAULT_HORIZON_FACTOR = 4.0


@dataclass(order=True)
class _PQItem:
    """Priority queue entry. Ordering is on ``(f, -g, tiebreak)`` so that:

    - lower f is popped first (A* requirement);
    - among equal-f nodes, larger g is preferred (deeper search wins ties);
    - finally, an integer tiebreak ensures stable, deterministic ordering even
      across nodes with identical (f, g) values.
    """

    f_val: float
    neg_g: float  # negative g so that larger g ties get popped first
    tiebreak: int
    node_idx: int = field(compare=False)


@dataclass
class _Node:
    state: Loc
    label: int  # 1 = seeking pickup, 2 = seeking delivery
    g: int
    parent_idx: Optional[int]


def _manhattan(a: Loc, b: Loc) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ReservationTable:
    """Vertex + edge reservations against which a single-agent MLA* plans.

    Reservations are keyed on absolute simulator timesteps (the same scale as
    ``current_t`` passed to :func:`mla_star_search`), matching the paper's
    "token" abstraction. ``permanent_locs`` records, for agents whose planned
    path eventually parks them, the (loc, start_time) at which they will
    occupy ``loc`` *forever after*. This is what gives us ``t_max`` in the
    paper -- if another agent will permanently sit at ``pi1`` from time T,
    then our agent must reach ``pi1`` strictly before T.
    """

    def __init__(self) -> None:
        self._vertex: Dict[ReservationKey, int] = {}
        self._edge: Dict[EdgeKey, int] = {}
        # agent_id -> (loc, start_t): from time start_t onwards the agent
        # sits at loc indefinitely. Only one entry per agent is meaningful;
        # we keep the latest.
        self._permanent: Dict[int, Tuple[Loc, int]] = {}

    def reserve_path(
        self,
        agent_id: int,
        start_loc: Loc,
        path: List[Loc],
        start_t: int,
        is_permanent_terminal: bool = True,
    ) -> None:
        """Reserve the agent's full trajectory from ``start_t`` onwards.

        ``start_loc`` is the agent's location at time ``start_t``; ``path``
        is the sequence of *successor* locations (one per timestep), so the
        agent is at ``path[i]`` at time ``start_t + 1 + i``. If
        ``is_permanent_terminal`` is true, the agent sits at the final cell
        from ``start_t + len(path)`` onwards (the lifelong-MAPF assumption
        from the paper); set false to leave the trailing cell unreserved
        beyond the explicit path.
        """
        prev = start_loc
        self._vertex[(start_t, start_loc)] = agent_id
        for i, loc in enumerate(path):
            t = start_t + 1 + i
            self._vertex[(t, loc)] = agent_id
            # Edge reservation: forbid (loc -> prev) at the same step, which
            # would be a swap collision with our (prev -> loc) move.
            self._edge[(t, loc, prev)] = agent_id
            prev = loc
        if is_permanent_terminal:
            terminal = path[-1] if path else start_loc
            terminal_t = start_t + len(path)
            self._permanent[agent_id] = (terminal, terminal_t)

    def is_vertex_reserved(self, t: int, loc: Loc, ignore_agent: Optional[int] = None) -> bool:
        owner = self._vertex.get((t, loc))
        if owner is not None and owner != ignore_agent:
            return True
        # Permanent reservations apply for any t >= start_t.
        for ag, (perm_loc, perm_start) in self._permanent.items():
            if ag == ignore_agent:
                continue
            if perm_loc == loc and t >= perm_start:
                return True
        return False

    def is_edge_reserved(
        self, t: int, from_loc: Loc, to_loc: Loc, ignore_agent: Optional[int] = None
    ) -> bool:
        """Return ``True`` if the move ``from_loc -> to_loc`` at time ``t``
        (i.e. arriving at ``to_loc`` at ``t`` from ``from_loc`` at ``t-1``)
        would swap with another agent moving ``to_loc -> from_loc``.

        The reservation-table convention keys edge entries as
        ``(t, my_to, my_from)`` (set by :meth:`reserve_path`). A swap
        conflict therefore looks up the symmetric key
        ``(t, from_loc, to_loc)``: an entry there means *another* agent's
        ``my_to == from_loc`` and ``my_from == to_loc``, which is exactly
        the conflicting move.
        """
        owner = self._edge.get((t, from_loc, to_loc))
        return owner is not None and owner != ignore_agent

    def t_max_at(self, loc: Loc, ignore_agent: Optional[int] = None) -> Optional[int]:
        """Latest time an agent may *occupy* ``loc`` before some other agent
        permanently terminates there. ``None`` means no permanent reserver
        (i.e. ``t_max = infinity`` in paper notation).
        """
        earliest = None
        for ag, (perm_loc, perm_start) in self._permanent.items():
            if ag == ignore_agent:
                continue
            if perm_loc == loc:
                if earliest is None or perm_start < earliest:
                    earliest = perm_start
        if earliest is None:
            return None
        # Other agent occupies loc from earliest onwards, so we must arrive
        # strictly before that.
        return earliest - 1


def mla_star_search(
    G,
    start: Loc,
    pi1: Loc,
    pi2: Loc,
    reservations: ReservationTable,
    current_t: int,
    agent_id: int,
    obstacles: Optional[FrozenSet[Loc]] = None,
    horizon: Optional[int] = None,
) -> Optional[List[Loc]]:
    """Run MLA* for a single agent through pickup ``pi1`` then delivery ``pi2``.

    Args:
        G: The M2M ``Graph`` instance (used for ``get_if_obstacle`` and shape).
        start: Agent's current location at simulator time ``current_t``.
        pi1: Pickup location.
        pi2: Delivery location.
        reservations: Vertex/edge reservations from already-planned agents.
        current_t: Simulator timestep at which the agent is at ``start``.
        agent_id: Identity of the agent we're planning for, so we don't
            collide with our own (already-removed) reservations.
        obstacles: Optional pre-computed obstacle set; if None we query
            ``G.get_if_obstacle`` per neighbour.
        horizon: Maximum search depth (g-value cap). If ``None``, defaults
            to ``ceil(_DEFAULT_HORIZON_FACTOR * (h(start, pi1) + h(pi1, pi2)))``,
            which gives the search ample slack but bounds the worst case.

    Returns:
        The list of *successor* locations (one per timestep), so the agent is
        at ``result[i]`` at time ``current_t + 1 + i``. The final element is
        ``pi2``. Returns ``None`` if no feasible path exists within the
        horizon.

    Implementation notes:
        Algorithm 1 from the paper. Wait actions are explicitly enumerated
        as a self-loop neighbour (``(state, label) -> (state, label)`` at
        ``g + 1``). Edge collisions are checked via the reservation table's
        ``is_edge_reserved``. Vertex collisions check the agent's intended
        position at the *destination* timestep ``g + 1``.

        The label flip is a *zero-cost transition*: when ``ell == 1`` and
        ``state == pi1`` we generate a sibling node with the same ``state``
        and ``g`` but ``ell == 2``, push it to the heap, and *do not* expand
        the ``ell == 1`` node further (the node "becomes" its successor).
        This matches the paper's Algorithm 1 verbatim; if we expanded the
        ``ell == 1`` node first, the search would unnecessarily reserve
        ``pi1`` for the ``ell == 1`` agent path beyond the pickup tick.
    """
    h_total = _manhattan(start, pi1) + _manhattan(pi1, pi2)
    if horizon is None:
        horizon = max(8, int(_DEFAULT_HORIZON_FACTOR * h_total) + 4)

    t_max_pickup = reservations.t_max_at(pi1, ignore_agent=agent_id)

    nodes: List[_Node] = []
    open_heap: List[_PQItem] = []
    # closed key is (state, label, g). Using g in the key (rather than just
    # (state, label)) keeps the search space fully time-extended, which is
    # required for wait actions and for revisiting cells while another agent
    # passes through. We retire (state, label, g) once popped so the same
    # (state, label) at the same time isn't expanded twice.
    closed: set = set()
    # Best g-value seen for (state, label); used to prune duplicates pushed
    # before the optimal one is popped.
    best_g: Dict[Tuple[Loc, int], int] = {}

    tiebreak_counter = 0

    def push(state: Loc, label: int, g: int, parent_idx: Optional[int]) -> None:
        nonlocal tiebreak_counter
        key = (state, label)
        prev_g = best_g.get(key)
        if prev_g is not None and prev_g <= g:
            # Already enqueued at an equal-or-better g; new node is dominated.
            return
        best_g[key] = g
        if label == 1:
            h = _manhattan(state, pi1) + _manhattan(pi1, pi2)
        else:
            h = _manhattan(state, pi2)
        f = g + h
        idx = len(nodes)
        nodes.append(_Node(state=state, label=label, g=g, parent_idx=parent_idx))
        heapq.heappush(
            open_heap,
            _PQItem(f_val=f, neg_g=-g, tiebreak=tiebreak_counter, node_idx=idx),
        )
        tiebreak_counter += 1

    push(start, 1, 0, None)

    def is_obstacle(loc: Loc) -> bool:
        if obstacles is not None:
            return loc in obstacles
        rows, cols = G.get_graph_size()
        if loc[0] < 0 or loc[0] >= rows or loc[1] < 0 or loc[1] >= cols:
            return True
        return G.get_if_obstacle(loc)

    while open_heap:
        item = heapq.heappop(open_heap)
        node = nodes[item.node_idx]
        closed_key = (node.state, node.label, node.g)
        if closed_key in closed:
            continue
        closed.add(closed_key)

        # Algorithm 1 line: "if ell == 1 and g_n > t_max then continue"
        if (
            node.label == 1
            and t_max_pickup is not None
            and node.g > t_max_pickup
        ):
            continue

        # Algorithm 1 line: "if ell == 1 and p_n == pi1 then create n' with
        # same g and label 2; add to Q". We do *not* expand n further here.
        if node.label == 1 and node.state == pi1:
            push(node.state, 2, node.g, item.node_idx)
            continue

        # Algorithm 1 line: "if ell == 2 and p_n == pi2 then return path".
        if node.label == 2 and node.state == pi2:
            return _reconstruct(nodes, item.node_idx, start)

        if node.g >= horizon:
            continue

        # Expand: 4 cardinal neighbours + wait. The simulator's grid uses
        # (row, col); we don't lean on G.get_neighbors here because that
        # filters by *current* occupancy rather than time-extended
        # reservations -- we need full control over what gets pruned.
        absolute_t = current_t + node.g + 1
        nbrs: List[Loc] = [
            (node.state[0] - 1, node.state[1]),
            (node.state[0] + 1, node.state[1]),
            (node.state[0], node.state[1] - 1),
            (node.state[0], node.state[1] + 1),
            node.state,  # wait
        ]
        for nb in nbrs:
            if is_obstacle(nb):
                continue
            if reservations.is_vertex_reserved(absolute_t, nb, ignore_agent=agent_id):
                continue
            # Edge-swap collision: our move ``node.state -> nb`` at
            # ``absolute_t`` swaps with another agent moving ``nb -> node.state``
            # at the same step. The reservation-table convention keys edge
            # reservations as ``(t, my_to, my_from)``, so to detect a swap we
            # ask whether ``(absolute_t, node.state, nb)`` is reserved -- that
            # entry would belong to an agent whose ``my_to == node.state`` and
            # ``my_from == nb``, which is exactly the conflicting swap.
            if nb != node.state and reservations.is_edge_reserved(
                absolute_t, node.state, nb, ignore_agent=agent_id
            ):
                continue
            push(nb, node.label, node.g + 1, item.node_idx)

    return None


def _reconstruct(nodes: List[_Node], goal_idx: int, start: Loc) -> List[Loc]:
    """Walk parent pointers from ``goal_idx`` back to the root, then strip
    the start cell and intermediate label-flip stubs.

    The label flip emits a child node with the same state and g as its
    parent (``state == pi1`` at the moment of flip). Those duplicates would
    pad the returned path with a no-op step at the pickup; we collapse runs
    of consecutive identical states *only when they straddle the flip*,
    which we detect by a parent-child pair whose g-value is identical (a
    proper move increments g, the flip does not).

    Special case (start == pi1). When the agent is already at the pickup
    when the search begins, the label flip happens at i=1 with both nodes
    sharing the start cell. Replacing the start node would drop pi1 out
    of the final ``pruned[1:]`` slice, which means the caller's
    ``agent.path_sequence`` would never contain pi1; the M2M simulator
    only flips ``status=1 -> 2`` when ``agent.state == pickup`` *after a
    path advance*, so the agent would walk straight past the pickup
    without registering it and stay stuck in status=1 forever (this was
    observed in 30-bot study_small_restricted runs producing complete
    deadlock by ~tick 600). To preserve correctness in that case we
    *append* the flipped node rather than replacing, which yields a path
    that begins with pi1 as an explicit one-step "wait at pickup" so the
    simulator sees the pickup and fires the transition. The non-trivial
    label flips deeper in the path are still collapsed because the parent
    we'd be replacing is the actual movement-into-pi1 node, which is
    redundant with the immediately-following flipped node at the same g.
    """
    chain: List[_Node] = []
    idx: Optional[int] = goal_idx
    while idx is not None:
        chain.append(nodes[idx])
        idx = nodes[idx].parent_idx
    chain.reverse()  # root-first
    pruned: List[_Node] = [chain[0]]
    for i in range(1, len(chain)):
        if chain[i].g == chain[i - 1].g:
            # Label flip. Normally collapse to keep the path tight; but if
            # the parent we'd be replacing is the start cell (i==1) and
            # both nodes share state, the flipped node IS the pickup and
            # dropping it via ``pruned[1:]`` later would silently break
            # the simulator's pickup transition. Append in that one case.
            if i == 1 and chain[0].state == chain[1].state:
                pruned.append(chain[i])
            else:
                pruned[-1] = chain[i]
        else:
            pruned.append(chain[i])
    return [n.state for n in pruned[1:]]


def mla_star_single_goal(
    G,
    start: Loc,
    goal: Loc,
    reservations: ReservationTable,
    current_t: int,
    agent_id: int,
    obstacles: Optional[FrozenSet[Loc]] = None,
    horizon: Optional[int] = None,
) -> Optional[List[Loc]]:
    """Single-goal degenerate variant used by HBH's endpoint-clearing step.

    Strictly speaking the paper's HBH only invokes MLA* with two goals (a
    pickup followed by a delivery); the endpoint-clearing step just says
    "move to the closest free endpoint" without specifying which planner to
    use. We use the single-label form of the same algorithm so that endpoint
    moves see and respect the same reservation table (no double-booking).
    """
    h = _manhattan(start, goal)
    if horizon is None:
        horizon = max(8, int(_DEFAULT_HORIZON_FACTOR * h) + 4)

    nodes: List[_Node] = []
    open_heap: List[_PQItem] = []
    closed: set = set()
    best_g: Dict[Tuple[Loc, int], int] = {}
    tiebreak_counter = 0

    def push(state: Loc, g: int, parent_idx: Optional[int]) -> None:
        nonlocal tiebreak_counter
        key = (state, 2)
        prev_g = best_g.get(key)
        if prev_g is not None and prev_g <= g:
            return
        best_g[key] = g
        f = g + _manhattan(state, goal)
        idx = len(nodes)
        nodes.append(_Node(state=state, label=2, g=g, parent_idx=parent_idx))
        heapq.heappush(
            open_heap,
            _PQItem(f_val=f, neg_g=-g, tiebreak=tiebreak_counter, node_idx=idx),
        )
        tiebreak_counter += 1

    def is_obstacle(loc: Loc) -> bool:
        if obstacles is not None:
            return loc in obstacles
        rows, cols = G.get_graph_size()
        if loc[0] < 0 or loc[0] >= rows or loc[1] < 0 or loc[1] >= cols:
            return True
        return G.get_if_obstacle(loc)

    push(start, 0, None)
    while open_heap:
        item = heapq.heappop(open_heap)
        node = nodes[item.node_idx]
        ck = (node.state, node.label, node.g)
        if ck in closed:
            continue
        closed.add(ck)
        if node.state == goal:
            return _reconstruct(nodes, item.node_idx, start)
        if node.g >= horizon:
            continue
        absolute_t = current_t + node.g + 1
        nbrs: List[Loc] = [
            (node.state[0] - 1, node.state[1]),
            (node.state[0] + 1, node.state[1]),
            (node.state[0], node.state[1] - 1),
            (node.state[0], node.state[1] + 1),
            node.state,
        ]
        for nb in nbrs:
            if is_obstacle(nb):
                continue
            if reservations.is_vertex_reserved(absolute_t, nb, ignore_agent=agent_id):
                continue
            if nb != node.state and reservations.is_edge_reserved(
                absolute_t, node.state, nb, ignore_agent=agent_id
            ):
                continue
            push(nb, node.g + 1, item.node_idx)
    return None
